"""开环循迹验证脚本。

通过 Web 推流提供可视化，人工确认每一步：
    矩形检测 → 标定(可选) → 归零 → 开环循迹

开环控制: 标定矩阵 M⁻¹ + CALIB_SPEED 盲算每段时间和速度，
两轴同时匀速，激光在像素空间走直线。

用法:
    python3 sample/main_test.py
    浏览器访问 http://<ip>:5000 查看实时画面
"""

import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from calibration import Calibrator, load_calibration, save_calibration
from camera import Camera
from gpio_keys import KeypadController
from motor import MotorController
from rectangle import RectangleManager
from screen_cal import ScreenCalibrator
from tracker import process_laser_detection
from uart import UartController
from web_stream import MjpegStream

# ===== 配置 =====
CAN_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
CAN_BAUD = 4000000
CALIB_SPEED = 2.0       # °/s, 与 calibration.py 一致
CALIB_FILE = "calibration.json"

# ===== GPIO 按键映射 =====
PIN_MAP = {
    # Phase 1 屏幕标定
    5:  "enter",    # 记录当前点
    27: "undo",     # 撤销上一点
    24: "q",        # 取消退出
    # Phase 2/3 复用 enter + 主菜单按键
    # Phase 4 主菜单
    6:  "1",        # 复位到屏幕中心
    26: "2",        # 绕矩形循迹
    17: "3",        # 绕屏幕循迹
    25: "r",        # 重新检测矩形
    23: "c",        # 重新云台标定
    # GPIO27 在 Phase 4 复用为 "s" (重新屏幕标定), Phase 1 为 "undo"
    # GPIO16, GPIO22 预留，不接线
}

# ===== 全局状态 (供 frame_provider) =====
_cam: Camera | None = None
_rect_mgr: RectangleManager | None = None
_calibrator: Calibrator | None = None
_screen_corners: list[tuple[float, float]] = []   # 屏幕 4 角点 (实时渲染)
_screen_done: list[bool] = [False]                 # 标定完成标志 (可变引用)
_screen_laser_pos: list = [None]                    # 实时激光位置 (可变引用)
_rect_center_display: list = [None]               # 矩形中心显示 (可变引用)


def _frame_provider() -> bytes | None:
    """MjpegStream 回调：分阶段渲染画面。"""
    if _cam is None:
        return None
    frame = _cam.read()
    if frame is None:
        return None

    # 标定模式：显示标定轨迹
    if _calibrator is not None:
        annotated = _calibrator.annotate(frame)
        ok, buf = cv2.imencode(".jpg", annotated)
        return buf.tobytes() if ok else None

    # 矩形标注
    if _rect_mgr is not None:
        annotated = _rect_mgr.annotate(frame)
    else:
        annotated = frame

    # 屏幕角点标注
    if _screen_corners:
        # 已完成标定 → 画完整四边形轮廓 (蓝色)
        if _screen_done[0] and len(_screen_corners) == 4:
            pts = np.array(_screen_corners + [_screen_corners[0]], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (255, 128, 0), 2)
        # 标定中 → 画已记录的点 (红圈) + 进度线
        for i, (x, y) in enumerate(_screen_corners):
            xi, yi = int(x), int(y)
            cv2.circle(annotated, (xi, yi), 8, (0, 0, 255), -1)
            cv2.putText(annotated, f"P{i}", (xi + 10, yi - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        if len(_screen_corners) >= 3 and not _screen_done[0]:
            pts = np.array(_screen_corners + [_screen_corners[0]], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (128, 128, 255), 1)

    # 标定中 → 画实时激光位置 (绿色十字，独立于角点数量)
    if not _screen_done[0] and _screen_laser_pos[0] is not None:
        lx, ly = int(_screen_laser_pos[0][0]), int(_screen_laser_pos[0][1])
        cv2.drawMarker(annotated, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(annotated, f"({lx},{ly})", (lx + 15, ly - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 品红色斜十字 — 矩形中心（用于验证复位精度）
    if _rect_center_display[0] is not None:
        rx, ry = int(_rect_center_display[0][0]), int(_rect_center_display[0][1])
        cv2.drawMarker(annotated, (rx, ry), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.putText(annotated, "C", (rx + 15, ry - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    ok, buf = cv2.imencode(".jpg", annotated)
    return buf.tobytes() if ok else None


def calc_move(
    origin: tuple[float, float],
    target: tuple[float, float],
    M_inv: np.ndarray,
    speed: float,
) -> tuple[float, float, float] | None:
    """计算开环移动参数。

    返回: (az_speed, pt_speed, duration) 或 None（距离为0）
    """
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    dpixel = np.array([dx, dy])

    dtheta = M_inv @ dpixel
    d_az = float(dtheta[0])
    d_pt = float(dtheta[1])

    t_az = abs(d_az) / speed if speed > 0 else 0.0
    t_pt = abs(d_pt) / speed if speed > 0 else 0.0
    t_total = max(t_az, t_pt)

    if t_total == 0:
        return None

    return (d_az / t_total, d_pt / t_total, t_total)


def execute_move(
    motor: MotorController,
    move_params: tuple[float, float, float],
) -> None:
    """执行开环移动。"""
    az_speed, pt_speed, duration = move_params
    print(
        f"  az={az_speed:+.2f}°/s  pt={pt_speed:+.2f}°/s  t={duration:.2f}s"
    )
    motor.set_speed(az_speed, pt_speed)
    time.sleep(duration)
    motor.stop()
    time.sleep(0.3)


def detect_laser(cam: Camera) -> tuple[float, float] | None:
    """检测激光当前位置。"""
    for _ in range(5):
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pos, _ = process_laser_detection(frame)
        if pos is not None:
            return pos
        time.sleep(0.05)
    return None


def detect_laser_single(cam: Camera) -> tuple[float, float] | None:
    """单帧检测激光，无重试。用于视觉矫正的快速轮询。"""
    frame = cam.read()
    if frame is None:
        return None
    pos, _ = process_laser_detection(frame)
    return pos


def project_to_edge(
    pos: tuple[float, float],
    P_start: tuple[float, float],
    P_end: tuple[float, float],
) -> tuple[float, float]:
    """计算激光点到线段上的最近点（垂足）。"""
    ab = np.array(P_end) - np.array(P_start)
    ap = np.array(pos) - np.array(P_start)
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return P_start
    t = np.dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    result = np.array(P_start) + t * ab
    return (float(result[0]), float(result[1]))


def execute_move_with_correction(
    motor: MotorController,
    cam: Camera,
    move_params: tuple[float, float, float],
    P_start: tuple[float, float],
    P_end: tuple[float, float],
    M_inv: np.ndarray,
    check_interval: float = 0.1,
    correction_gain: float = 0.35,
) -> None:
    """开环移动 + 视觉矫正叠加。

    每 check_interval 秒检测激光位置：
    - 检测成功 → 计算到当前边的垂足 → 垂直推回边框
    - 检测失败 → 激光在黑色边框上 → 矫正衰减
    - 距终点 < 15px → 提前退出

    矫正速度持久化，不每帧重置。
    """
    az_speed, pt_speed, duration = move_params
    t_start = time.time()
    next_check = t_start + check_interval
    MIN_ERROR = 5.0
    ARRIVE_EARLY = 15.0

    # 持久化的矫正量 (°/s)
    az_correction = 0.0
    pt_correction = 0.0

    print(
        f"  az={az_speed:+.2f}°/s  pt={pt_speed:+.2f}°/s  t={duration:.2f}s"
        f"  (edge-correct, KP={correction_gain})"
    )

    while time.time() - t_start < duration:
        if time.time() >= next_check:
            next_check = time.time() + check_interval
            pos = detect_laser_single(cam)
            if pos is not None:
                dist_to_end = np.linalg.norm(
                    np.array(P_end) - np.array(pos)
                )
                if dist_to_end < ARRIVE_EARLY:
                    motor.stop()
                    time.sleep(0.3)  # 等待云台停稳，避免惯性导致斜穿拐角
                    return

                closest = project_to_edge(pos, P_start, P_end)
                error = np.array(closest) - np.array(pos)
                dist = np.linalg.norm(error)

                if dist > MIN_ERROR:
                    angles = M_inv @ error
                    # 转为持久化速度矫正 (°/s)
                    az_correction = float(angles[0]) * correction_gain / check_interval
                    pt_correction = float(angles[1]) * correction_gain / check_interval
                else:
                    az_correction = 0.0
                    pt_correction = 0.0
            else:
                # 检测失败（激光在黑色胶带上）→ 矫正衰减
                az_correction *= 0.5
                pt_correction *= 0.5

        # 基础速度 + 持久矫正（不每帧重置）
        current_az = az_speed + az_correction
        current_pt = pt_speed + pt_correction
        motor.set_speed(current_az, current_pt)
        time.sleep(0.05)

    motor.stop()
    time.sleep(0.3)


def track_one_loop(
    motor: MotorController,
    cam: Camera,
    corners: list[tuple[float, float]],
    M_inv: np.ndarray,
    speed: float,
) -> None:
    """开环绕矩形一圈：P0 → P1 → P2 → P3 → P0。"""
    current_pos = corners[0]
    for i in range(4):
        P_start = corners[i]
        P_end = corners[(i + 1) % 4]
        move = calc_move(current_pos, P_end, M_inv, speed)
        if move:
            execute_move_with_correction(
                motor, cam, move, P_start, P_end, M_inv,
            )
            current_pos = P_end


def closed_loop_move(
    motor: MotorController,
    cam: Camera,
    target: tuple[float, float],
    M_inv: np.ndarray,
    timeout: float = 10.0,
) -> bool:
    """闭环移动到目标像素坐标。

    实时检测激光 → 像素误差 → M⁻¹ → 角度 → set_speed。
    噪声剔除(>50px)、丢失潮汐(0.25)、到达阈值(10px)。
    返回: True 到达 / False 超时
    """
    KP = 0.8
    COAST_FACTOR = 0.25
    ARRIVE_THRESHOLD = 10.0
    NOISE_THRESHOLD = 50.0

    last_valid_pos: tuple[float, float] | None = None
    t_start = time.time()

    while True:
        if time.time() - t_start > timeout:
            print(f"\n  [超时] {timeout:.0f}s未到达 ({target[0]:.0f},{target[1]:.0f})")
            motor.stop()
            return False

        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue

        pos, _ = process_laser_detection(frame)

        if pos is not None:
            if last_valid_pos is None:
                last_valid_pos = pos
                filtered_pos = pos
            elif np.linalg.norm(np.array(pos[:2]) - np.array(last_valid_pos[:2])) > NOISE_THRESHOLD:
                time.sleep(0.05)
                continue
            else:
                last_valid_pos = pos
                filtered_pos = pos

            error = np.array(target[:2]) - np.array(filtered_pos[:2])
            dist = np.linalg.norm(error)

            if dist < ARRIVE_THRESHOLD:
                motor.stop()
                return True

            angles = M_inv @ error
            motor.set_speed(float(angles[0]) * KP, float(angles[1]) * KP)

        else:
            if last_valid_pos is not None:
                error = np.array(target[:2]) - np.array(last_valid_pos[:2])
                angles = M_inv @ error
                motor.set_speed(
                    float(angles[0]) * KP * COAST_FACTOR,
                    float(angles[1]) * KP * COAST_FACTOR,
                )
            else:
                motor.stop()

        time.sleep(0.05)


def closed_loop_retry(
    motor: MotorController,
    cam: Camera,
    target: tuple[float, float],
    M_inv: np.ndarray,
    max_retries: int = 3,
    settle: float = 0.5,
) -> bool:
    """闭环归位，最多重试 max_retries 次。

    每次失败后静置 settle 秒让云台停稳，再重试。
    返回: True 到达 / False 全部失败
    """
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"    重试 ({attempt}/{max_retries})...")
            time.sleep(settle)
            motor.stop()
            time.sleep(0.2)
        ok = closed_loop_move(motor, cam, target, M_inv)
        if ok:
            return True
    return False


def main():
    global _cam, _rect_mgr, _calibrator

    print("=" * 50)
    print("开环循迹验证 (Open Loop)")
    print("=" * 50)

    subprocess.run(["pkill", "-f", "flask"], check=False)
    time.sleep(0.3)

    # ---- 摄像头 ----
    _cam = Camera()
    _cam.start()
    print("等待摄像头就绪...", end="", flush=True)
    for _ in range(50):
        if _cam.read() is not None:
            print(" OK")
            break
        time.sleep(0.1)
    else:
        print(" 超时！")
        _cam.release()
        return

    # ---- Web 推流 ----
    stream = MjpegStream(frame_provider=_frame_provider, title="开环验证")
    stream.start()
    print(f"Web 监控: http://0.0.0.0:5000")

    # ---- 串口 & 电机 ----
    uart = UartController(port=CAN_PORT, baudrate=CAN_BAUD,
                          dtr=False, rts=False, open_delay=0.5)
    motor = MotorController(uart)
    print(f"串口: {CAN_PORT} @ {CAN_BAUD}")

    # ---- GPIO 按键 ----
    keys = KeypadController(PIN_MAP)
    print(f"GPIO 按键: {len(PIN_MAP)} 个, 映射 {PIN_MAP}")

    # ===== [0/4] 屏幕标定 (强制) =====
    print()
    print("[0/4] 屏幕角点标定 (每次启动强制)")
    sc = ScreenCalibrator(_cam, keys, _screen_corners, _screen_done, _screen_laser_pos)
    screen_corners = sc.run()
    if screen_corners is None:
        motor.stop()
        keys.cleanup()
        _cam.release()
        uart.close()
        return

    # ===== [1/3] 矩形检测 =====
    _rect_mgr = RectangleManager()
    print()
    print("[1/3] 矩形检测")
    print("将黑色胶带矩形放入画面，浏览器确认后按 [enter] (GPIO5)")

    corners = None
    center = None
    labels = ["左上", "右上", "右下", "左下"]

    while True:
        print("\n等待按键... (按 [enter] 开始检测, 按 [q] 退出)")
        key = keys.wait_key()
        if key == "q":
            _cam.release()
            uart.close()
            keys.cleanup()
            return
        if key != "enter":
            continue

        print("检测中...")
        detected = False
        for _ in range(10):
            frame = _cam.read()
            if frame is None:
                time.sleep(0.1)
                continue
            if _rect_mgr.detect(frame):
                detected = True
                break
            time.sleep(0.2)

        if detected:
            corners = _rect_mgr.get_ordered_corners()
            center = _rect_mgr.get_center()
            print("检测成功！角点:")
            for i, (x, y) in enumerate(corners):
                print(f"  P{i}({labels[i]}): ({x:.1f}, {y:.1f})")
            print(f"  中心: ({center[0]:.1f}, {center[1]:.1f})")
            _rect_center_display[0] = center
            break

        print("未检测到矩形，调整矩形位置后重试")

    # ===== [2/3] 标定 =====
    print()
    print("[2/3] 云台标定")
    M_inv = load_calibration(CALIB_FILE)
    if M_inv is not None:
        print(f"已找到标定文件: {CALIB_FILE}")
        print("  [1] 跳过标定，使用已有数据")
        print("  [2] 重新标定")
    else:
        print("未找到标定文件")
        print("  [1] 跳过标定，使用默认值（单位矩阵）")
        print("  [2] 运行标定流程")

    print("  等待按键... (GPIO6=跳过, GPIO26=标定)")
    choice = keys.wait_key() or "1"

    if choice == "2":
        # 标定前确保电机静止
        motor.stop()
        # 启动标定器 (Web 推流会自动显示轨迹)
        _calibrator = Calibrator(_cam, motor)
        print("\n标定中... 浏览器可查看实时轨迹")
        print("云台将先归零，然后分别沿 az/pt 轴匀速运动\n")

        M_inv = _calibrator.run()
        _calibrator = None  # 退出标定模式

        if M_inv is not None:
            save_calibration(M_inv, CALIB_FILE)
            motor.stop()
            print("标定完成，结果已保存")
        else:
            print("标定失败，使用默认值")
            M_inv = np.eye(2)
    else:
        if M_inv is None:
            print("使用默认单位矩阵")
            M_inv = np.eye(2)

    print("标定矩阵 M⁻¹:")
    print(f"  [{M_inv[0,0]:8.4f}  {M_inv[0,1]:8.4f}]")
    print(f"  [{M_inv[1,0]:8.4f}  {M_inv[1,1]:8.4f}]")

    # ===== [3/3] 就绪 =====
    print()
    print("[3/3] 就绪")
    print("=" * 50)
    print("  [1] 复位到屏幕中心    [2] 绕矩形循迹    [3] 绕屏幕循迹")
    print("  [r] 重新检测矩形      [s] 重新屏幕标定  [c] 重新云台标定")
    print("  [q] 退出")
    print("=" * 50)

    # 主菜单按键提示
    print("  按键: GPIO6=1  GPIO26=2  GPIO17=3")
    print("        GPIO25=r  GPIO27=s  GPIO23=c  GPIO24=q")

    # 屏幕中心
    sc = screen_corners
    screen_center: tuple[float, float] = (
        (sc[0][0] + sc[1][0] + sc[2][0] + sc[3][0]) / 4.0,
        (sc[0][1] + sc[1][1] + sc[2][1] + sc[3][1]) / 4.0,
    )

    try:
        while True:
            print("\n等待按键...  ", end="", flush=True)
            cmd = keys.wait_key()
            if cmd is None:
                continue

            # GPIO27 在 Phase 1 是 "undo"，Phase 4 复用为 "s"
            if cmd == "undo":
                cmd = "s"

            print(f"[GPIO] {cmd}")

            if cmd == "1":
                print("闭环复位到屏幕中心...")
                ok = closed_loop_retry(motor, _cam, screen_center, M_inv)
                print("已到达屏幕中心" if ok else "复位未完全到达，可再次尝试")

            elif cmd == "2":
                rcorners = _rect_mgr.get_ordered_corners()
                rcenter = _rect_mgr.get_center()
                if not rcorners or rcenter is None:
                    print("无矩形数据")
                    continue
                # 显示矩形中心（品红斜十字）
                _rect_center_display[0] = rcenter
                p0 = rcorners[0]

                # 开环走到 P0
                pos = detect_laser(_cam)
                if pos is not None:
                    print(f"当前位置: ({pos[0]:.0f},{pos[1]:.0f})")
                    move = calc_move(pos, p0, M_inv, CALIB_SPEED)
                    if move:
                        execute_move(motor, move)
                else:
                    print("未检测到激光，跳过归位 (可按 [2] 重试)")

                print("绕矩形循迹一圈...")
                track_one_loop(motor, _cam, rcorners, M_inv, CALIB_SPEED)

                # 循迹结束在 P0，开环回到中心
                print("循迹完成，开环复位到矩形中心...")
                move = calc_move(p0, rcenter, M_inv, CALIB_SPEED)
                if move:
                    execute_move(motor, move)
                print("已完成")

            elif cmd == "3":
                pos = detect_laser(_cam)
                if pos is None:
                    print("未检测到激光，无法循迹")
                    continue
                print(f"绕屏幕循迹一圈 (闭环)...")
                print(f"  起点: ({pos[0]:.0f}, {pos[1]:.0f})")
                sc_labels = ["左上", "右上", "右下", "左下"]
                all_ok = True
                # 走到每个角点
                for i, corner in enumerate(screen_corners):
                    print(f"  [{i+1}/4] → {sc_labels[i]} ({corner[0]:.0f}, {corner[1]:.0f})")
                    ok = closed_loop_move(motor, _cam, corner, M_inv)
                    if not ok:
                        print(f"  [跳过] 未到达 {sc_labels[i]}")
                        all_ok = False
                # 闭合回到起点
                if all_ok:
                    print(f"  [闭合] → 起点 ({screen_corners[0][0]:.0f}, {screen_corners[0][1]:.0f})")
                    closed_loop_move(motor, _cam, screen_corners[0], M_inv)
                print("循迹完成")

            elif cmd == "r":
                print("重新检测矩形...")
                detected = False
                for _ in range(10):
                    frame = _cam.read()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    if _rect_mgr.detect(frame):
                        detected = True
                        break
                    time.sleep(0.2)
                if detected:
                    rcorners = _rect_mgr.get_ordered_corners()
                    rcenter = _rect_mgr.get_center()
                    rlabels = ["左上", "右上", "右下", "左下"]
                    print("矩形已更新:")
                    for i, (x, y) in enumerate(rcorners):
                        print(f"  P{i}({rlabels[i]}): ({x:.1f}, {y:.1f})")
                    print(f"  中心: ({rcenter[0]:.1f}, {rcenter[1]:.1f})")
                    _rect_center_display[0] = rcenter
                else:
                    print("未检测到矩形，保持旧数据")

            elif cmd in ("s", "undo"):
                print("重新屏幕标定...")
                sc_re = ScreenCalibrator(_cam, keys, _screen_corners, _screen_done, _screen_laser_pos)
                new_sc = sc_re.run()
                if new_sc is not None:
                    screen_corners = new_sc
                    sc = screen_corners
                    screen_center = (
                        (sc[0][0] + sc[1][0] + sc[2][0] + sc[3][0]) / 4.0,
                        (sc[0][1] + sc[1][1] + sc[2][1] + sc[3][1]) / 4.0,
                    )
                    print("屏幕标定已更新")

            elif cmd == "c":
                print("重新标定云台...")
                motor.stop()
                _calibrator = Calibrator(_cam, motor)
                print("标定中... 浏览器可查看实时轨迹")
                new_M_inv = _calibrator.run()
                _calibrator = None
                if new_M_inv is not None:
                    M_inv = new_M_inv
                    save_calibration(M_inv, CALIB_FILE)
                    motor.stop()
                    print("标定完成，结果已保存")
                else:
                    print("标定失败，保持旧矩阵")

            elif cmd == "q":
                break

            else:
                print(f"未知命令: {cmd}")

    finally:
        motor.stop()
        keys.cleanup()
        _cam.release()
        uart.close()
        print("已退出")


if __name__ == "__main__":
    main()

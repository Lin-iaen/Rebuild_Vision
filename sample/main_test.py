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
CALIB_SPEED = 6.0       # °/s
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


def track_one_loop(
    motor: MotorController,
    corners: list[tuple[float, float]],
    center: tuple[float, float],
    M_inv: np.ndarray,
    speed: float,
) -> None:
    """开环绕矩形一圈：center → P0→P1→P2→P3 → center。"""
    current_pos = center
    for target in corners:
        move = calc_move(current_pos, target, M_inv, speed)
        if move:
            execute_move(motor, move)
            current_pos = target
    # 回到中心
    move = calc_move(current_pos, center, M_inv, speed)
    if move:
        execute_move(motor, move)


def track_screen_loop(
    motor: MotorController,
    corners: list[tuple[float, float]],
    start_pos: tuple[float, float],
    M_inv: np.ndarray,
    speed: float,
) -> None:
    """开环绕屏幕一圈：start_pos → P0→P1→P2→P3 → P0。"""
    move = calc_move(start_pos, corners[0], M_inv, speed)
    if move:
        execute_move(motor, move)
    current_pos = corners[0]
    for target in corners:
        move = calc_move(current_pos, target, M_inv, speed)
        if move:
            execute_move(motor, move)
            current_pos = target
    move = calc_move(current_pos, corners[0], M_inv, speed)
    if move:
        execute_move(motor, move)


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
    sc = ScreenCalibrator(_cam, keys, _screen_corners, _screen_done)
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
    print("等待按键...")
    key = keys.wait_key()
    if key != "enter":
        _cam.release()
        uart.close()
        keys.cleanup()
        return

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

    if not detected:
        print("错误：未检测到矩形")
        _cam.release()
        uart.close()
        return

    corners = _rect_mgr.get_ordered_corners()
    center = _rect_mgr.get_center()
    labels = ["左上", "右上", "右下", "左下"]
    print("检测成功！角点:")
    for i, (x, y) in enumerate(corners):
        print(f"  P{i}({labels[i]}): ({x:.1f}, {y:.1f})")
    print(f"  中心: ({center[0]:.1f}, {center[1]:.1f})")

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
                print("复位到屏幕中心...")
                pos = detect_laser(_cam)
                if pos is not None:
                    print(f"  当前位置: ({pos[0]:.1f}, {pos[1]:.1f})")
                else:
                    pos = screen_center
                    print("  未检测到激光，假设位于屏幕中心")
                move = calc_move(pos, screen_center, M_inv, CALIB_SPEED)
                if move:
                    execute_move(motor, move)
                print("已到达屏幕中心")

            elif cmd == "2":
                rcorners = _rect_mgr.get_ordered_corners()
                rcenter = _rect_mgr.get_center()
                if not rcorners or rcenter is None:
                    print("无矩形数据")
                    continue
                print("绕矩形循迹一圈...")
                track_one_loop(motor, rcorners, rcenter, M_inv, CALIB_SPEED)
                print("循迹完成")

            elif cmd == "3":
                pos = detect_laser(_cam)
                start_pos = pos if pos is not None else screen_center
                print(f"绕屏幕循迹一圈... (起点: ({start_pos[0]:.0f}, {start_pos[1]:.0f}))")
                track_screen_loop(motor, screen_corners, start_pos, M_inv, CALIB_SPEED)
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
                else:
                    print("未检测到矩形，保持旧数据")

            elif cmd in ("s", "undo"):
                print("重新屏幕标定...")
                sc_re = ScreenCalibrator(_cam, keys, _screen_corners, _screen_done)
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

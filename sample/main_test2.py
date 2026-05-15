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

from calibration_test import Calibrator, load_calibration, save_calibration, LUTMapper
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
    mapper, # 变成了 LUTMapper 对象
) -> None:
    """开环绕矩形循迹：利用高密度子目标点进行平滑绝对定点跳转。"""
    # 获取所有的边沿高密度点
    targets = _rect_mgr.get_targets()
    
    # 将中心点加入作为起始和结束
    full_path = [center] + targets + [center]
    
    for target in full_path:
        # 直接通过网格插值器，算出该像素精确对应的绝对角度
        az, pt = mapper.pixel_to_angle(target[0], target[1])
        motor.set_position(az, pt)
        # 短暂休眠，等待电机走到该点。休眠越短，走得越顺滑（配合你设定的子目标密度）
        time.sleep(0.1) 


def closed_loop_move(
    motor: MotorController,
    cam: Camera,
    target: tuple[float, float],
    mapper: LUTMapper, # 换成 mapper
    timeout: float = 10.0,
) -> bool:
    """闭环移动到目标像素坐标 (基于 LUT 插值的 P 控制)"""
    KP_SPEED = 2.0  # 闭环追赶的速度比例
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

            # 距离判断
            dist = np.linalg.norm(np.array(target[:2]) - np.array(filtered_pos[:2]))
            if dist < ARRIVE_THRESHOLD:
                motor.stop()
                return True

            # 【核心魔法】：通过 LUT 分别算出目标和当前像素对应的“绝对物理角度”
            target_az, target_pt = mapper.pixel_to_angle(target[0], target[1])
            curr_az, curr_pt = mapper.pixel_to_angle(filtered_pos[0], filtered_pos[1])
            
            # 角度差即为我们需要的速度方向和大小 (P控制)
            d_az = target_az - curr_az
            d_pt = target_pt - curr_pt
            motor.set_speed(d_az * KP_SPEED, d_pt * KP_SPEED)

        else:
            # 丢失处理
            if last_valid_pos is not None:
                target_az, target_pt = mapper.pixel_to_angle(target[0], target[1])
                curr_az, curr_pt = mapper.pixel_to_angle(last_valid_pos[0], last_valid_pos[1])
                motor.set_speed(
                    (target_az - curr_az) * KP_SPEED * COAST_FACTOR,
                    (target_pt - curr_pt) * KP_SPEED * COAST_FACTOR,
                )
            else:
                motor.stop()

        time.sleep(0.05)


def closed_loop_retry(
    motor: MotorController,
    cam: Camera,
    target: tuple[float, float],
    mapper: LUTMapper,
    max_retries: int = 3,
    settle: float = 0.5,
) -> bool:
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"    重试 ({attempt}/{max_retries})...")
            time.sleep(settle)
            motor.stop()
            time.sleep(0.2)
        ok = closed_loop_move(motor, cam, target, mapper)
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
    print("[2/3] 云台标定 (LUT网格)")
    mapper = load_calibration(CALIB_FILE)
    if mapper is not None:
        print(f"已找到标定文件: {CALIB_FILE} (包含 {len(mapper.lut)} 个网格点)")
        print("  [1] 跳过标定，使用已有数据")
        print("  [2] 重新标定")
    else:
        print("未找到有效标定文件")
        print("  [1] 跳过标定 (系统将无法正常循迹)")
        print("  [2] 运行标定流程")

    print("  等待按键... (GPIO6=跳过, GPIO26=标定)")
    choice = keys.wait_key() or "1"

    if choice == "2":
        motor.stop()
        _calibrator = Calibrator(_cam, motor)
        print("\n全自动盲探标定中... 浏览器可查看实时探针")

        lut_data = _calibrator.run()
        _calibrator = None  # 退出标定模式

        if lut_data is not None:
            save_calibration(lut_data, CALIB_FILE)
            mapper = LUTMapper(lut_data) # 实例化插值器
            motor.stop()
            print("标定完成，结果已保存")
        else:
            print("标定失败！")
            mapper = None

    if mapper is None:
        print("❌ 警告：缺少标定数据，运动指令将被拦截！")

    # ===== [3/3] 就绪 =====
    print()
    print("[3/3] 就绪")
    print("=" * 50)
    print("  [1] 复位到屏幕中心    [2] 绕矩形循迹    [3] 绕屏幕循迹")
    print("  [r] 重新检测矩形      [s] 重新屏幕标定  [c] 重新云台标定")
    print("  [q] 退出")
    print("=" * 50)

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

            if cmd == "undo":
                cmd = "s"

            print(f"[GPIO] {cmd}")

            if cmd in ("1", "2", "3") and mapper is None:
                print("❌ 没有 LUT 标定数据，无法运动，请先执行标定 [c]。")
                continue

            if cmd == "1":
                print("闭环复位到屏幕中心...")
                ok = closed_loop_retry(motor, _cam, screen_center, mapper)
                print("已到达屏幕中心" if ok else "复位未完全到达，可再次尝试")

            elif cmd == "2":
                rcorners = _rect_mgr.get_ordered_corners()
                rcenter = _rect_mgr.get_center()
                if not rcorners or rcenter is None:
                    print("无矩形数据")
                    continue
                _rect_center_display[0] = rcenter
                p0 = rcorners[0]

                # 直接开环绝对位置跳转到 P0
                print("开环定点跳转到起点 P0...")
                az, pt = mapper.pixel_to_angle(p0[0], p0[1])
                motor.set_position(az, pt)
                time.sleep(0.5)

                print("绕矩形循迹一圈 (LUT 高密度插值)...")
                track_one_loop(motor, rcorners, rcenter, mapper)

                print("循迹完成，复位到矩形中心...")
                az, pt = mapper.pixel_to_angle(rcenter[0], rcenter[1])
                motor.set_position(az, pt)
                print("已完成")

            elif cmd == "3":
                pos = detect_laser(_cam)
                if pos is None:
                    print("未检测到激光，无法循迹")
                    continue
                print(f"绕屏幕循迹一圈 (闭环)...")
                sc_labels = ["左上", "右上", "右下", "左下"]
                all_ok = True
                for i, corner in enumerate(screen_corners):
                    print(f"  [{i+1}/4] → {sc_labels[i]} ({corner[0]:.0f}, {corner[1]:.0f})")
                    ok = closed_loop_move(motor, _cam, corner, mapper)
                    if not ok:
                        print(f"  [跳过] 未到达 {sc_labels[i]}")
                        all_ok = False
                if all_ok:
                    print(f"  [闭合] → 起点 ({screen_corners[0][0]:.0f}, {screen_corners[0][1]:.0f})")
                    closed_loop_move(motor, _cam, screen_corners[0], mapper)
                print("循迹完成")

            # ... 下方的 'r', 's', 'c', 'q' 的逻辑保持不变 ...
    finally:
        motor.stop()
        keys.cleanup()
        _cam.release()
        uart.close()
        print("已退出")


if __name__ == "__main__":
    main()

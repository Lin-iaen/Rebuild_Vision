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
    5:  "enter",
    6:  "1",
    26: "2",
    16: "3",
    25: "r",
    23: "c",
    24: "q",
}

# ===== 全局状态 (供 frame_provider) =====
_cam: Camera | None = None
_rect_mgr: RectangleManager | None = None
_calibrator: Calibrator | None = None


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

    # ===== [1/3] 矩形检测 =====
    _rect_mgr = RectangleManager()
    print()
    print("[1/3] 矩形检测")
    print("将黑色胶带矩形放入画面，浏览器确认后按 Enter (GPIO5)")
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
    print("  [1] 复位到矩形中心 (开环, 自动重检)")
    print("  [2] 绕矩形循迹一圈 (开环, 跑完自动停)")
    print("  [3] 停止")
    print("  [r] 重新检测矩形")
    print("  [c] 重新标定云台")
    print("  [q] 退出")
    print("=" * 50)

    try:
        while True:
            print("\n等待按键...  ", end="", flush=True)
            cmd = keys.wait_key()
            if cmd is None:
                continue
            print(f"[GPIO] {cmd}")

            if cmd == "1":
                # 复位：先重检矩形，再检测激光移到新中心
                print("复位中... (自动重检矩形)")
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
                    print("  错误：未检测到矩形")
                    continue
                rcenter = _rect_mgr.get_center()
                if rcenter is None:
                    print("  无矩形数据")
                    continue
                print(f"  新中心: ({rcenter[0]:.1f}, {rcenter[1]:.1f})")

                pos = detect_laser(_cam)
                if pos is not None:
                    print(f"  当前位置: ({pos[0]:.1f}, {pos[1]:.1f})")
                else:
                    pos = rcenter
                    print("  未检测到激光，假设位于中心")
                move = calc_move(pos, rcenter, M_inv, CALIB_SPEED)
                if move:
                    execute_move(motor, move)
                print("已到达中心")

            elif cmd == "2":
                # 循迹一圈，跑完自动停
                rcorners = _rect_mgr.get_ordered_corners()
                rcenter = _rect_mgr.get_center()
                if not rcorners or rcenter is None:
                    print("无矩形数据")
                    continue
                print("循迹一圈... (跑完自动停)")
                track_one_loop(motor, rcorners, rcenter, M_inv, CALIB_SPEED)
                print("循迹完成")

            elif cmd == "3":
                motor.stop()
                print("已停止")

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

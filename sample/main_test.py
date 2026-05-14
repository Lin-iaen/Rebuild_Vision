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
from motor import MotorController
from rectangle import RectangleManager
from tracker import process_laser_detection
from uart import UartController
from web_stream import MjpegStream

# ===== 配置 =====
CAN_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
CAN_BAUD = 4000000
CALIB_SPEED = 2.0       # °/s, 与标定时一致
CALIB_FILE = "calibration.json"
SETTLE_TIME = 3.0       # 归零等待

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


def wait_for_enter(prompt: str = "按 Enter 继续，Ctrl+C 取消") -> bool:
    try:
        input(prompt)
        return True
    except (KeyboardInterrupt, EOFError):
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

    # ===== [1/3] 矩形检测 =====
    _rect_mgr = RectangleManager()
    print()
    print("[1/3] 矩形检测")
    print("将黑色胶带矩形放入画面，浏览器确认后按 Enter")
    if not wait_for_enter():
        _cam.release()
        uart.close()
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

    try:
        choice = input("选择: ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

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

    # ===== [3/3] 开环循迹 =====
    print()
    print("[3/3] 开环循迹")

    # 检测激光当前位置
    print("检测激光位置...")
    current_pos = None
    for _ in range(5):
        frame = _cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pos, _ = process_laser_detection(frame)
        if pos is not None:
            current_pos = pos
            break
        time.sleep(0.05)

    if current_pos is not None:
        print(f"  当前位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
    else:
        current_pos = center
        print("  未检测到激光，假设位于中心")

    # 速度模式移到循迹起点（中心）
    print("\n移到循迹起点...")
    dx = center[0] - current_pos[0]
    dy = center[1] - current_pos[1]
    dpixel = np.array([dx, dy])
    dtheta = M_inv @ dpixel
    d_az = float(dtheta[0])
    d_pt = float(dtheta[1])
    t_az = abs(d_az) / CALIB_SPEED if CALIB_SPEED > 0 else 0.0
    t_pt = abs(d_pt) / CALIB_SPEED if CALIB_SPEED > 0 else 0.0
    t_total = max(t_az, t_pt)
    if t_total > 0:
        az_speed = d_az / t_total
        pt_speed = d_pt / t_total
        print(
            f"  Δp=({dx:+6.1f},{dy:+6.1f})px  "
            f"t={t_total:.2f}s  "
            f"az={az_speed:+.2f}°/s  pt={pt_speed:+.2f}°/s"
        )
        motor.set_speed(az_speed, pt_speed)
        time.sleep(t_total)
        motor.stop()
        time.sleep(0.3)
    current_pos = center
    print("已到达循迹起点")

    print("\n按 Enter 开始开环绕矩形一圈")
    if not wait_for_enter():
        motor.stop()
        _cam.release()
        uart.close()
        return

    print(f"\n开环循迹开始 (速度={CALIB_SPEED}°/s)")
    print("-" * 50)

    segments = 0
    for target_label, target_pos in zip(labels, corners):
        segments += 1
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dpixel = np.array([dx, dy])

        dtheta = M_inv @ dpixel
        d_az = float(dtheta[0])
        d_pt = float(dtheta[1])

        t_az = abs(d_az) / CALIB_SPEED if CALIB_SPEED > 0 else 0.0
        t_pt = abs(d_pt) / CALIB_SPEED if CALIB_SPEED > 0 else 0.0
        t_total = max(t_az, t_pt)

        if t_total == 0:
            print(f"  → {target_label}  距离为0，跳过")
            continue

        az_speed = d_az / t_total
        pt_speed = d_pt / t_total

        print(
            f"  → {target_label}  "
            f"Δp=({dx:+6.1f},{dy:+6.1f})px  "
            f"t={t_total:.2f}s  "
            f"az={az_speed:+.2f}°/s  pt={pt_speed:+.2f}°/s"
        )

        motor.set_speed(az_speed, pt_speed)
        time.sleep(t_total)
        motor.stop()
        time.sleep(0.3)

        current_pos = target_pos

    print("-" * 50)
    print(f"开环循迹完成，共 {segments} 段")

    # ---- 回到中心 ----
    print("\n回到中心...")
    dx = center[0] - current_pos[0]
    dy = center[1] - current_pos[1]
    dpixel = np.array([dx, dy])
    dtheta = M_inv @ dpixel
    d_az = float(dtheta[0])
    d_pt = float(dtheta[1])
    t_az = abs(d_az) / CALIB_SPEED
    t_pt = abs(d_pt) / CALIB_SPEED
    t_total = max(t_az, t_pt)
    if t_total > 0:
        az_speed = d_az / t_total
        pt_speed = d_pt / t_total
        print(
            f"  Δp=({dx:+6.1f},{dy:+6.1f})px  "
            f"t={t_total:.2f}s  "
            f"az={az_speed:+.2f}°/s  pt={pt_speed:+.2f}°/s"
        )
        motor.set_speed(az_speed, pt_speed)
        time.sleep(t_total)
        motor.stop()

    print("\n完成，退出")
    _cam.release()
    uart.close()


if __name__ == "__main__":
    main()

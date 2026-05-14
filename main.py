"""IBVS 极简视觉追踪系统 - 主入口。

状态机流程:
    INIT → 矩形检测 → CALIBRATE(可选) → READY → RESET / TRACK

用法:
    python3 main.py
"""

import logging
import subprocess
import time

import cv2
import numpy as np

from calibration import Calibrator, load_calibration, save_calibration
from camera import Camera
from control import LaserTracker
from motor import MotorController
from rectangle import RectangleManager
from uart import UartController
from web_stream import MjpegStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)    

# ===== 默认配置 =====
CAN_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
CAN_BAUD = 4000000
CALIB_FILE = "calibration.json"
DEFAULT_M_INV = np.array([[1.0, 0.0], [0.0, 1.0]])

# ===== 全局状态（供 frame_provider 读取）=====
_cam: Camera | None = None
_rect_mgr: RectangleManager | None = None
_tracker: LaserTracker | None = None
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

    # 循迹状态
    if _tracker is not None:
        annotated = _tracker.annotate(annotated)

    ok, buf = cv2.imencode(".jpg", annotated)
    return buf.tobytes() if ok else None


def wait_for_enter(prompt: str = "按 Enter 继续，Ctrl+C 取消") -> bool:
    """等待用户按 Enter。"""
    try:
        input(prompt)
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def main() -> None:
    global _cam, _rect_mgr, _tracker, _calibrator

    # ===== 初始化 =====
    print("=" * 50)
    print("IBVS 追踪系统")
    print("=" * 50)

    subprocess.run(["pkill", "-f", "flask"], check=False)
    time.sleep(0.3)

    # 摄像头
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

    # 立即启动 Web 推流
    stream = MjpegStream(frame_provider=_frame_provider, title="IBVS 追踪")
    stream.start()
    print(f"Web 监控: http://0.0.0.0:5000")

    # 串口
    uart = UartController(port=CAN_PORT, baudrate=CAN_BAUD,
                          dtr=False, rts=False, open_delay=0.5)
    print(f"串口 (CAN): {CAN_PORT} @ {CAN_BAUD}")

    # 电机
    motor = MotorController(uart)

    # 矩形管理器
    _rect_mgr = RectangleManager()

    # ===== 矩形检测 =====
    print()
    print("[1/3] 矩形检测")
    print("将黑色胶带矩形放入画面，按 Enter 开始检测")

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

    # ===== 标定 =====
    print()
    print("[2/3] 云台标定")

    # 尝试加载已有标定文件
    saved_M_inv = load_calibration(CALIB_FILE)
    if saved_M_inv is not None:
        print(f"已找到标定文件: {CALIB_FILE}")
        print("  [1] 跳过标定，使用已有数据")
        print("  [2] 重新标定")
    else:
        print("未找到标定文件")
        print("  [1] 跳过标定，使用默认值（单位矩阵）")
        print("  [2] 运行标定流程")

    M_inv = saved_M_inv if saved_M_inv is not None else DEFAULT_M_INV.copy()

    try:
        choice = input("选择: ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    if choice == "2":
        # 执行标定
        _calibrator = Calibrator(cam=_cam, motor=motor)
        print("标定中... 浏览器可查看实时轨迹")
        print("请确保激光点打在屏幕上")

        new_M_inv = _calibrator.run()
        _calibrator = None  # 退出标定模式

        if new_M_inv is not None:
            M_inv = new_M_inv
            save_calibration(M_inv, CALIB_FILE)
            print("标定完成，结果已保存")
        else:
            print("标定失败，使用已有数据或默认值")

    # ===== 初始化控制模块 =====
    _tracker = LaserTracker(cam=_cam, motor=motor, rect_mgr=_rect_mgr, M_inv=M_inv)

    # ===== 主菜单 =====
    print()
    print("[3/3] 就绪")
    print("=" * 50)
    print("  [1] 复位到矩形中心")
    print("  [2] 绕矩形循迹一圈")
    print("  [3] 停止")
    print("  [q] 退出")
    print("=" * 50)

    try:
        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break

            if cmd == "1":
                print("复位中... (浏览器可查看)")
                _tracker.start_reset()
            elif cmd == "2":
                print("循迹中... (浏览器可查看)")
                _tracker.start_track()
            elif cmd == "3":
                _tracker.stop()
                print("已停止")
            elif cmd == "q":
                break
            else:
                print("未知命令，输入 1/2/3/q")

    finally:
        if _tracker:
            _tracker.stop()
        _cam.release()
        uart.close()
        print("已退出")


if __name__ == "__main__":
    main()

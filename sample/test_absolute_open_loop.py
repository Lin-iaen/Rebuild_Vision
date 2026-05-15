"""绝对角度标定 + 开环绝对坐标轨迹循迹 测试脚本。

核心思路：
1. 彻底弃用 `set_speed` + `time.sleep` 的积分标定方式，避免时间调度带来的误差。
2. 发送死位置 `set_position`，等待稳定后直接测像素差，计算极其精确的 M 矩阵。
3. 循迹时，把矩形拆成多个点，全部用 `set_position` 走轨迹，杜绝开环平移误差！

用法:
    python3 sample/test_absolute_open_loop.py
    浏览器访问 http://<ip>:5000 查看实时画面，终端按提示操作。
"""

import json
import subprocess
import sys
import time
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from motor import MotorController
from rectangle import RectangleManager
from tracker import process_laser_detection
from uart import UartController
from web_stream import MjpegStream

# ===== 配置 =====
CAN_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
CAN_BAUD = 4000000

CALIB_ANGLE = 5.0          # 绝对标定的测试角度(°)
CALIB_SETTLE = 2.0         # 移动到绝对位置后等稳定(s)
CALIB_FILE = "calibration_abs.json"

TRACK_SPEED = 150.0        # 循迹速度 (像素/秒)
TRACK_FPS = 20.0           # 循迹发送指令的频率(Hz)

# ===== 状态共享 =====
_cam = None
_rect_mgr = None
_current_laser = None
_trail_points = []
_mode_text = "[Init]"


def _frame_provider() -> bytes | None:
    if _cam is None: return None
    frame = _cam.read()
    if frame is None: return None

    annotated = frame.copy()
    
    # 矩形
    if _rect_mgr:
        annotated = _rect_mgr.annotate(annotated)

    # 轨迹点
    for pt in _trail_points:
        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    # 激光点
    if _current_laser:
        lx, ly = int(_current_laser[0]), int(_current_laser[1])
        cv2.drawMarker(annotated, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # 状态
    cv2.putText(annotated, _mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    ok, buf = cv2.imencode(".jpg", annotated)
    return buf.tobytes() if ok else None


def update_laser_bg_task():
    """后台持续更新激光位置，用于显示"""
    global _current_laser
    while True:
        if _cam is None: 
            time.sleep(0.1)
            continue
        frame = _cam.read()
        if frame is not None:
            pos, _ = process_laser_detection(frame)
            _current_laser = pos
        time.sleep(0.05)


def sample_laser(cam, n=10):
    positions = []
    print(f"采样激光点中...", end="", flush=True)
    for _ in range(n):
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pos, _ = process_laser_detection(frame)
        if pos is not None:
            positions.append(pos)
            print(".", end="", flush=True)
        time.sleep(0.05)
    print()
    if len(positions) < n * 0.5:
        return None
    return np.mean(positions, axis=0)


def generate_trajectory(corners, speed_px, fps):
    """基于离散点插值生成平滑轨迹"""
    trajectory = []
    dt = 1.0 / fps
    step_px = speed_px * dt

    path = corners + [corners[0]]
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i+1])
        dist = np.linalg.norm(end - start)
        num_steps = max(1, int(dist / step_px))
        
        for k in range(num_steps):
            pt = start + (end - start) * (k / num_steps)
            trajectory.append(pt)
    
    # 终点加入
    trajectory.append(np.array(path[-1]))
    return trajectory


def main():
    global _cam, _rect_mgr, _mode_text, _trail_points

    print("=" * 55)
    print("绝对角度标定 + 开环绝对循迹 测试")
    print("=" * 55)

    subprocess.run(["pkill", "-f", "flask"], check=False)
    time.sleep(0.3)

    # 1. 启动摄像头推流
    _cam = Camera()
    _cam.start()
    for _ in range(30):
        if _cam.read() is not None: break
        time.sleep(0.1)
    
    import threading
    t = threading.Thread(target=update_laser_bg_task, daemon=True)
    t.start()

    stream = MjpegStream(frame_provider=_frame_provider, title="Abs Open Loop")
    stream.start()
    print("浏览器查看: http://0.0.0.0:5000\n")

    # 2. 串口电机
    uart = UartController(port=CAN_PORT, baudrate=CAN_BAUD)
    motor = MotorController(uart)
    
    # === [Phase 1] 矩形检测 ===
    _mode_text = "Phase 1: Rectangle"
    print("\n[阶段1: 矩形检测]")
    _rect_mgr = RectangleManager()
    corners = None
    input("请布置好矩形后，按 Enter 键搜索矩形...")
    for _ in range(20):
        f = _cam.read()
        if f is not None and _rect_mgr.detect(f):
            corners = _rect_mgr.get_ordered_corners()
            break
        time.sleep(0.1)
    
    if not corners:
        print("未检测到矩形，退出")
        _cam.release(); uart.close(); return
    print(f"矩形检测成功，角点: {[(round(x,1), round(y,1)) for (x,y) in corners]}")


    # === [Phase 2] 绝对标定 ===
    _mode_text = "Phase 2: Abs Calib"
    print("\n[阶段2: 绝对角度标定]")
    print(f"提示: 将复位到(0,0)，请确保云台激光能打在屏幕上。")
    input("按 Enter 键开始绝对标定...")

    motor.set_position(0.0, 0.0)
    print("正在移动至 (0.0, 0.0) 并稳定...")
    time.sleep(CALIB_SETTLE)
    p0 = sample_laser(_cam)

    if p0 is None:
        print("在(0,0)处无法看到激光偏点！请手动调整云台底座后重试。")
        _cam.release(); uart.close(); return
    print(f"原点像素: {p0[0]:.1f}, {p0[1]:.1f}")

    motor.set_position(CALIB_ANGLE, 0.0)
    print(f"正在移动至 (+{CALIB_ANGLE}, 0.0)...")
    time.sleep(CALIB_SETTLE)
    p1 = sample_laser(_cam)
    
    motor.set_position(0.0, CALIB_ANGLE)
    print(f"正在移动至 (0.0, +{CALIB_ANGLE})...")
    time.sleep(CALIB_SETTLE)
    p2 = sample_laser(_cam)

    if p1 is None or p2 is None:
        print("标定偏移太远丢点，请缩小 CALIB_ANGLE !")
        _cam.release(); uart.close(); return

    dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
    dx2, dy2 = p2[0] - p0[0], p2[1] - p0[1]
    
    a = dx1 / CALIB_ANGLE
    c = dy1 / CALIB_ANGLE
    b = dx2 / CALIB_ANGLE
    d = dy2 / CALIB_ANGLE

    M = np.array([[a, b], [c, d]])
    det = a * d - b * c
    if abs(det) < 1e-6:
        print("矩阵共线异常!")
        return

    M_inv = np.array([[d, -b], [-c, a]]) / det
    
    print("绝对标定完成 M⁻¹:")
    print(M_inv)
    with open(CALIB_FILE, "w") as f:
        json.dump({"M_inv": M_inv.tolist()}, f)
    
    # 停回原点
    motor.set_position(0.0, 0.0)
    time.sleep(CALIB_SETTLE)
    actual_origin = sample_laser(_cam)
    if actual_origin is None: actual_origin = p0
    print(f"回到原点，修正基准像素: {actual_origin[0]:.1f}, {actual_origin[1]:.1f}")


    # === [Phase 3] 轨迹循迹 ===
    _mode_text = "Phase 3: Trajectory Track"
    _trail_points.clear()
    
    print("\n[阶段3: 绝对开环循迹]")
    input("按 Enter 键开始沿矩形跑圈...")
    
    # 插值产生所有需要执行的像素点
    traj_pixels = generate_trajectory(corners, TRACK_SPEED, TRACK_FPS)
    print(f"生成了 {len(traj_pixels)} 个轨迹点插值，准备发送死位置指令...")

    for i, target_px in enumerate(traj_pixels):
        _trail_points.append(target_px) # UI显示
        
        # 计算该像素点对应的绝对物理角度: dθ = M⁻¹ * dP
        dpixel = np.array([target_px[0] - actual_origin[0], target_px[1] - actual_origin[1]])
        dtheta = M_inv @ dpixel
        
        az = float(dtheta[0])
        pt = float(dtheta[1])
        
        # 直接发送极高频绝对位置，杜绝速度误差累积
        motor.set_position(az, pt)
        time.sleep(1.0 / TRACK_FPS)

    print("\n跑圈完成。因为全程使用的是基于 (0,0) 的绝对位置解算，理论上绝不会发生平移误差。")
    print("查看浏览器确认效果。")
    
    _mode_text = "[Done] Check UI"
    input("按 Enter 结束程序")
    motor.set_position(0, 0)
    time.sleep(0.5)
    _cam.release()
    uart.close()

if __name__ == "__main__":
    main()

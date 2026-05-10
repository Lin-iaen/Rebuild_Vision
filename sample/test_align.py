"""云台坐标系标定工具（轨迹版）。

摄像头和云台坐标系之间可能存在旋转、缩放、轴耦合，
本脚本通过发送已知角度指令，记录激光完整轨迹，反解出 2×2 变换矩阵。

物理情境：
    摄像头固定，与云台同水平线，面向约 1 米外的屏幕。
    激光笔装在云台上，云台转动 → 激光移动 → 摄像头看到像素偏移。

数学模型：
    [Δpx]   [a  b] [Δθx]
    [Δpy] = [c  d] [Δθy]

    正向矩阵 M: 角度 → 像素
    逆矩阵 M⁻¹: 像素 → 角度（实际跟踪需要的）

用法：
    python3 sample/test_align.py
    浏览器访问 http://<树莓派IP>:5000 查看实时轨迹
"""

import logging
import struct
import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import serial

from camera import Camera
from web_stream import MjpegStream
from tracker import process_laser_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ===== 通信协议常量 =====
HEADER_X = bytes([0x02, 0x01])  # X轴帧头
HEADER_Y = bytes([0x02, 0x02])  # Y轴帧头
SERIAL_PORT = "/dev/ttyAMA0"  # 根据实际情况修改串口设备路径
SERIAL_BAUD = 115200
ANGLE_SCALE = 100  # 角度 × 100 转为整数

# ===== 标定参数 =====
CALIB_ANGLE = 30.0    # 标定转动角度（度）
SETTLE_TIME = 5.0    # 每轴等待云台稳定时间（秒）
SAMPLE_COUNT = 10     # 初始位置采样帧数

# ===== 颜色 =====
COLOR_X = (255, 128, 0)   # X轴轨迹：蓝色 (BGR)
COLOR_Y = (0, 60, 255)    # Y轴轨迹：红色
COLOR_CURRENT = (0, 255, 0)  # 当前位置：绿色
COLOR_START = (255, 200, 0)  # 起点：亮蓝
COLOR_TEXT = (0, 255, 255)   # 状态文字：黄色

# ===== 全局状态 =====
_trail_x: list[tuple[float, float, float]] = []  # (timestamp, x, y)
_trail_y: list[tuple[float, float, float]] = []
_trail_phase: str = "idle"  # "idle" / "x_axis" / "y_axis"
_trail_lock = threading.Lock()
_latest_annotated: np.ndarray | None = None
_annotated_lock = threading.Lock()
_current_pos: tuple[float, float] | None = None
_status_text: str = "等待开始"


def send_gimbal_cmd(ser: serial.Serial, header: bytes, angle_deg: float) -> None:
    """发送单轴云台指令。

    帧格式: [header_1] [header_2] [data_h] [data_l]，共 4 字节
    数据: 有符号 int16，大端，值 = 角度 × 100
    """
    value = int(round(angle_deg * ANGLE_SCALE))
    value = max(-32768, min(32767, value))
    data = struct.pack(">h", value)
    frame = header + data
    ser.write(frame)
    logger.info(f"发送: {header.hex()} | 角度={angle_deg:.1f}° | 值={value} | 帧={frame.hex()}")


def sample_laser_pos(cam: Camera, n: int = SAMPLE_COUNT) -> tuple[float, float] | None:
    """多次采样激光位置，返回平均坐标 (x, y)。未检测到返回 None。"""
    positions = []
    for _ in range(n):
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pos, _ = process_laser_detection(frame)
        if pos is not None:
            positions.append(pos)
        time.sleep(0.05)

    if len(positions) < n * 0.5:
        return None

    arr = np.array(positions)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    logger.info(f"采样 {len(positions)}/{n} 帧 | 均值=({mean[0]:.1f}, {mean[1]:.1f}) | 标准差=({std[0]:.2f}, {std[1]:.2f})")
    return float(mean[0]), float(mean[1])


def _record_loop(cam: Camera) -> None:
    """后台线程：持续检测激光位置，追加到当前阶段的轨迹列表。"""
    global _trail_x, _trail_y, _trail_phase, _current_pos

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.03)
            continue

        pos, annotated = process_laser_detection(frame)
        now = time.time()

        # 更新当前位置
        with _trail_lock:
            _current_pos = pos

            if pos is not None:
                if _trail_phase == "x_axis":
                    _trail_x.append((now, pos[0], pos[1]))
                elif _trail_phase == "y_axis":
                    _trail_y.append((now, pos[0], pos[1]))

        # 更新标注帧（带轨迹）
        _update_annotated(annotated, pos)

        time.sleep(0.03)


def _draw_trail(annotated: np.ndarray, trail: list, color: tuple) -> None:
    """在画面上画轨迹折线 + 方向箭头。"""
    if len(trail) < 2:
        return

    points = [(int(x), int(y)) for _, x, y in trail]

    # 折线
    for i in range(1, len(points)):
        thickness = 2 if i > len(points) - 10 else 1
        cv2.line(annotated, points[i - 1], points[i], color, thickness)

    # 起点圆圈
    cv2.circle(annotated, points[0], 8, color, 2)

    # 终点方块
    ex, ey = points[-1]
    cv2.rectangle(annotated, (ex - 6, ey - 6), (ex + 6, ey + 6), color, -1)

    # 方向箭头（每隔一定距离画一个）
    if len(points) >= 20:
        step = len(points) // 4
        for i in range(step, len(points) - 1, step):
            p1 = points[i]
            p2 = points[min(i + 5, len(points) - 1)]
            cv2.arrowedLine(annotated, p1, p2, color, 2, tipLength=0.3)


def _update_annotated(annotated: np.ndarray, current_pos: tuple[float, float] | None) -> None:
    """更新带轨迹标注的帧。"""
    global _latest_annotated

    with _trail_lock:
        trail_x = list(_trail_x)
        trail_y = list(_trail_y)
        phase = _trail_phase
        pos = current_pos

    # 画 X 轴轨迹
    _draw_trail(annotated, trail_x, COLOR_X)

    # 画 Y 轴轨迹
    _draw_trail(annotated, trail_y, COLOR_Y)

    # 画当前位置
    if pos is not None:
        xi, yi = int(pos[0]), int(pos[1])
        cv2.drawMarker(annotated, (xi, yi), COLOR_CURRENT, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            annotated,
            f"({xi},{yi})",
            (xi + 12, yi - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_CURRENT,
            1,
        )

    # 状态文字
    cv2.putText(
        annotated,
        _status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
    )

    # 轨迹点数
    cv2.putText(
        annotated,
        f"X:{len(trail_x)}pts  Y:{len(trail_y)}pts",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    with _annotated_lock:
        _latest_annotated = annotated


def _frame_provider() -> bytes | None:
    """MjpegStream 回调：返回最新标注帧的 JPEG。"""
    with _annotated_lock:
        frame = _latest_annotated

    if frame is None:
        return None

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def _trail_endpoint(trail: list, tail_n: int = 15) -> tuple[float, float] | None:
    """取轨迹尾部 N 帧的平均值作为终点。"""
    if len(trail) < 3:
        return None
    tail = trail[-tail_n:]
    arr = np.array([(x, y) for _, x, y in tail])
    mean = arr.mean(axis=0)
    return float(mean[0]), float(mean[1])


def main() -> None:
    global _trail_phase, _status_text

    # 初始化摄像头
    cam = Camera()
    cam.start()

    print("等待摄像头就绪...", end="", flush=True)
    for _ in range(50):
        if cam.read() is not None:
            print(" OK")
            break
        time.sleep(0.1)
    else:
        print(" 超时！未收到帧")
        cam.release()
        return

    # 启动轨迹记录线程
    recorder = threading.Thread(target=_record_loop, args=(cam,), daemon=True)
    recorder.start()

    # 启动推流
    stream = MjpegStream(frame_provider=_frame_provider, title="云台标定 - 轨迹")
    stream.start()

    # 初始化串口
    try:
        ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=0.1)
        logger.info(f"串口 {SERIAL_PORT} 打开成功")
    except serial.SerialException as e:
        logger.error(f"无法打开串口: {e}")
        cam.release()
        return

    print()
    print("=" * 55)
    print("云台坐标系标定（轨迹版）")
    print("=" * 55)
    print(f"标定角度: {CALIB_ANGLE}°")
    print(f"每轴等待: {SETTLE_TIME}s")
    print(f"浏览器查看: http://0.0.0.0:5000")
    print()
    print("请确保激光点打在屏幕上，且在摄像头视野内")
    print("按 Enter 开始标定，Ctrl+C 取消")
    print("=" * 55)

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        print("已取消")
        cam.release()
        ser.close()
        return

    # ===== 步骤 1: 采样初始位置 =====
    _status_text = "[1/3] 采样初始位置..."
    print()
    print(_status_text)

    pos0 = sample_laser_pos(cam)
    if pos0 is None:
        print("错误：未检测到激光点，请检查激光是否开启")
        cam.release()
        ser.close()
        return
    x0, y0 = pos0
    print(f"  初始位置: ({x0:.1f}, {y0:.1f})")

    # ===== 步骤 2: X 轴 =====
    _status_text = f"[2/3] X轴 +{CALIB_ANGLE}° 采集中..."
    print()
    print(_status_text)

    with _trail_lock:
        _trail_x.clear()
        _trail_phase = "x_axis"

    send_gimbal_cmd(ser, HEADER_X, CALIB_ANGLE)

    # 等待云台稳定
    for remaining in range(int(SETTLE_TIME), 0, -1):
        _status_text = f"[2/3] X轴 +{CALIB_ANGLE}° 等待稳定 {remaining}s..."
        time.sleep(1)

    # 取轨迹终点
    with _trail_lock:
        x_trail = list(_trail_x)

    pos1 = _trail_endpoint(x_trail)
    if pos1 is None:
        print("错误：X 轴轨迹数据不足")
        cam.release()
        ser.close()
        return

    x1, y1 = pos1
    dx1 = x1 - x0
    dy1 = y1 - y0
    print(f"  轨迹点数: {len(x_trail)}")
    print(f"  终点位置: ({x1:.1f}, {y1:.1f})")
    print(f"  偏移: Δpx={dx1:+.1f}, Δpy={dy1:+.1f}")

    # ===== 步骤 3: Y 轴 =====
    _status_text = f"[3/3] Y轴 +{CALIB_ANGLE}° 采集中..."
    print()
    print(_status_text)

    with _trail_lock:
        _trail_y.clear()
        _trail_phase = "y_axis"

    send_gimbal_cmd(ser, HEADER_Y, CALIB_ANGLE)

    for remaining in range(int(SETTLE_TIME), 0, -1):
        _status_text = f"[3/3] Y轴 +{CALIB_ANGLE}° 等待稳定 {remaining}s..."
        time.sleep(1)

    with _trail_lock:
        y_trail = list(_trail_y)
        _trail_phase = "idle"

    pos2 = _trail_endpoint(y_trail)
    if pos2 is None:
        print("错误：Y 轴轨迹数据不足")
        cam.release()
        ser.close()
        return

    x2, y2 = pos2
    dx2 = x2 - x1
    dy2 = y2 - y1
    print(f"  轨迹点数: {len(y_trail)}")
    print(f"  终点位置: ({x2:.1f}, {y2:.1f})")
    print(f"  偏移: Δpx={dx2:+.1f}, Δpy={dy2:+.1f}")

    # ===== 计算变换矩阵 =====
    a = dx1 / CALIB_ANGLE
    c = dy1 / CALIB_ANGLE
    b = dx2 / CALIB_ANGLE
    d = dy2 / CALIB_ANGLE

    M = np.array([[a, b], [c, d]])

    det = a * d - b * c
    if abs(det) < 1e-6:
        print("\n错误：矩阵奇异（行列式 ≈ 0），无法求逆")
        print("可能原因：激光未移动或两轴完全共线")
        _status_text = "标定失败：矩阵奇异"
        cam.release()
        ser.close()
        return

    M_inv = np.array([[d, -b], [-c, a]]) / det

    # ===== 输出结果 =====
    _status_text = "标定完成！查看终端结果"

    print()
    print("=" * 55)
    print("标定结果")
    print("=" * 55)

    print()
    print("正向矩阵 M (角度→像素):")
    print(f"  [{a:8.4f}  {b:8.4f}]    X轴转1° → 水平{a:+.2f}px, 垂直{c:+.2f}px")
    print(f"  [{c:8.4f}  {d:8.4f}]    Y轴转1° → 水平{b:+.2f}px, 垂直{d:+.2f}px")

    print()
    print("逆矩阵 M⁻¹ (像素→角度):")
    print(f"  [{M_inv[0,0]:8.4f}  {M_inv[0,1]:8.4f}]")
    print(f"  [{M_inv[1,0]:8.4f}  {M_inv[1,1]:8.4f}]")

    print()
    print("耦合分析:")
    coupling_x = abs(b) / (abs(a) + 1e-9)
    coupling_y = abs(c) / (abs(d) + 1e-9)
    if coupling_x < 0.1 and coupling_y < 0.1:
        print("  非对角元素 ≈ 0 → 两轴近似独立，可用简化公式:")
        print(f"    Δθx ≈ Δpx / {a:.2f}")
        print(f"    Δθy ≈ Δpy / {d:.2f}")
    else:
        print("  非对角元素显著 → 存在轴间耦合，必须用矩阵变换")

    print()
    print("验证示例:")
    for label, px, py in [("X轴指令效果", dx1, dy1), ("Y轴指令效果", dx2, dy2)]:
        angle = M_inv @ np.array([px, py])
        print(f"  {label}: 偏移({px:+.1f}px, {py:+.1f}px) → 逆解({angle[0]:+.1f}°, {angle[1]:+.1f}°)")

    print()
    print("=" * 55)
    print("轨迹保留在画面中，浏览器可继续查看")
    print("Ctrl+C 退出")
    print("=" * 55)

    # 保持运行，让用户在浏览器查看轨迹
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        ser.close()


if __name__ == "__main__":
    main()

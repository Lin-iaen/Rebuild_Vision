"""云台坐标系标定模块。

通过发送已知角度指令，记录激光轨迹，反解出 2×2 变换矩阵。
标定结果保存到文件，下次启动可自动加载。
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time

import cv2
import numpy as np

from camera import Camera
from tracker import process_laser_detection
from uart import UartController

logger = logging.getLogger(__name__)

# ===== 协议常量 =====
HEADER_X = bytes([0x02, 0x01])
HEADER_Y = bytes([0x02, 0x02])
ANGLE_SCALE = 100

# ===== 标定参数 =====
CALIB_ANGLE = 30.0
SETTLE_TIME = 3.0
SAMPLE_COUNT = 10

# ===== 颜色 (BGR) =====
COLOR_X = (255, 128, 0)
COLOR_Y = (0, 60, 255)
COLOR_CURRENT = (0, 255, 0)
COLOR_TEXT = (0, 255, 255)

# 默认标定文件路径
DEFAULT_CALIB_FILE = "calibration.json"


def load_calibration(path: str = DEFAULT_CALIB_FILE) -> np.ndarray | None:
    """从文件加载 M⁻¹ 矩阵。文件不存在返回 None。"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        M_inv = np.array(data["M_inv"])
        logger.info(f"已加载标定文件: {path}")
        return M_inv
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def save_calibration(M_inv: np.ndarray, path: str = DEFAULT_CALIB_FILE) -> None:
    """保存 M⁻¹ 矩阵到文件。"""
    data = {"M_inv": M_inv.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"标定结果已保存: {path}")


class Calibrator:
    """云台坐标系标定器。

    用法::

        calibrator = Calibrator(cam, uart)
        M_inv = calibrator.run()
        # 或在标定过程中通过 annotate() 获取可视化帧
    """

    def __init__(self, cam: Camera, uart: UartController) -> None:
        self._cam = cam
        self._uart = uart

        # 轨迹状态
        self._trail_x: list[tuple[float, float, float]] = []
        self._trail_y: list[tuple[float, float, float]] = []
        self._trail_phase: str = "idle"
        self._trail_lock = threading.Lock()

        # 可视化状态
        self._current_pos: tuple[float, float] | None = None
        self._status_text: str = "Calib ready"
        self._latest_annotated: np.ndarray | None = None
        self._annotated_lock = threading.Lock()

        # 记录线程
        self._recorder_thread: threading.Thread | None = None
        self._recording = False

    def run(self) -> np.ndarray | None:
        """执行标定流程，返回 M⁻¹ 矩阵。失败返回 None。"""
        # 启动记录线程
        self._recording = True
        self._recorder_thread = threading.Thread(
            target=self._record_loop, daemon=True
        )
        self._recorder_thread.start()

        try:
            return self._do_calibration()
        finally:
            self._recording = False

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """绘制标定轨迹（供 Web 监控使用）。"""
        annotated = frame.copy()

        with self._trail_lock:
            trail_x = list(self._trail_x)
            trail_y = list(self._trail_y)
            pos = self._current_pos

        # X 轴轨迹（蓝色）
        self._draw_trail(annotated, trail_x, COLOR_X)

        # Y 轴轨迹（红色）
        self._draw_trail(annotated, trail_y, COLOR_Y)

        # 当前位置
        if pos is not None:
            xi, yi = int(pos[0]), int(pos[1])
            cv2.drawMarker(
                annotated, (xi, yi), COLOR_CURRENT, cv2.MARKER_CROSS, 20, 2
            )
            cv2.putText(
                annotated, f"({xi},{yi})", (xi + 12, yi - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CURRENT, 1,
            )

        # 状态文字
        cv2.putText(
            annotated, self._status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2,
        )

        # 轨迹点数
        cv2.putText(
            annotated, f"X:{len(trail_x)}pts  Y:{len(trail_y)}pts",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        return annotated

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _record_loop(self) -> None:
        """后台线程：持续检测激光位置，追加到轨迹列表。"""
        while self._recording:
            frame = self._cam.read()
            if frame is None:
                time.sleep(0.03)
                continue

            pos, _ = process_laser_detection(frame)
            now = time.time()

            with self._trail_lock:
                self._current_pos = pos
                if pos is not None:
                    if self._trail_phase == "x_axis":
                        self._trail_x.append((now, pos[0], pos[1]))
                    elif self._trail_phase == "y_axis":
                        self._trail_y.append((now, pos[0], pos[1]))

            time.sleep(0.03)

    def _send_axis(self, header: bytes, angle_deg: float) -> None:
        """发送单轴角度指令。"""
        value = int(round(angle_deg * ANGLE_SCALE))
        value = max(-32768, min(32767, value))
        data = struct.pack(">h", value)
        frame = header + data
        self._uart.send_raw(frame)
        logger.info(f"发送: {header.hex()} | 角度={angle_deg:.1f}° | 帧={frame.hex()}")

    def _sample_pos(self, n: int = SAMPLE_COUNT) -> tuple[float, float] | None:
        """多次采样激光位置，返回平均坐标。"""
        positions = []
        for _ in range(n):
            frame = self._cam.read()
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
        return float(mean[0]), float(mean[1])

    def _trail_endpoint(
        self, trail: list[tuple[float, float, float]], tail_n: int = 15
    ) -> tuple[float, float] | None:
        """取轨迹尾部 N 帧的平均值作为终点。"""
        if len(trail) < 3:
            return None
        tail = trail[-tail_n:]
        arr = np.array([(x, y) for _, x, y in tail])
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1])

    def _do_calibration(self) -> np.ndarray | None:
        """执行实际的标定流程。"""
        # 步骤 1: 采样初始位置
        self._status_text = "[1/3] Sampling initial..."
        print()
        print(self._status_text)

        pos0 = self._sample_pos()
        if pos0 is None:
            print("错误：未检测到激光点")
            return None
        x0, y0 = pos0
        print(f"  初始位置: ({x0:.1f}, {y0:.1f})")

        # 步骤 2: X 轴
        self._status_text = f"[2/3] X +{CALIB_ANGLE} deg..."
        print()
        print(self._status_text)

        with self._trail_lock:
            self._trail_x.clear()
            self._trail_phase = "x_axis"

        self._send_axis(HEADER_X, CALIB_ANGLE)

        for remaining in range(int(SETTLE_TIME), 0, -1):
            self._status_text = f"[2/3] X +{CALIB_ANGLE}° wait {remaining}s..."
            time.sleep(1)

        with self._trail_lock:
            x_trail = list(self._trail_x)

        pos1 = self._trail_endpoint(x_trail)
        if pos1 is None:
            print("错误：X 轴轨迹数据不足")
            return None

        x1, y1 = pos1
        dx1, dy1 = x1 - x0, y1 - y0
        print(f"  轨迹点数: {len(x_trail)}")
        print(f"  终点: ({x1:.1f}, {y1:.1f})")
        print(f"  偏移: Δpx={dx1:+.1f}, Δpy={dy1:+.1f}")

        # 步骤 3: Y 轴
        self._status_text = f"[3/3] Y +{CALIB_ANGLE} deg..."
        print()
        print(self._status_text)

        with self._trail_lock:
            self._trail_y.clear()
            self._trail_phase = "y_axis"

        self._send_axis(HEADER_Y, CALIB_ANGLE)

        for remaining in range(int(SETTLE_TIME), 0, -1):
            self._status_text = f"[3/3] Y +{CALIB_ANGLE}° wait {remaining}s..."
            time.sleep(1)

        with self._trail_lock:
            y_trail = list(self._trail_y)
            self._trail_phase = "idle"

        pos2 = self._trail_endpoint(y_trail)
        if pos2 is None:
            print("错误：Y 轴轨迹数据不足")
            return None

        x2, y2 = pos2
        dx2, dy2 = x2 - x1, y2 - y1
        print(f"  轨迹点数: {len(y_trail)}")
        print(f"  终点: ({x2:.1f}, {y2:.1f})")
        print(f"  偏移: Δpx={dx2:+.1f}, Δpy={dy2:+.1f}")

        # 计算变换矩阵
        a = dx1 / CALIB_ANGLE
        c = dy1 / CALIB_ANGLE
        b = dx2 / CALIB_ANGLE
        d = dy2 / CALIB_ANGLE

        M = np.array([[a, b], [c, d]])
        det = a * d - b * c

        if abs(det) < 1e-6:
            print("\n错误：矩阵奇异，无法求逆")
            self._status_text = "Calib failed"
            return None

        M_inv = np.array([[d, -b], [-c, a]]) / det

        # 输出结果
        self._status_text = "Calib done"
        print()
        print("=" * 50)
        print("标定结果")
        print("=" * 50)
        print(f"正向矩阵 M:")
        print(f"  [{a:8.4f}  {b:8.4f}]")
        print(f"  [{c:8.4f}  {d:8.4f}]")
        print(f"逆矩阵 M⁻¹:")
        print(f"  [{M_inv[0,0]:8.4f}  {M_inv[0,1]:8.4f}]")
        print(f"  [{M_inv[1,0]:8.4f}  {M_inv[1,1]:8.4f}]")
        print("=" * 50)

        return M_inv

    @staticmethod
    def _draw_trail(
        annotated: np.ndarray,
        trail: list[tuple[float, float, float]],
        color: tuple,
    ) -> None:
        """绘制轨迹折线。"""
        if len(trail) < 2:
            return

        points = [(int(x), int(y)) for _, x, y in trail]

        for i in range(1, len(points)):
            thickness = 2 if i > len(points) - 10 else 1
            cv2.line(annotated, points[i - 1], points[i], color, thickness)

        cv2.circle(annotated, points[0], 8, color, 2)

        ex, ey = points[-1]
        cv2.rectangle(annotated, (ex - 6, ey - 6), (ex + 6, ey + 6), color, -1)

        if len(points) >= 20:
            step = len(points) // 4
            for i in range(step, len(points) - 1, step):
                p1 = points[i]
                p2 = points[min(i + 5, len(points) - 1)]
                cv2.arrowedLine(annotated, p1, p2, color, 2, tipLength=0.3)

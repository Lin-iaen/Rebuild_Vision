"""云台坐标系标定模块。

通过发送已知绝对位置指令（CAN ID 0x0173），记录激光轨迹，反解出 2×2 变换矩阵。
标定结果保存到文件，下次启动可自动加载。
"""

from __future__ import annotations

import json
import logging
import threading
import time

import cv2
import numpy as np

from camera import Camera
from motor import MotorController
from tracker import process_laser_detection

logger = logging.getLogger(__name__)

# ===== 标定参数 =====
CALIB_SPEED = 2.0       # 标定角速度 (°/s)
SETTLE_TIME = 3.0       # 单轴运动时间 (s), dθ = CALIB_SPEED × SETTLE_TIME
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

        calibrator = Calibrator(cam, motor)
        M_inv = calibrator.run()
        # 或在标定过程中通过 annotate() 获取可视化帧
    """

    def __init__(self, cam: Camera, motor: MotorController) -> None:
        self._cam = cam
        self._motor = motor

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

    def _send_position(self, azimuth_deg: float, pitch_deg: float) -> None:
        """发送绝对位置指令（CAN ID 0x0173）。"""
        self._motor.set_position(azimuth_deg, pitch_deg)
        logger.info(
            "发送位置指令: az=%.1f°  pt=%.1f°", azimuth_deg, pitch_deg
        )

    def _start_move(self, az_dps: float, pt_dps: float) -> None:
        """开始匀速运动（CAN ID 0x0682）。"""
        self._motor.set_speed(az_dps, pt_dps)
        logger.info("发送速度指令: az=%.1f°/s  pt=%.1f°/s", az_dps, pt_dps)

    def _stop_move(self) -> None:
        """停止运动。"""
        self._motor.stop()
        logger.info("发送停止指令")

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
        """执行实际的标定流程。

        流程:
            Step 0: 绝对位置归零 (0°, 0°), 停止, 采样 pos0
            Step 1: az 轴匀速 CALIB_SPEED, pt=0, 待 SETTLE_TIME, 停止, 采样 pos1
            Step 2: pt 轴匀速 CALIB_SPEED, az=0, 待 SETTLE_TIME, 停止, 采样 pos2
            偏转角: dθ = CALIB_SPEED × SETTLE_TIME
        """
        dtheta = CALIB_SPEED * SETTLE_TIME

        # ---- Step 0: 归零并采样 ----
        self._status_text = "[0/3] Zeroing to (0, 0)..."
        print(f"\n{self._status_text}")

        self._send_position(0.0, 0.0)
        print(f"  等待云台归位中... (0°, 0°)")
        time.sleep(SETTLE_TIME)
        self._stop_move()

        pos0 = self._sample_pos()
        if pos0 is None:
            print("错误：未检测到激光点")
            return None
        x0, y0 = pos0
        print(f"  初始位置: ({x0:.1f}, {y0:.1f})")

        # ---- Step 1: az 轴匀速, pt=0 ----
        self._status_text = (
            f"[1/3] az {CALIB_SPEED}°/s × {SETTLE_TIME}s = {dtheta}° (pt=0)..."
        )
        print(f"\n{self._status_text}")

        with self._trail_lock:
            self._trail_x.clear()
            self._trail_phase = "x_axis"

        self._start_move(CALIB_SPEED, 0.0)

        for remaining in range(int(SETTLE_TIME), 0, -1):
            self._status_text = (
                f"[1/3] az {CALIB_SPEED}°/s  wait {remaining}s..."
            )
            time.sleep(1)

        self._stop_move()

        # 等电机完全停稳
        time.sleep(0.5)

        with self._trail_lock:
            x_trail = list(self._trail_x)

        pos1 = self._trail_endpoint(x_trail)
        if pos1 is None:
            print("错误：az 轴轨迹数据不足")
            return None

        x1, y1 = pos1
        dx1, dy1 = x1 - x0, y1 - y0
        print(f"  轨迹点数: {len(x_trail)}")
        print(f"  终点: ({x1:.1f}, {y1:.1f})")
        print(f"  偏移: Δpx={dx1:+.1f}, Δpy={dy1:+.1f}")

        # ---- Step 2: pt 轴匀速, az=0 ----
        self._status_text = (
            f"[2/3] pt {CALIB_SPEED}°/s × {SETTLE_TIME}s = {dtheta}° (az=0)..."
        )
        print(f"\n{self._status_text}")

        with self._trail_lock:
            self._trail_y.clear()
            self._trail_phase = "y_axis"

        self._start_move(0.0, CALIB_SPEED)

        for remaining in range(int(SETTLE_TIME), 0, -1):
            self._status_text = (
                f"[2/3] pt {CALIB_SPEED}°/s  wait {remaining}s..."
            )
            time.sleep(1)

        self._stop_move()
        time.sleep(0.5)

        with self._trail_lock:
            y_trail = list(self._trail_y)
            self._trail_phase = "idle"

        pos2 = self._trail_endpoint(y_trail)
        if pos2 is None:
            print("错误：pt 轴轨迹数据不足")
            return None

        x2, y2 = pos2
        dx2, dy2 = x2 - x1, y2 - y1
        print(f"  轨迹点数: {len(y_trail)}")
        print(f"  终点: ({x2:.1f}, {y2:.1f})")
        print(f"  偏移: Δpx={dx2:+.1f}, Δpy={dy2:+.1f}")

        # ----- 计算变换矩阵 -----
        a = dx1 / dtheta
        c = dy1 / dtheta
        b = dx2 / dtheta
        d = dy2 / dtheta

        M = np.array([[a, b], [c, d]])
        det = a * d - b * c

        if abs(det) < 1e-6:
            print("\n错误：矩阵奇异，无法求逆")
            self._status_text = "Calib failed"
            return None

        M_inv = np.array([[d, -b], [-c, a]]) / det

        # ---- 输出结果 ----
        self._status_text = "Calib done"
        print()
        print("=" * 50)
        print(f"标定结果  (dθ = {CALIB_SPEED}°/s × {SETTLE_TIME}s = {dtheta}°)")
        print("=" * 50)
        print("正向矩阵 M:")
        print(f"  [{a:8.4f}  {b:8.4f}]")
        print(f"  [{c:8.4f}  {d:8.4f}]")
        print("逆矩阵 M⁻¹:")
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

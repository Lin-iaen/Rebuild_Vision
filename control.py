"""激光循迹控制模块。

实现连续反馈循环：检测激光 → 计算误差 → M⁻¹ 变换 → 发送指令 → 循环。
提供复位到中心和绕矩形循迹两种模式。
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from camera import Camera
from gimbal import GimbalController
from rectangle import RectangleManager
from tracker import process_laser_detection

logger = logging.getLogger(__name__)

# 到达目标的判定阈值（像素）
DEFAULT_THRESHOLD = 10.0


class LaserTracker:
    """激光循迹控制器。

    通过摄像头反馈实现闭环控制，将激光移动到目标位置。
    """

    def __init__(
        self,
        cam: Camera,
        gimbal: GimbalController,
        rect_mgr: RectangleManager,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._cam = cam
        self._gimbal = gimbal
        self._rect = rect_mgr
        self._threshold = threshold

        # 状态（供 Web 监控读取）
        self._current_pos: tuple[float, float] | None = None
        self._current_target: tuple[float, float] | None = None
        self._target_index: int = 0
        self._total_targets: int = 0
        self._path_history: list[tuple[float, float]] = []
        self._status_text: str = "Ready"
        self._lock = threading.Lock()

        # 控制循环线程
        self._running = False

    @property
    def status_text(self) -> str:
        with self._lock:
            return self._status_text

    @property
    def current_pos(self) -> tuple[float, float] | None:
        with self._lock:
            return self._current_pos

    @property
    def current_target(self) -> tuple[float, float] | None:
        with self._lock:
            return self._current_target

    @property
    def target_index(self) -> int:
        with self._lock:
            return self._target_index

    @property
    def total_targets(self) -> int:
        with self._lock:
            return self._total_targets

    @property
    def path_history(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._path_history)

    def move_to_target(self, target: tuple[float, float]) -> bool:
        """移动激光到目标位置，连续反馈循环。

        使用简单噪声剔除：距离上次位置超过阈值则判定为噪声。
        检测失败时发送零指令（云台不移动）。
        返回: 是否成功到达
        """
        with self._lock:
            self._current_target = target
            self._status_text = f"Move to ({target[0]:.0f}, {target[1]:.0f})"

        last_valid_pos = None
        NOISE_THRESHOLD = 50.0  # 像素，超过此距离判定为噪声

        while self._running:
            frame = self._cam.read()
            if frame is None:
                continue

            pos, _ = process_laser_detection(frame)

            if pos is not None:
                # 检测成功
                if last_valid_pos is None:
                    # 首次检测，直接接受
                    last_valid_pos = pos
                    filtered_pos = pos
                elif np.linalg.norm(np.array(pos) - np.array(last_valid_pos)) > NOISE_THRESHOLD:
                    # 异常值（噪声），丢弃，不发送指令
                    continue
                else:
                    # 正常检测
                    last_valid_pos = pos
                    filtered_pos = pos

                with self._lock:
                    self._current_pos = filtered_pos

                error = np.array(target) - np.array(filtered_pos)
                dist = np.linalg.norm(error)

                if dist < self._threshold:
                    with self._lock:
                        self._path_history.append(filtered_pos)
                        self._current_pos = None
                        self._current_target = None
                        self._status_text = f"Reached ({target[0]:.0f}, {target[1]:.0f})"
                    return True

                self._gimbal.move(error[0], error[1])
            else:
                # 检测失败，发送零指令（云台不移动）
                self._gimbal.move(0, 0)

            time.sleep(0.05)  # 50ms 延时 (20fps)

        return False

    def reset_to_center(self) -> bool:
        """复位激光到矩形中心。"""
        center = self._rect.get_center()
        if center is None:
            with self._lock:
                self._status_text = "Error: No rect"
            return False

        with self._lock:
            self._status_text = "Resetting..."
            self._path_history.clear()

        return self.move_to_target(center)

    def track_rectangle(self) -> None:
        """绕矩形循迹一圈。"""
        targets = self._rect.get_targets()
        if not targets:
            with self._lock:
                self._status_text = "Error: No targets"
            return

        with self._lock:
            self._total_targets = len(targets)
            self._target_index = 0
            self._path_history.clear()
            self._status_text = f"Tracking: {len(targets)} targets"

        for i, target in enumerate(targets):
            if not self._running:
                break

            with self._lock:
                self._target_index = i + 1
                self._status_text = f"Track [{i + 1}/{len(targets)}]"

            self.move_to_target(target)

        with self._lock:
            if self._running:
                self._status_text = "Track done"
            self._current_pos = None
            self._current_target = None

    def start_reset(self) -> threading.Thread:
        """在后台线程中执行复位。"""
        self._running = True
        thread = threading.Thread(target=self.reset_to_center, daemon=True)
        thread.start()
        return thread

    def start_track(self) -> threading.Thread:
        """在后台线程中执行循迹。"""
        self._running = True
        thread = threading.Thread(target=self.track_rectangle, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """停止循迹。"""
        self._running = False
        with self._lock:
            self._current_pos = None
            self._current_target = None
            self._status_text = "Stopped"

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """在画面上绘制循迹状态。"""
        annotated = frame.copy()

        with self._lock:
            pos = self._current_pos
            target = self._current_target
            path = list(self._path_history)
            targets_total = self._total_targets
            target_idx = self._target_index
            status = self._status_text

        # 矩形边框
        corners = self._rect.get_ordered_corners()
        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (255, 128, 0), 2)

        # 子目标点（灰色）
        all_targets = self._rect.get_targets()
        for t in all_targets:
            cv2.circle(annotated, (int(t[0]), int(t[1])), 3, (128, 128, 128), -1)

        # 当前目标（黄色）
        if target:
            tx, ty = int(target[0]), int(target[1])
            cv2.circle(annotated, (tx, ty), 6, (0, 255, 255), 2)

        # 已走路径（绿色）
        if len(path) >= 2:
            for i in range(1, len(path)):
                p1 = (int(path[i - 1][0]), int(path[i - 1][1]))
                p2 = (int(path[i][0]), int(path[i][1]))
                cv2.line(annotated, p1, p2, (0, 200, 0), 2)

        # 当前激光位置（绿色十字）
        if pos:
            px, py = int(pos[0]), int(pos[1])
            cv2.drawMarker(annotated, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # 状态信息
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if targets_total > 0:
            cv2.putText(annotated, f"Progress: {target_idx}/{targets_total}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if pos and target:
            error = np.linalg.norm(np.array(target) - np.array(pos))
            cv2.putText(annotated, f"Error: {error:.1f}px", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated

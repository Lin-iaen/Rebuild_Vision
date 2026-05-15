"""卡尔曼滤波跟踪模块。

用于激光位置预测和平滑，抑制噪声误检。
状态：[x, y, vx, vy]，测量：[x, y]
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_PROCESS_NOISE = 1e-2      # 过程噪声
DEFAULT_MEASUREMENT_NOISE = 1e-1  # 测量噪声
DEFAULT_OUTLIER_THRESHOLD = 50.0  # 异常值阈值（像素）


class KalmanTracker:
    """卡尔曼滤波跟踪器。

    用法::

        tracker = KalmanTracker()
        tracker.init((320, 240))      # 用首次检测初始化
        tracker.predict()             # 预测下一帧位置
        tracker.update((325, 238))    # 用测量值更新
        pos = tracker.get_state()     # 获取当前估计位置
        is_bad = tracker.is_outlier((500, 100))  # 判断是否为异常值
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ) -> None:
        # cv2.KalmanFilter(状态维度, 测量维度, 控制维度)
        self._kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 F
        # [x, y, vx, vy] -> [x+vx*dt, y+vy*dt, vx, vy]
        # dt = 1 (一帧)，由 predict 时自动更新
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # 测量矩阵 H
        # 测量值 = [x, y]，从状态中提取
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # 过程噪声协方差 Q
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 测量噪声协方差 R
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 后验误差协方差 P（初始值较大，表示不确定）
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """是否已初始化。"""
        return self._initialized

    def init(self, pos: tuple[float, float]) -> None:
        """用首次检测位置初始化滤波器。"""
        self._kf.statePost = np.array(
            [[pos[0]], [pos[1]], [0], [0]],  # 初始速度为 0
            dtype=np.float32,
        )
        self._initialized = True

    def predict(self) -> tuple[float, float]:
        """预测下一帧位置。返回预测坐标。"""
        predicted = self._kf.predict()
        return float(predicted[0, 0]), float(predicted[1, 0])

    def update(self, measurement: tuple[float, float]) -> None:
        """用测量值更新滤波器状态。"""
        measurement_vec = np.array(
            [[measurement[0]], [measurement[1]]],
            dtype=np.float32,
        )
        self._kf.correct(measurement_vec)

    def get_state(self) -> tuple[float, float]:
        """获取当前估计位置（后验状态）。"""
        state = self._kf.statePost
        return float(state[0, 0]), float(state[1, 0])

    def is_outlier(
        self, measurement: tuple[float, float], threshold: float = DEFAULT_OUTLIER_THRESHOLD
    ) -> bool:
        """判断测量值是否为异常值。

        通过计算测量值与预测值的距离来判断。
        超过阈值则判定为异常（噪声）。
        """
        predicted = self.predict()
        dist = np.linalg.norm(
            np.array(measurement) - np.array(predicted)
        )
        return dist > threshold

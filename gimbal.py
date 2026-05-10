"""云台控制模块。

封装新云台协议，通过 M⁻¹ 矩阵将像素误差转换为角度指令。
协议格式：[帧头1] [帧头2] [角度高字节] [角度低字节]，共 4 字节。
"""

from __future__ import annotations

import logging
import struct

import numpy as np

from uart import UartController

logger = logging.getLogger(__name__)

# 协议常量
HEADER_X = bytes([0x02, 0x01])  # X轴帧头
HEADER_Y = bytes([0x02, 0x02])  # Y轴帧头
ANGLE_SCALE = 100  # 角度 × 100 转为整数

# 默认 M⁻¹ 矩阵（单位矩阵：不做变换）
DEFAULT_M_INV = np.array([[1.0, 0.0], [0.0, 1.0]])


class GimbalController:
    """云台控制器。

    将像素误差通过 M⁻¹ 矩阵转换为角度指令，分别发送给 X/Y 轴。
    """

    def __init__(
        self,
        uart: UartController,
        M_inv: np.ndarray | None = None,
    ) -> None:
        """
        参数:
            uart: 串口控制器
            M_inv: 2×2 逆变换矩阵（像素→角度），None 则使用单位矩阵
        """
        self._uart = uart
        self._M_inv = M_inv if M_inv is not None else DEFAULT_M_INV.copy()

    def move(self, delta_px: float, delta_py: float) -> None:
        """根据像素误差发送云台角度指令。

        流程: [Δpx, Δpy] → M⁻¹ 变换 → 分别发送 X/Y 轴角度
        """
        error = np.array([delta_px, delta_py])
        angles = self._M_inv @ error

        self._send_axis(HEADER_X, float(angles[0]))
        self._send_axis(HEADER_Y, float(angles[1]))

    def _send_axis(self, header: bytes, angle_deg: float) -> None:
        """发送单轴角度指令。"""
        value = int(round(angle_deg * ANGLE_SCALE))
        value = max(-32768, min(32767, value))
        data = struct.pack(">h", value)
        frame = header + data
        self._uart.send_raw(frame)

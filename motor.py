"""云台电机 CAN 控制模块。

封装 CAN 帧协议，提供速度/位置控制接口，不耦合上层追踪逻辑。

ID 语义:
    0x0682 - 速度模式: az/pt 在 payload bytes 4-7, 值/100 = °/s
    0x0173 - 位置模式: az/pt 在 payload bytes 0-3, 值/100 = °
值 ×100 = 角度，int16 big-endian，帧总长 10 字节。
"""

from __future__ import annotations

from uart import UartController

CAN_ID_SPEED = 0x0682
CAN_ID_POSITION = 0x0173
ANGLE_SCALE = 100


def _clamp_i16(value: int) -> int:
    return max(-32768, min(32767, value))


class MotorController:
    """云台电机 CAN 控制器。"""

    def __init__(self, uart: UartController) -> None:
        self._uart = uart

    def set_position(self, azimuth_deg: float, pitch_deg: float) -> None:
        """绝对位置控制 (ID 0x0173)。

        帧结构:
            [0x01 0x73] [az×100 i16] [pt×100 i16] [0x00 0x00 0x00 0x00]
        """
        az = _clamp_i16(int(round(azimuth_deg * ANGLE_SCALE)))
        pt = _clamp_i16(int(round(pitch_deg * ANGLE_SCALE)))

        frame = bytearray()
        frame.extend(CAN_ID_POSITION.to_bytes(2, "big"))
        frame.extend(az.to_bytes(2, "big", signed=True))
        frame.extend(pt.to_bytes(2, "big", signed=True))
        frame.extend(b"\x00\x00\x00\x00")
        self._uart.send_raw(bytes(frame))

    def set_speed(self, azimuth_dps: float, pitch_dps: float) -> None:
        """角速度控制 (ID 0x0682)。

        帧结构:
            [0x06 0x82] [0] [0] [az_dps×100 i16] [pt_dps×100 i16]
        """
        az = _clamp_i16(int(round(azimuth_dps * ANGLE_SCALE)))
        pt = _clamp_i16(int(round(pitch_dps * ANGLE_SCALE)))

        frame = bytearray()
        frame.extend(CAN_ID_SPEED.to_bytes(2, "big"))
        frame.extend((0).to_bytes(2, "big", signed=True))
        frame.extend((0).to_bytes(2, "big", signed=True))
        frame.extend(az.to_bytes(2, "big", signed=True))
        frame.extend(pt.to_bytes(2, "big", signed=True))
        self._uart.send_raw(bytes(frame))

    def stop(self) -> None:
        """急停 — 发送零速度。"""
        self.set_speed(0.0, 0.0)

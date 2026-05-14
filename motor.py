"""云台电机 CAN 控制模块。

封装 CAN 帧协议，提供速度/位置控制接口，不耦合上层追踪逻辑。

ID 语义:
    发送:
        0x0682 - 速度模式: az/pt 在 payload bytes 4-7, 值/100 = °/s
        0x0173 - 位置模式: az/pt 在 payload bytes 0-3, 值/100 = °
    回传:
        0x201  - 方位角电机位置: bytes 12-13, int16 大端, 值/100 = °
        0x202  - 俯仰电机位置:   bytes 12-13, int16 大端, 值/100 = °

帧总长: 发送 10 字节，回传 20 字节（9×0FF 同步头 + DLC + ID + 数据）。
"""

from __future__ import annotations

from uart import UartController

CAN_ID_SPEED = 0x0682
CAN_ID_POSITION = 0x0173
CAN_ID_FB_AZ = 0x201       # 方位角电机回传 ID
CAN_ID_FB_PT = 0x202       # 俯仰电机回传 ID
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
        pt = _clamp_i16(int(round(-pitch_dps * ANGLE_SCALE)))

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

    def read_feedback(self, timeout: float = 0.0) -> tuple[int, float] | None:
        """读取一个电机回传帧。

        回传帧格式 (20 字节):
            9×0FF 同步头 + DLC + StdID(2B) + 数据(8B)
            数据 bytes 12-13: int16 大端, 值/100 = °

        参数:
            timeout: 等待超时(秒)。0=不等待。

        返回: (motor_id, position_deg) 或 None
            motor_id: CAN_ID_FB_AZ (0x201) 或 CAN_ID_FB_PT (0x202)
            position_deg: 绝对角度 (°)
        """
        frame = self._uart.read_can_frame(timeout=timeout)
        if frame is None:
            return None

        std_id, data = frame

        if std_id not in (CAN_ID_FB_AZ, CAN_ID_FB_PT):
            return None

        # bytes 12-13: int16 大端, 值/100 = °
        raw = (data[0] << 8) | data[1]
        if raw > 32767:
            raw -= 65536
        pos_deg = raw / 100.0

        return (std_id, pos_deg)

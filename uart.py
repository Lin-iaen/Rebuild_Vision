"""通用串口通信模块。

提供底层串口收发能力，不耦合任何协议格式。
协议解析由上层模块（如 gimbal.py）负责。
"""

from __future__ import annotations

import logging
import serial


logger = logging.getLogger(__name__)


class UartController:
    """通用串口控制器。"""

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial: serial.Serial | None = None
        self._connect()

    def _connect(self) -> None:
        """尝试打开物理串口。"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
            )
            logger.info(f"串口 {self.port} 打开成功，波特率: {self.baudrate}")
        except serial.SerialException as e:
            logger.error(f"无法打开串口 {self.port}: {e}")
            self.serial = None

    def send_raw(self, data: bytes) -> None:
        """发送原始字节数据。"""
        if self.serial is None or not self.serial.is_open:
            logger.warning("串口未开启，跳过发送")
            return
        try:
            self.serial.write(data)
            hex_str = " ".join(f"{b:02X}" for b in data)
            logger.debug(f"UART 发送: [{hex_str}]")
        except Exception as e:
            logger.error(f"串口发送异常: {e}")

    def close(self) -> None:
        """关闭串口。"""
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            logger.info("串口已关闭")

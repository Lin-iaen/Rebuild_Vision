"""通用串口通信模块。

提供底层串口收发能力，不耦合任何协议格式。
协议解析由上层模块（如 motor.py）负责。
"""

from __future__ import annotations

import logging
import time

import serial


logger = logging.getLogger(__name__)


class UartController:
    """通用串口控制器。"""

    def __init__(
        self,
        port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00",
        baudrate: int = 115200,
        dtr: bool | None = None,
        rts: bool | None = None,
        open_delay: float = 0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self._dtr = dtr
        self._rts = rts
        self._open_delay = open_delay
        self.serial: serial.Serial | None = None
        self._connect()

    def _connect(self) -> None:
        """尝试打开物理串口。"""
        try:
            self.serial = serial.Serial()
            self.serial.port = self.port
            self.serial.baudrate = self.baudrate
            self.serial.timeout = 0.1

            # 禁用硬件控制线，防止 STM32 一直处于复位状态
            if self._dtr is not None:
                self.serial.dtr = self._dtr
            if self._rts is not None:
                self.serial.rts = self._rts

            self.serial.open()

            if self._open_delay > 0:
                time.sleep(self._open_delay)

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

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
        self._rx_buf = bytearray()  # 接收缓冲区
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

    def read_can_frame(self, timeout: float = 0.0) -> tuple[int, bytes] | None:
        """读取一个 20 字节 CAN 回传帧。

        帧格式: 9×0xFF 同步头 + DLC + StdID(2B) + 数据(8B) = 20 字节。
        扫描同步头，解析 ID 和数据段。

        参数:
            timeout: 等待超时(秒)。0=不等待，立即返回。

        返回: (std_id, data_8bytes) 或 None（无完整帧）
        """
        if self.serial is None or not self.serial.is_open:
            return None

        # 读取可用字节到缓冲区
        n = self.serial.in_waiting
        if n > 0:
            self._rx_buf.extend(self.serial.read(n))
        elif timeout > 0:
            time.sleep(timeout)
            n = self.serial.in_waiting
            if n > 0:
                self._rx_buf.extend(self.serial.read(n))
            else:
                return None
        else:
            return None

        # 扫描 9×0FF 同步头，提取完整帧
        while len(self._rx_buf) >= 20:
            # 寻找 9×0xFF
            pos = -1
            for i in range(len(self._rx_buf) - 8):
                if all(self._rx_buf[i + j] == 0xFF for j in range(9)):
                    pos = i
                    break

            if pos < 0:
                # 未找到同步头，保留尾部 8 字节（可能是跨包的同步头）
                if len(self._rx_buf) > 8:
                    del self._rx_buf[:-8]
                break

            # 删除同步头之前的垃圾字节
            if pos > 0:
                del self._rx_buf[:pos]

            # 不足 20 字节，等待下次读取
            if len(self._rx_buf) < 20:
                break

            # 解析帧
            dlc = (self._rx_buf[9] >> 4) & 0x0F
            std_id = (self._rx_buf[10] << 8) | self._rx_buf[11]
            data = bytes(self._rx_buf[12:20])

            # 从缓冲区删除已解析的帧
            del self._rx_buf[:20]

            logger.debug(
                "CAN 回传: ID=0x%03X DLC=%d data=[%s]",
                std_id, dlc, " ".join(f"{b:02X}" for b in data),
            )

            return (std_id, data)

        return None

    def close(self) -> None:
        """关闭串口。"""
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            logger.info("串口已关闭")

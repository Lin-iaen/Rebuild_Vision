"""UART 通信模块（真实串口驱动）。"""

from __future__ import annotations

import logging
import struct
import time
import serial


logger = logging.getLogger(__name__)


class UartController:
    """真实物理串口控制器。"""

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200) -> None:
        self._last_log_time = 0.0
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

    def send_error(self, dx: float, dy: float) -> None:
        """打包并发送闭环微调误差。"""
        # 日志节流阀（每秒打印 5 次）
        now = time.time()
        should_log = (now - self._last_log_time) >= 0.2
        if should_log:
            self._last_log_time = now

        if self.serial is None or not self.serial.is_open:
            if should_log:
                logger.warning("串口未开启，跳过发送")
            return

        # 1. 数据缩放：放大 10 倍并转为整数，保留 1 位小数精度
        # 例如: -15.6 -> -156
        idx = int(round(dx * 10))
        idy = int(round(dy * 10))

        # 限制范围在 int16 以内 (-32768 ~ 32767)
        idx = max(-32768, min(32767, idx))
        idy = max(-32768, min(32767, idy))

        # 2. 结构化打包 (Big-Endian 大端模式)
        # > : 大端模式 (高位在前，低位在后)
        # B : unsigned char (1字节)
        # h : short (2字节，有符号)
        try:
            # 先打包前 6 个字节
            header_data = struct.pack('>BBhh', 0xAA, 0x55, idx, idy)
            
            # 3. 计算 Checksum (累加和校验)
            checksum = sum(header_data) & 0xFF
            
            # 4. 组装完整帧：帧头数据 + 校验和 + 帧尾 0x0A
            frame = header_data + struct.pack('>BB', checksum, 0x0A)
            
            # 5. 发送物理字节
            self.serial.write(frame)

            if should_log:
                # 打印 Hex 格式方便对比调试
                hex_str = " ".join([f"{b:02X}" for b in frame])
                logger.info(f"UART 发送: dx={dx:.1f}, dy={dy:.1f} | Frame: [{hex_str}]")
                
        except Exception as e:
            logger.error(f"串口发送异常: {e}")
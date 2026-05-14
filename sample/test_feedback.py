"""电机回传位置测试脚本。

持续读取 CAN 回传帧，验证电机位置反馈功能。

用法:
    python3 sample/test_feedback.py
    Ctrl+C 退出
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from motor import MotorController, CAN_ID_FB_AZ, CAN_ID_FB_PT
from uart import UartController

# ===== 配置 =====
CAN_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
CAN_BAUD = 4000000


def main():
    print("=" * 60)
    print("电机回传位置测试")
    print("=" * 60)

    uart = UartController(port=CAN_PORT, baudrate=CAN_BAUD,
                          dtr=False, rts=False, open_delay=0.5)
    motor = MotorController(uart)
    print(f"串口: {CAN_PORT} @ {CAN_BAUD}")

    print()
    print("持续读取电机回传位置...  Ctrl+C 退出")
    print("-" * 60)
    print(f"{'时间':>8s}  {'ID':>8s}  {'角度':>10s}")
    print("-" * 60)

    az_pos = None
    pt_pos = None

    try:
        while True:
            fb = motor.read_feedback(timeout=0.1)
            if fb is None:
                time.sleep(0.01)
                continue

            motor_id, pos_deg = fb

            if motor_id == CAN_ID_FB_AZ:
                id_str = "0x201(az)"
                az_pos = pos_deg
            elif motor_id == CAN_ID_FB_PT:
                id_str = "0x202(pt)"
                pt_pos = pos_deg
            else:
                id_str = f"0x{motor_id:03X}"

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"{ts:>8s}  {id_str:>8s}  {pos_deg:>8.2f}°")

            # 收集到一对后打印汇总
            if az_pos is not None and pt_pos is not None:
                print(f"  {'→ 当前位':>8s}  {'  '}  az={az_pos:.2f}°  pt={pt_pos:.2f}°")
                az_pos = None
                pt_pos = None

    except KeyboardInterrupt:
        print()
    finally:
        uart.close()
        print("已退出")


if __name__ == "__main__":
    main()

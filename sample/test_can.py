import serial
import time

# 替换成你的串口路径
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5972089810-if00"
BAUD = 4000000

CAN_FRAME_ID_S = 0x0682
CAN_FRAME_ID_P = 0x0173
ANGLE_SCALE = 100

# ===== 流程参数（可按需调整） =====
RESET_AZ_DEG = 0.0
RESET_PT_DEG = 0.0
RESET_WAIT_S = 3.0

X_SPEED_DPS = 80.0
Y_SPEED_DPS = 12.0
MOVE_DURATION_S = 4.0
STOP_WAIT_S = 0.5


def open_serial(port: str, baud: int) -> serial.Serial:
    """打开串口，禁用 DTR/RTS 防复位。"""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 1.0
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(0.5)
    return ser


def clamp_i16(value: int) -> int:
    """限制到 int16 范围。"""
    return max(-32768, min(32767, value))


def pack_four_int16(values: tuple[int, int, int, int]) -> bytes:
    """将 4 个 int16 打包成 8 字节大端数据区。"""
    payload = bytearray()
    for value in values:
        payload.extend(int(value).to_bytes(2, "big", signed=True))
    return bytes(payload)


def send_can_frame(
    ser: serial.Serial,
    frame_id: int,
    frame_values: tuple[int, int, int, int],
    label: str,
) -> None:
    """发送 CAN 帧: [ID 2B] [8B 数据区(4个 int16)]。"""
    payload_8_bytes = pack_four_int16(frame_values)

    payload = bytearray()
    payload.extend(frame_id.to_bytes(2, "big"))
    payload.extend(payload_8_bytes)
    ser.write(payload)
    ser.flush()
    print(f"[{label}] {payload.hex(' ')}")
    print(
        "        int16 = "
        f"{frame_values[0]}, {frame_values[1]}, {frame_values[2]}, {frame_values[3]}"
    )


def send_position(ser: serial.Serial, azimuth_deg: float, pitch_deg: float) -> None:
    """发送绝对位置指令 (0x0173): [az, pt, 0, 0]。"""
    az = clamp_i16(int(round(azimuth_deg * ANGLE_SCALE)))
    pt = clamp_i16(int(round(pitch_deg * ANGLE_SCALE)))
    send_can_frame(ser, CAN_FRAME_ID_P, (az, pt, 0, 0), "位置指令 0x0173")


def send_speed(ser: serial.Serial, azimuth_dps: float, pitch_dps: float) -> None:
    """发送速度指令 (0x0682): [0, 0, az_dps, pt_dps]。"""
    az = clamp_i16(int(round(azimuth_dps * ANGLE_SCALE)))
    pt = clamp_i16(int(round(pitch_dps * ANGLE_SCALE)))
    send_can_frame(ser, CAN_FRAME_ID_S, (0, 0, az, pt), "速度指令 0x0682")


def stop_move(ser: serial.Serial) -> None:
    """停止运动（发送零速度）。"""
    send_speed(ser, 0.0, 0.0)


def run_sequence(ser: serial.Serial) -> None:
    """按标定节奏执行：复位 -> X 轴 -> Y 轴。"""
    print("\n========== 自动控制流程 ==========")
    print("Step 0: 发送绝对位置复位")
    send_position(ser, RESET_AZ_DEG, RESET_PT_DEG)
    print(f"等待复位完成: {RESET_WAIT_S:.1f}s")
    time.sleep(RESET_WAIT_S)
    stop_move(ser)
    time.sleep(STOP_WAIT_S)

    print("\nStep 1: X 轴运动")
    print(f"速度: az={X_SPEED_DPS:.2f}°/s, pt=0.00°/s, 持续 {MOVE_DURATION_S:.1f}s")
    send_speed(ser, X_SPEED_DPS, 0.0)
    time.sleep(MOVE_DURATION_S)
    stop_move(ser)
    print(f"X 轴停止, 等待停稳: {STOP_WAIT_S:.1f}s")
    time.sleep(STOP_WAIT_S)

    print("\nStep 2: Y 轴运动")
    print(f"速度: az=0.00°/s, pt={Y_SPEED_DPS:.2f}°/s, 持续 {MOVE_DURATION_S:.1f}s")
    send_speed(ser, 0.0, Y_SPEED_DPS)
    time.sleep(MOVE_DURATION_S)
    stop_move(ser)
    print(f"Y 轴停止, 等待停稳: {STOP_WAIT_S:.1f}s")
    time.sleep(STOP_WAIT_S)

    print("\n流程结束。")
    print("==================================")


def main():
    try:
        ser = open_serial(PORT, BAUD)
        print(f"串口已打开: {PORT} @ {BAUD}bps (DTR/RTS = False)")
        run_sequence(ser)

    except Exception as e:
        print(f"错误: {e}")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
import serial
import time

print("--- 物理串口环回测试 (Loopback Test) ---")
print("警告：请确保 main.py 已经关闭！")

try:
    # 打开串口
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    print("✅ 串口打开成功！")
    
    # 清空可能残留的旧数据
    ser.reset_input_buffer()
    
    # 模拟打包好的 8 字节数据帧
    test_frame = bytes([0xAA, 0x55, 0xFF, 0x01, 0x00, 0x10, 0x65, 0x0A])
    
    print(f"发送数据 -> : {' '.join([f'{b:02X}' for b in test_frame])}")
    ser.write(test_frame)
    
    # 稍微等个 0.1 秒，让电子在短接线上飞一会儿
    time.sleep(0.1)
    
    # 检查接收缓冲区有没有回来的数据
    if ser.in_waiting >= len(test_frame):
        recv_data = ser.read(ser.in_waiting)
        print(f"接收数据 <- : {' '.join([f'{b:02X}' for b in recv_data])}")
        
        if recv_data == test_frame:
            print("🎉 环回测试完美成功！硬件链路和串口驱动没有任何问题！")
        else:
            print("⚠️ 收到了数据，但内容对不上，可能有干扰。")
    else:
        print("❌ 发送了数据，但什么都没收到。请检查：")
        print("1. 杜邦线是否确切短接了引脚 8 和引脚 10？")
        print("2. 树莓派配置 (raspi-config) 中是否关闭了 Login Shell 并开启了 Hardware？")
        
except serial.SerialException as e:
    print(f"❌ 串口打开失败，被其他程序占用了？错误信息: {e}")
except KeyboardInterrupt:
    print("\n测试手动中止")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
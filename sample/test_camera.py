# tests/test_01_camera.py
import subprocess
import os
import cv2
import datetime

def test_camera_capture():
    print("🚀 [TEST 01] 正在唤醒底层硬件，并准备印制诊断水印...")
    output_file = "test_capture.jpg"
    watermark_file = "test_capture_watermarked.jpg"

    # 👑 你的环境专属黄金参数 (填入你窃听到的数据)
    # 我们把参数提出来作为变量，既传给硬件，也写进水印
    SHUTTER = 33239
    GAIN = 8.0

    # 清理旧测试数据
    for f in [output_file, watermark_file]:
        if os.path.exists(f):
            os.remove(f)

    # 1. 硬件感知层：严格注入物理参数进行拍摄
    cmd = [
        "rpicam-jpeg", 
        "-t", "1500",  # 给一点时间缓冲
        "--width", "640", 
        "--height", "480", 
        "--shutter", str(SHUTTER),
        "--gain", str(GAIN),
        "--vflip", "--hflip", 
        "-o", output_file
    ]
    
    try:
        print(f"📸 正在向底层请求强制曝光 (Shutter: {SHUTTER}us, Gain: {GAIN}x)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_file):
            print("✅ 底层图像获取成功！OpenCV 视觉层开始介入...")
            
            # 2. 视觉处理层：读取图像并打下硬核参数水印
            img = cv2.imread(output_file)
            
            if img is not None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 准备要印上去的技术档案
                tech_logs = [
                    f"IBVS Core - Auto Parameter Test",
                    f"Time: {timestamp}",
                    f"Shutter (exp): {SHUTTER} us",
                    f"Analog Gain: {GAIN} x",
                    f"AWB: Auto",
                    f"Status: HARDWARE_OK"
                ]
                
                # 在画面左上角逐行打印
                y_offset = 30
                for line in tech_logs:
                    # 第一层：加粗的黑色描边 (防止背景太白看不清字)
                    cv2.putText(img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # 第二层：明亮的黑客绿 (经典控制台配色)
                    cv2.putText(img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y_offset += 25
                
                # 覆写并保存最终的诊断图
                cv2.imwrite(watermark_file, img)
                
                # 阅后即焚原始底片，保持目录整洁
                os.remove(output_file)
                
                print(f"🎯 [大功告成] 自证水印已永久烙印！请在当前目录查看 '{watermark_file}'。")
            else:
                print("❌ [失败] OpenCV 无法读取生成的图像，可能是文件损坏。")
                
        else:
            print("❌ [失败] 底层命令执行失败。")
            print(f"--- 错误日志 ---\n{result.stderr}")
            
    except Exception as e:
        print(f"❌ [崩溃] Python 调用发生异常: {e}")

if __name__ == "__main__":
    test_camera_capture()
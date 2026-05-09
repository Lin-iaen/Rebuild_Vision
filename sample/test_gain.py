import subprocess
import threading
import time
import io
from typing import cast
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)
latest_jpeg = None
frame_lock = threading.Lock()

# ⚠️ 注意这里：既然我们要排查过曝，我先帮你把过于暴力的参数降下来！
# 如果画面暗，我们宁愿暗一点，也不能让白墙过曝。
RPICAM_CMD = [
    "rpicam-vid", "-t", "0",
    "--width", "640", "--height", "480", "--framerate", "30",
    "--codec", "mjpeg", "-o", "-", "--nopreview",
    "--shutter", "10000",  # 从极限的 33239 降到了 20000
    "--gain", "4.0",       # 从暴力的 8.0 降到了 4.0
    "--awb", "auto",
    "--vflip", "--hflip",
]

def kill_zombie_rpicam_processes():
    subprocess.run(["pkill", "-f", "rpicam-vid"], check=False)

def process_frame(frame_bgr: np.ndarray) -> bytes | None:
    """左右拼接：左侧彩色原图，右侧黑白掩膜"""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 提取红色
    lower1 = np.array([0, 40, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 40, 60], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # ROI 与 膨胀
    margin_x, margin_y = int(w * 0.12), int(h * 0.12)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
    mask = cv2.bitwise_and(mask, roi_mask)
    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL)

    # 寻找并绘制最终光斑
    final_mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找最大的面积画准星（方便你在左边彩图也看到它瞄准了哪）
    best_cnt = None
    max_area = 30
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # 在左侧原图上画准星，看看它到底抓了什么鬼东西
    if best_cnt is not None:
        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.drawMarker(frame_bgr, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # 将单通道的黑白 mask 转成 3 通道，为了能和 BGR 原图无缝拼接
    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    
    # 👑 核心逻辑：横向拼接（左边彩图，右边掩膜）
    combined = np.hstack((frame_bgr, mask_bgr))
    
    # 编码为更宽的 JPEG (1280 x 480)
    ok, encoded = cv2.imencode(".jpg", combined)
    if not ok: return None
    return encoded.tobytes()

def capture_loop():
    global latest_jpeg
    while True:
        process = None
        try:
            process = subprocess.Popen(RPICAM_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if process.stdout is None: continue
            stdout = cast(io.BufferedReader, process.stdout)
            buffer = bytearray()
            while True:
                chunk = stdout.read1(4096)
                if not chunk:
                    if process.poll() is not None: break
                    time.sleep(0.01)
                    continue
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 1024 * 1024: del buffer[:-1024 * 1024]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0: del buffer[:start]
                        break
                    jpg_bytes = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    npbuf = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
                    if frame is None: continue
                    
                    # 传入拼接器
                    combined_jpeg = process_frame(frame)
                    if combined_jpeg is None: continue
                    with frame_lock:
                        latest_jpeg = combined_jpeg
        except Exception:
            time.sleep(0.2)
        finally:
            if process is not None:
                try: process.terminate(); process.wait(timeout=0.5)
                except Exception:
                    try: process.kill()
                    except Exception: pass
        time.sleep(0.2)

def generate_stream():
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n\r\n"
    while True:
        with frame_lock: frame = latest_jpeg
        if frame is None:
            time.sleep(0.01)
            continue
        yield boundary + content_type + frame + b"\r\n"
        time.sleep(1.0 / 30.0)

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Test 04: 双摄照妖镜</title>
  <style>
    body { background: #111; color: white; text-align: center; font-family: sans-serif; }
    img { width: 90%; max-width: 1280px; border: 2px solid #444; border-radius: 10px; margin-top: 20px;}
  </style>
</head>
<body>
  <h2>🔍 Test 04: 双摄照妖镜 (左: 物理现实 | 右: 算法思维)</h2>
  <img src="/video_feed" alt="stream" />
</body>
</html>
"""

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    kill_zombie_rpicam_processes()
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
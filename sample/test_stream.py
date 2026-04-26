import subprocess
import threading
import time
import io
from typing import cast

import cv2
import numpy as np
from flask import Flask, Response


app = Flask(__name__)

# 仅保存“单通道二值掩膜”的 JPEG 编码结果，绝不对外输出 BGR。
latest_mask_jpeg = None
frame_lock = threading.Lock()

# 固定形态学内核：15x15 全 1，uint8。
# KERNEL = np.ones((15, 15), dtype=np.uint8)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) # 圆形内核更适合激光点的扩散

# rpicam-vid 启动参数严格锁定，按需求顺序不做改动。
RPICAM_CMD = [
	"rpicam-vid",
	"-t",
	"0",
	"--width",
	"640",
	"--height",
	"480",
	"--framerate",
	"30",
	"--codec",
	"mjpeg",
	"-o",
	"-",
	"--nopreview",
	"--shutter",
	"33239",
	"--gain",
	"8.0",
	"--awb",
	"auto",
	"--vflip",
	"--hflip",
]


def kill_zombie_rpicam_processes() -> None:
	"""启动前清理可能残留的 rpicam-vid 进程。"""
	subprocess.run(["pkill", "-f", "rpicam-vid"], check=False)


def make_laser_mask_bgr_to_jpeg_mask(frame_bgr: np.ndarray) -> bytes | None:
    """输入 BGR，输出“膨胀并过滤后的纯净单通道黑白掩膜 JPEG”"""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 1. 提取基础红色掩膜 (你调好的宽容参数)
    lower1 = np.array([0, 40, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 40, 70], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # 2. 刀法一：ROI 数字遮罩 (剔除边缘 10% 的光学垃圾区)
    # 计算需要切除的边界宽度 (比如 640 的 5% 是 32 像素)
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    # 创建一个全黑的掩膜
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    # 只把中间 90% 的区域涂白（有效跟踪区）
    roi_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
    # 把我们的激光掩膜和 ROI 掩膜做一个“与”操作，瞬间抹杀四周边缘的所有干扰！
    mask = cv2.bitwise_and(mask, roi_mask)

    # 3. 形态学发面 (使用圆润的印章)
    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL)

    # 4. 刀法二：轮廓面积过滤 (只保留真正的大圆饼)
    # 建立一张纯黑的最终画布
    final_mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 如果面积大于 100（说明是个实在的激光光斑，而不是紫边碎屑）
        if area > 100:
            # 才把它画到最终画布上
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # 5. 编码为 JPEG 发送
    ok, encoded = cv2.imencode(".jpg", final_mask)
    if not ok:
        return None
    return encoded.tobytes()


def capture_loop() -> None:
	"""
	持续读取 rpicam-vid 的 MJPEG stdout：
	- 必须使用 read1(4096) 防止管道阻塞；
	- 使用 JPEG SOI/EOI 做帧拼接并 imdecode。
	"""
	global latest_mask_jpeg

	while True:
		process = None
		try:
			process = subprocess.Popen(
				RPICAM_CMD,
				stdout=subprocess.PIPE,
				stderr=subprocess.DEVNULL,
			)

			if process.stdout is None:
				time.sleep(0.5)
				continue

			stdout = cast(io.BufferedReader, process.stdout)

			buffer = bytearray()

			while True:
				chunk = stdout.read1(4096)
				if not chunk:
					if process.poll() is not None:
						break
					time.sleep(0.01)
					continue

				buffer.extend(chunk)

				while True:
					start = buffer.find(b"\xff\xd8")
					if start < 0:
						if len(buffer) > 1024 * 1024:
							del buffer[:-1024 * 1024]
						break

					end = buffer.find(b"\xff\xd9", start + 2)
					if end < 0:
						if start > 0:
							del buffer[:start]
						break

					jpg_bytes = bytes(buffer[start : end + 2])
					del buffer[: end + 2]

					npbuf = np.frombuffer(jpg_bytes, dtype=np.uint8)
					frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
					if frame is None:
						continue

					mask_jpeg = make_laser_mask_bgr_to_jpeg_mask(frame)
					if mask_jpeg is None:
						continue

					with frame_lock:
						latest_mask_jpeg = mask_jpeg

		except Exception:
			# 子线程容错：异常后短暂等待并自动重启采集。
			time.sleep(0.2)
		finally:
			if process is not None:
				try:
					process.terminate()
					process.wait(timeout=0.5)
				except Exception:
					try:
						process.kill()
					except Exception:
						pass

		time.sleep(0.2)


def generate_mask_stream():
	"""输出 MJPEG multipart，仅包含单通道 mask jpeg。"""
	boundary = b"--frame\r\n"
	content_type = b"Content-Type: image/jpeg\r\n\r\n"

	while True:
		with frame_lock:
			frame = latest_mask_jpeg

		if frame is None:
			time.sleep(0.01)
			continue

		yield boundary + content_type + frame + b"\r\n"
		time.sleep(1.0 / 30.0)


@app.route("/")
def index() -> str:
	return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Test 02: 盲人视觉掩膜流测试</title>
  <style>
	:root {
	  color-scheme: dark;
	}
	body {
	  margin: 0;
	  min-height: 100vh;
	  display: flex;
	  align-items: center;
	  justify-content: center;
	  background: radial-gradient(circle at top, #1a1a1a 0%, #0d0d0d 60%, #050505 100%);
	  font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
	  color: #f2f2f2;
	}
	.panel {
	  width: min(96vw, 760px);
	  text-align: center;
	}
	h1 {
	  margin: 0 0 16px;
	  font-size: clamp(20px, 2.5vw, 30px);
	  font-weight: 700;
	  letter-spacing: 0.02em;
	}
	img {
	  width: 100%;
	  max-width: 640px;
	  height: auto;
	  border: 1px solid #2f2f2f;
	  border-radius: 10px;
	  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
	  background: #000;
	}
  </style>
</head>
<body>
  <main class="panel">
	<h1>Test 02: 盲人视觉掩膜流测试</h1>
	<img src="/video_feed" alt="mask stream" />
  </main>
</body>
</html>
"""


@app.route("/video_feed")
def video_feed() -> Response:
	return Response(
		generate_mask_stream(),
		mimetype="multipart/x-mixed-replace; boundary=frame",
	)


def main() -> None:
	kill_zombie_rpicam_processes()

	worker = threading.Thread(target=capture_loop, daemon=True)
	worker.start()

	app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
	main()

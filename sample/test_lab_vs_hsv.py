"""LAB vs HSV 激光识别算法对比测试。

功能：
- 同时应用HSV和LAB两种算法识别红色激光
- 左右拼接显示对比结果
- 通过Web推流实时观察哪种算法更稳定

使用方法：
- 运行脚本: python test_lab_vs_hsv.py
- 浏览器访问: http://<树莓派IP>:5000
"""

import subprocess
import threading
import time
import io
from typing import cast

import cv2
import numpy as np
from flask import Flask, Response


app = Flask(__name__)

# 保存对比结果的JPEG
latest_comparison_jpeg = None
frame_lock = threading.Lock()

# 物理参数（基于Rebuild_vision的测试结果）
RPICAM_CMD = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "-o", "-",
    "--nopreview",
    "--shutter", "33239",  # 极限进光量
    "--gain", "8.0",       # 确保黑胶带上的弱激光可见
    "--awb", "auto",
    "--vflip", "--hflip",
]


def kill_zombie_rpicam_processes() -> None:
    """启动前清理可能残留的 rpicam-vid 进程。"""
    subprocess.run(["pkill", "-f", "rpicam-vid"], check=False)


def process_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """HSV算法：来自Rebuild_vision/test_stream.py"""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 1. 提取基础红色掩膜
    lower1 = np.array([0, 40, 45], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 40, 45], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    # 2. ROI遮罩（切除边缘12%）
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.12)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
    mask_hsv = cv2.bitwise_and(mask_hsv, roi_mask)

    # 3. 形态学膨胀（9x9圆形内核）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.morphologyEx(mask_hsv, cv2.MORPH_DILATE, kernel)

    # 4. 轮廓面积过滤（>30）
    final_mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask


def process_lab(frame_bgr: np.ndarray) -> np.ndarray:
    """LAB算法：来自vision_project/tracker.py"""
    h, w = frame_bgr.shape[:2]

    # 1. LAB空间提取红激光：结合发白中心与红色光晕两类响应
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    center_mask = (l_ch > 180) & (a_ch > 130)  # 发白中心
    halo_mask = (l_ch > 60) & (a_ch > 135)     # 红色光晕
    mask = center_mask | halo_mask
    mask_u8 = (mask.astype(np.uint8)) * 255

    # 2. 先闭后开：修复光斑内部断裂，再去除小孤立噪声
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # 3. 带通滤波器：过滤面积<5或>300的轮廓
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5.0 <= area <= 300.0:
            valid_contours.append(cnt)

    if valid_contours:
        # 在合格的轮廓里，挑一个最大的
        largest = max(valid_contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest], -1, 255, -1)

    return final_mask


def process_frame(frame_bgr: np.ndarray) -> bytes | None:
    """对比两种算法，左右拼接显示"""
    h, w = frame_bgr.shape[:2]

    # 分别应用两种算法
    mask_hsv = process_hsv(frame_bgr)
    mask_lab = process_lab(frame_bgr)

    # 在原图上标记质心（如果检测到）
    hsv_contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lab_contours, _ = cv2.findContours(mask_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # HSV质心标记
    if hsv_contours:
        largest = max(hsv_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawMarker(frame_bgr, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # 转换为3通道以便拼接
    mask_hsv_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
    mask_lab_bgr = cv2.cvtColor(mask_lab, cv2.COLOR_GRAY2BGR)

    # 添加标签
    cv2.putText(mask_hsv_bgr, "HSV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(mask_lab_bgr, "LAB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 横向拼接：原图 | HSV掩膜 | LAB掩膜
    combined = np.hstack((frame_bgr, mask_hsv_bgr, mask_lab_bgr))

    # 编码为JPEG
    ok, encoded = cv2.imencode(".jpg", combined)
    if not ok:
        return None
    return encoded.tobytes()


def capture_loop() -> None:
    """持续读取 rpicam-vid 的 MJPEG stdout"""
    global latest_comparison_jpeg

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

                    comparison_jpeg = process_frame(frame)
                    if comparison_jpeg is None:
                        continue

                    with frame_lock:
                        latest_comparison_jpeg = comparison_jpeg

        except Exception:
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


def generate_stream():
    """输出 MJPEG multipart"""
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n\r\n"

    while True:
        with frame_lock:
            frame = latest_comparison_jpeg

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
    <title>LAB vs HSV 激光识别对比测试</title>
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
            width: min(96vw, 1400px);
            text-align: center;
        }
        h1 {
            margin: 0 0 16px;
            font-size: clamp(20px, 2.5vw, 30px);
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .description {
            margin: 0 0 20px;
            font-size: 14px;
            color: #aaa;
        }
        img {
            width: 100%;
            max-width: 1400px;
            height: auto;
            border: 1px solid #2f2f2f;
            border-radius: 10px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
            background: #000;
        }
        .legend {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 30px;
            font-size: 14px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .color-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <main class="panel">
        <h1>LAB vs HSV 激光识别对比测试</h1>
        <p class="description">左: 原图 | 中: HSV算法掩膜 | 右: LAB算法掩膜</p>
        <img src="/video_feed" alt="comparison stream" />
        <div class="legend">
            <div class="legend-item">
                <div class="color-box" style="background: #00ff00;"></div>
                <span>绿色准星 = HSV检测到的质心</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #ffffff;"></div>
                <span>白色区域 = 激光掩膜</span>
            </div>
        </div>
    </main>
</body>
</html>
"""


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def main() -> None:
    kill_zombie_rpicam_processes()

    worker = threading.Thread(target=capture_loop, daemon=True)
    worker.start()

    print("=" * 60)
    print("LAB vs HSV 激光识别对比测试")
    print("=" * 60)
    print("物理参数:")
    print(f"  快门: {RPICAM_CMD[11]} us")
    print(f"  增益: {RPICAM_CMD[13]}x")
    print("=" * 60)
    print("访问 http://0.0.0.0:5000 查看对比结果")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()

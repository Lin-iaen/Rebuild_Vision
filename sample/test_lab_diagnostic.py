"""LAB值诊断工具。

用于查看激光点和背景的实际LAB值，帮助调参。
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

latest_frame = None
frame_lock = threading.Lock()

# 物理参数
RPICAM_CMD = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "-o", "-",
    "--nopreview",
    "--shutter", "33239",
    "--gain", "8.0",
    "--awb", "auto",
    "--vflip", "--hflip",
]


def kill_zombie_rpicam_processes() -> None:
    subprocess.run(["pkill", "-f", "rpicam-vid"], check=False)


def process_frame(frame_bgr: np.ndarray) -> bytes | None:
    """分析LAB值分布"""
    h, w = frame_bgr.shape[:2]
    
    # 转换为LAB
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    # 计算统计信息
    l_mean = np.mean(l_ch)
    l_max = np.max(l_ch)
    a_mean = np.mean(a_ch)
    a_max = np.max(a_ch)
    
    # 找到L值最高的点（可能是激光点）
    l_max_pos = np.unravel_index(np.argmax(l_ch), l_ch.shape)
    l_max_y, l_max_x = l_max_pos
    
    # 获取该点周围的LAB值
    roi_size = 10
    x1 = max(0, l_max_x - roi_size)
    y1 = max(0, l_max_y - roi_size)
    x2 = min(w, l_max_x + roi_size)
    y2 = min(h, l_max_y + roi_size)
    
    roi_l = l_ch[y1:y2, x1:x2]
    roi_a = a_ch[y1:y2, x1:x2]
    roi_l_mean = np.mean(roi_l)
    roi_a_mean = np.mean(roi_a)
    
    # 应用当前阈值
    center_mask = (l_ch > 180) & (a_ch > 130)
    halo_mask = (l_ch > 60) & (a_ch > 135)
    mask = center_mask | halo_mask
    mask_u8 = (mask.astype(np.uint8)) * 255
    
    # 统计mask中的像素数
    mask_pixels = np.sum(mask_u8 > 0)
    
    # 在原图上标记L值最高的点
    annotated = frame_bgr.copy()
    cv2.circle(annotated, (l_max_x, l_max_y), 10, (0, 0, 255), 2)
    cv2.putText(annotated, f"L={l_ch[l_max_y, l_max_x]}", (l_max_x+15, l_max_y-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(annotated, f"A={a_ch[l_max_y, l_max_x]}", (l_max_x+15, l_max_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 添加统计信息
    info_lines = [
        f"L_mean={l_mean:.0f}, L_max={l_max}",
        f"A_mean={a_mean:.0f}, A_max={a_max}",
        f"ROI_L={roi_l_mean:.0f}, ROI_A={roi_a_mean:.0f}",
        f"Mask pixels={mask_pixels}",
        f"Thresholds: L>180 & A>130, L>60 & A>135",
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(annotated, line, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    
    # 拼接原图和mask
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "MASK (L>60 & A>135)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    combined = np.hstack((annotated, mask_bgr))
    
    # 编码为JPEG
    ok, encoded = cv2.imencode(".jpg", combined)
    if not ok:
        return None
    return encoded.tobytes()


def capture_loop() -> None:
    global latest_frame

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

                    result_jpeg = process_frame(frame)
                    if result_jpeg is None:
                        continue

                    with frame_lock:
                        latest_frame = result_jpeg

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
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n\r\n"

    while True:
        with frame_lock:
            frame = latest_frame

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
    <title>LAB值诊断工具</title>
    <style>
        :root { color-scheme: dark; }
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #111;
            font-family: monospace;
            color: #f2f2f2;
        }
        .panel { width: 95vw; text-align: center; }
        h1 { margin: 0 0 16px; font-size: 24px; color: #00ff00; }
        img { width: 100%; border: 1px solid #333; border-radius: 8px; }
        .instructions {
            margin-top: 20px;
            text-align: left;
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.6;
        }
        .instructions h3 { color: #00ff00; margin-top: 0; }
        code { background: #333; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <main class="panel">
        <h1>LAB值诊断工具</h1>
        <img src="/video_feed" alt="diagnostic stream" />
        <div class="instructions">
            <h3>使用说明</h3>
            <p>左侧：原图 + L值最高点标记（红圈）</p>
            <p>右侧：当前阈值的mask结果</p>
            <p><strong>请在黑色背景下测试激光点：</strong></p>
            <ol>
                <li>将激光点打在黑色背景上</li>
                <li>观察左上角显示的 <code>L_max</code> 和 <code>A_max</code> 值</li>
                <li>观察 <code>ROI_L</code> 和 <code>ROI_A</code>（激光点周围的平均值）</li>
                <li>如果 <code>Mask pixels</code> 为0，说明当前阈值无法检测到激光</li>
            </ol>
            <p><strong>请告诉我这些值是多少，我来调整阈值！</strong></p>
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
    print("LAB值诊断工具")
    print("=" * 60)
    print("请在黑色背景下测试激光点")
    print("观察L_max和A_max值，告诉我具体数值")
    print("=" * 60)
    print("访问 http://0.0.0.0:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()

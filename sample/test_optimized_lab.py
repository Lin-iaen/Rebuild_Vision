"""优化后的LAB算法测试。

测试tracker.py中的优化算法，特别关注白纸场景的识别效果。
"""

import subprocess
import threading
import time
import io
from typing import cast

import cv2
import numpy as np
from flask import Flask, Response

# 导入优化后的tracker
import sys
sys.path.insert(0, '/home/lin/workspace/Rebuild_vision')
from tracker import process_laser_detection


app = Flask(__name__)

latest_frame_jpeg = None
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
    """使用优化后的tracker处理帧"""
    h, w = frame_bgr.shape[:2]

    # 使用优化后的算法
    laser_pos, annotated = process_laser_detection(frame_bgr, debug=False)

    # 添加状态信息
    status = "DETECTED" if laser_pos else "NO LASER"
    color = (0, 255, 0) if laser_pos else (0, 0, 255)
    cv2.putText(
        annotated,
        f"Status: {status}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    # 编码为JPEG
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        return None
    return encoded.tobytes()


def capture_loop() -> None:
    global latest_frame_jpeg

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
                        latest_frame_jpeg = result_jpeg

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
            frame = latest_frame_jpeg

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
    <title>优化后的LAB算法测试</title>
    <style>
        :root { color-scheme: dark; }
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
        .panel { width: min(96vw, 800px); text-align: center; }
        h1 { margin: 0 0 16px; font-size: clamp(20px, 2.5vw, 30px); font-weight: 700; }
        .info { margin: 0 0 20px; font-size: 14px; color: #aaa; }
        img {
            width: 100%;
            max-width: 800px;
            height: auto;
            border: 1px solid #2f2f2f;
            border-radius: 10px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
        }
        .test-guide {
            margin-top: 20px;
            text-align: left;
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
        }
        .test-guide h3 { margin-top: 0; color: #00ff00; }
        .test-guide ul { margin: 5px 0; padding-left: 20px; }
    </style>
</head>
<body>
    <main class="panel">
        <h1>优化后的LAB算法测试</h1>
        <p class="info">自适应阈值 + 面积过滤 + 质心验证</p>
        <img src="/video_feed" alt="stream" />
        <div class="test-guide">
            <h3>测试指南</h3>
            <ul>
                <li>在白纸上移动激光点，观察识别效果</li>
                <li>在白墙上移动激光点，对比识别效果</li>
                <li>在黑色背景下测试，确认稳定性</li>
                <li>观察左上角的L_mean和thresh值变化</li>
            </ul>
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
    print("优化后的LAB算法测试")
    print("=" * 60)
    print("改进点:")
    print("  - 自适应阈值：根据环境亮度动态调整")
    print("  - 面积过滤：更宽松的范围适应白纸场景")
    print("  - 质心验证：检查检测点的亮度是否足够")
    print("=" * 60)
    print("访问 http://0.0.0.0:5000 查看效果")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()

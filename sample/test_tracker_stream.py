"""激光追踪联合验证。

Camera 采集帧 → tracker.process_laser_detection() 标注 → MjpegStream 推流。

用法：
    python3 sample/test_tracker_stream.py
    浏览器访问 http://<树莓派IP>:5000
    用激光笔照射画面，观察十字准星是否跟随。
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from camera import Camera
from web_stream import MjpegStream
from tracker import process_laser_detection


def main() -> None:
    cam = Camera()
    cam.start()

    # 等待摄像头就绪
    print("等待摄像头就绪...", end="", flush=True)
    for _ in range(50):
        if cam.read() is not None:
            print(" OK")
            break
        time.sleep(0.1)
    else:
        print(" 超时！未收到帧")
        cam.release()
        return

    def provide() -> bytes | None:
        frame = cam.read()
        if frame is None:
            return None
        _, annotated = process_laser_detection(frame)
        ok, buf = cv2.imencode(".jpg", annotated)
        return buf.tobytes() if ok else None

    stream = MjpegStream(frame_provider=provide, title="激光追踪测试")
    stream.start()

    print("=" * 50)
    print("激光追踪测试")
    print("=" * 50)
    print("浏览器访问 http://0.0.0.0:5000")
    print("用激光笔照射画面，观察十字准星是否跟随")
    print("Ctrl+C 退出")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        cam.release()


if __name__ == "__main__":
    main()

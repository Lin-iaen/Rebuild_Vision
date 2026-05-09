# test_camera_stream.py
import time, cv2
from camera import Camera
from web_stream import MjpegStream
cam = Camera()
cam.start()
def provide():
    frame = cam.read()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None
stream = MjpegStream(frame_provider=provide, title="摄像头直推测试")
stream.start()
while True:
    time.sleep(1)
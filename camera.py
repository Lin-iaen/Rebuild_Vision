"""摄像头采集模块。

封装 rpicam-vid 子进程，提供线程安全的帧获取接口。
硬件参数来自 README.md 实测基准，严禁随意修改。
"""

import io
import logging
import subprocess
import threading
import time
from typing import Optional, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# rpicam-vid 默认启动参数（实测基准）
_DEFAULT_CMD = [
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "-o", "-",
    "--nopreview",
    "--shutter", "33239",
    "--gain", "4.0",
    "--awb", "auto",
    "--vflip",
    "--hflip",
]


class Camera:
    """rpicam-vid 摄像头采集封装。

    用法::

        cam = Camera()
        cam.start()
        frame = cam.read()  # BGR ndarray 或 None
        cam.release()

        # 或用上下文管理器
        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        shutter: int = 30000,
        gain: float = 4.0,
        awb: str = "auto",
    ) -> None:
        self._width = width
        self._height = height
        self._framerate = framerate
        self._shutter = shutter
        self._gain = gain
        self._awb = awb

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """启动采集守护线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("摄像头采集已启动")

    def read(self) -> Optional[np.ndarray]:
        """返回最新 BGR 帧（线程安全）。未就绪时返回 None。"""
        with self._lock:
            return self._latest_frame

    def release(self) -> None:
        """停止采集并清理子进程。"""
        self._running = False
        self._kill_process()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("摄像头采集已停止")

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        """构造 rpicam-vid 命令行。"""
        return [
            "rpicam-vid",
            "-t", "0",
            "--width", str(self._width),
            "--height", str(self._height),
            "--framerate", str(self._framerate),
            "--codec", "mjpeg",
            "-o", "-",
            "--nopreview",
            "--shutter", str(self._shutter),
            "--gain", str(self._gain),
            "--awb", self._awb,
            "--vflip",
            "--hflip",
        ]

    def _kill_zombie(self) -> None:
        """清理可能残留的 rpicam-vid 进程。"""
        subprocess.run(["pkill", "-f", "rpicam-vid"], check=False)

    def _kill_process(self) -> None:
        """安全终止当前子进程。"""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=0.5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def _capture_loop(self) -> None:
        """守护线程主循环：启动子进程 → 读取 JPEG 帧 → 自动重启。"""
        while self._running:
            self._kill_zombie()
            self._process = None
            try:
                cmd = self._build_cmd()
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                if self._process.stdout is None:
                    time.sleep(0.5)
                    continue

                stdout = cast(io.BufferedReader, self._process.stdout)
                buffer = bytearray()

                while self._running:
                    chunk = stdout.read1(4096)
                    if not chunk:
                        if self._process.poll() is not None:
                            break
                        time.sleep(0.01)
                        continue

                    buffer.extend(chunk)

                    # JPEG SOI/EOI 帧分割
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            # 防止缓冲区无限增长
                            if len(buffer) > 1024 * 1024:
                                del buffer[:-1024 * 1024]
                            break

                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            break

                        jpg_bytes = bytes(buffer[start: end + 2])
                        del buffer[: end + 2]

                        frame = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if frame is None:
                            continue

                        with self._lock:
                            self._latest_frame = frame

            except Exception:
                logger.debug("采集线程异常，自动重启", exc_info=True)
                time.sleep(0.2)
            finally:
                self._kill_process()

            time.sleep(0.2)

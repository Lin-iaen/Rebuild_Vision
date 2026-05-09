"""通用 MJPEG 推流模块。

零耦合设计：不依赖 camera / tracker，由调用方决定推什么画面。

用法::

    from web_stream import MjpegStream

    # frame_provider 返回 JPEG bytes 或 None
    stream = MjpegStream(frame_provider=my_provider, title="IBVS 调试流")
    stream.start()
    # 浏览器访问 http://<ip>:5000
"""

import logging
import threading
import time
from typing import Callable, Optional

from flask import Flask, Response

logger = logging.getLogger(__name__)

# HTML 模板：深色主题调试页面
_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(circle at top, #1a1a1a 0%, #0d0d0d 60%, #050505 100%);
      font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
      color: #f2f2f2;
    }}
    .panel {{
      width: min(96vw, 800px);
      text-align: center;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: clamp(20px, 2.5vw, 30px);
      font-weight: 700;
    }}
    img {{
      width: 100%;
      max-width: 640px;
      height: auto;
      border: 1px solid #2f2f2f;
      border-radius: 10px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
      background: #000;
    }}
  </style>
</head>
<body>
  <main class="panel">
    <h1>{title}</h1>
    <img src="/video_feed" alt="stream" />
  </main>
</body>
</html>
"""


class MjpegStream:
    """通用 MJPEG 推流服务。

    参数:
        frame_provider: 回调函数，返回 JPEG 编码的 bytes，或 None 表示暂无帧。
        title: 页面标题。
    """

    def __init__(
        self,
        frame_provider: Callable[[], Optional[bytes]],
        title: str = "IBVS 调试流",
    ) -> None:
        self._frame_provider = frame_provider
        self._title = title
        self._app = Flask(__name__)
        self._app.add_url_rule("/", "index", self._route_index)
        self._app.add_url_rule("/video_feed", "video_feed", self._route_video_feed)

    def start(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """在守护线程中启动 Flask 服务。"""
        thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": host, "port": port, "threaded": True},
            daemon=True,
        )
        thread.start()
        logger.info(f"MJPEG 推流已启动: http://{host}:{port}")

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def _route_index(self) -> str:
        return _PAGE_HTML.format(title=self._title)

    def _route_video_feed(self) -> Response:
        return Response(
            self._generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _generate(self):
        """MJPEG 帧生成器。"""
        boundary = b"--frame\r\n"
        content_type = b"Content-Type: image/jpeg\r\n\r\n"

        while True:
            jpeg = self._frame_provider()
            if jpeg is None:
                time.sleep(0.01)
                continue

            yield boundary + content_type + jpeg + b"\r\n"
            time.sleep(1.0 / 30.0)

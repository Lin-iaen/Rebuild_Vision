"""黑色电工胶带矩形框的识别与中心线标定验证。

检测 1.8cm 宽黑色电工胶带围成的矩形边框，
标定出中心线角点，人工确认后打印坐标。

用法：
    python3 sample/test_rectangle.py
    浏览器访问 http://<树莓派IP>:5000 查看实时标注
    终端按 Enter 确认标定，输入 q 退出
"""

import math
import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from camera import Camera
from web_stream import MjpegStream
from tracker import _order_quad_points


# 检测参数（可调整）
LAB_THRESH = 80          # L通道阈值：黑色胶带 < 80，白色背景 > 200
MIN_AREA = 5000          # 最小轮廓面积（像素）
APPROX_EPS = 0.02        # 多边形近似精度（周长倍数）
PAIR_DIST = 60.0         # 内外配对中心最大距离（像素）
PAIR_RATIO_MIN = 1.2     # 内外面积比下限
PAIR_RATIO_MAX = 4.0     # 内外面积比上限

# 全局状态
_latest_annotated: np.ndarray | None = None
_latest_corners: list[tuple[float, float]] | None = None
_lock = threading.Lock()


def detect_tape_frame(frame: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """检测黑色胶带矩形，返回标注帧和中心线角点。

    返回：
    - annotated: 带标注的图像
    - corners: 4个角点坐标 [(x,y), ...]，顺序为 [左上, 右上, 右下, 左下]
    """
    global _latest_annotated, _latest_corners

    annotated = frame.copy()
    h, w = frame.shape[:2]

    # 1) LAB 阈值分割：提取黑色胶带区域
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    _, tape_mask = cv2.threshold(l_ch, LAB_THRESH, 255, cv2.THRESH_BINARY_INV)

    # 2) 形态学闭操作：填充胶带内部断裂
    kernel = np.ones((5, 5), np.uint8)
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, kernel)

    # 3) 轮廓提取
    contours, _ = cv2.findContours(tape_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 4) 筛选四边形
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS * peri, True)
        if len(approx) != 4:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        pts = approx.reshape(4, 2).astype(np.float32)
        quads.append({"pts": pts, "area": float(area), "cx": cx, "cy": cy})

    quads.sort(key=lambda q: q["area"], reverse=True)

    # 5) 寻找内外配对，计算中心线
    best_corners = None
    outer_pts = None
    inner_pts = None

    if len(quads) >= 2:
        search = quads[:5]
        for i in range(len(search) - 1):
            for j in range(i + 1, len(search)):
                q1, q2 = search[i], search[j]
                dist = math.hypot(q1["cx"] - q2["cx"], q1["cy"] - q2["cy"])
                if dist > PAIR_DIST:
                    continue
                ratio = q1["area"] / q2["area"] if q2["area"] > 0 else 999
                if PAIR_RATIO_MIN < ratio < PAIR_RATIO_MAX:
                    outer_pts = _order_quad_points(q1["pts"])
                    inner_pts = _order_quad_points(q2["pts"])
                    best_corners = (outer_pts + inner_pts) / 2.0
                    break
            if best_corners is not None:
                break

    # 降级：仅用最大的一个四边形
    if best_corners is None and len(quads) >= 1:
        best_corners = _order_quad_points(quads[0]["pts"])

    # 6) 标注
    if best_corners is not None:
        # 外边缘（蓝）
        if outer_pts is not None:
            cv2.drawContours(annotated, [outer_pts.astype(np.int32)], -1, (255, 0, 0), 2)
        # 内边缘（蓝）
        if inner_pts is not None:
            cv2.drawContours(annotated, [inner_pts.astype(np.int32)], -1, (255, 0, 0), 2)
        # 中心线（绿粗线）
        cv2.drawContours(annotated, [best_corners.astype(np.int32)], -1, (0, 255, 0), 3)
        # 角点
        labels = ["左上", "右上", "右下", "左下"]
        for i, (x, y) in enumerate(best_corners):
            xi, yi = int(x), int(y)
            cv2.circle(annotated, (xi, yi), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"P{i}({labels[i]})({xi},{yi})",
                (xi + 10, yi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        cv2.putText(
            annotated,
            f"RECT OK | quads={len(quads)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            annotated,
            f"NO RECT | quads={len(quads)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    corners = [(float(x), float(y)) for x, y in best_corners] if best_corners is not None else []

    with _lock:
        _latest_annotated = annotated
        _latest_corners = corners if corners else None

    return annotated, corners


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
        annotated, _ = detect_tape_frame(frame)
        ok, buf = cv2.imencode(".jpg", annotated)
        return buf.tobytes() if ok else None

    stream = MjpegStream(frame_provider=provide, title="黑色矩形标定")
    stream.start()

    print("=" * 55)
    print("黑色矩形标定验证")
    print("=" * 55)
    print("浏览器查看: http://0.0.0.0:5000")
    print("将黑色胶带矩形放入画面，等待绿色中心线出现")
    print()
    print("按 Enter 确认标定 | 输入 q 退出")
    print("=" * 55)

    try:
        while True:
            cmd = input()
            if cmd.strip().lower() == "q":
                break

            with _lock:
                corners = _latest_corners

            if corners is None:
                print("未检测到有效矩形，请调整位置后重试")
                continue

            print("角点坐标 [左上, 右上, 右下, 左下]:")
            labels = ["左上", "右上", "右下", "左下"]
            for i, (x, y) in enumerate(corners):
                print(f"  P{i}({labels[i]}): ({x:.1f}, {y:.1f})")
            print()

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("退出")
        cam.release()


if __name__ == "__main__":
    main()

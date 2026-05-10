"""矩形检测与子目标生成模块。

检测黑色电工胶带矩形框，管理角点数据，生成循迹子目标点。
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from tracker import _order_quad_points

logger = logging.getLogger(__name__)

# 检测参数
LAB_THRESH = 80          # L通道阈值：黑色胶带 < 80，白色背景 > 200
MIN_AREA = 5000          # 最小轮廓面积（像素）
APPROX_EPS = 0.02        # 多边形近似精度（周长倍数）
PAIR_DIST = 60.0         # 内外配对中心最大距离（像素）
PAIR_RATIO_MIN = 1.2     # 内外面积比下限
PAIR_RATIO_MAX = 4.0     # 内外面积比上限

# 子目标点密度（可修改）
EDGE_POINTS_LONG = 10    # 长边子目标数
EDGE_POINTS_SHORT = 5    # 短边子目标数


class RectangleManager:
    """矩形检测与管理。

    角点顺序: [左上(P0), 右上(P1), 右下(P2), 左下(P3)]
    遍历顺序: 顺时针 P0→P1→P2→P3→P0
    """

    def __init__(self) -> None:
        self._corners: list[tuple[float, float]] | None = None

    @property
    def corners(self) -> list[tuple[float, float]] | None:
        return self._corners

    def detect(self, frame: np.ndarray) -> bool:
        """检测黑色胶带矩形，成功则记录角点。

        返回: 是否检测到有效矩形
        """
        h, w = frame.shape[:2]

        # LAB 阈值分割：提取黑色胶带区域
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]
        _, tape_mask = cv2.threshold(l_ch, LAB_THRESH, 255, cv2.THRESH_BINARY_INV)

        # 形态学闭操作：填充胶带内部断裂
        kernel = np.ones((5, 5), np.uint8)
        tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, kernel)

        # 轮廓提取
        contours, _ = cv2.findContours(tape_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选四边形
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

        # 寻找内外配对，计算中心线
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

        if best_corners is None:
            self._corners = None
            return False

        self._corners = [(float(x), float(y)) for x, y in best_corners]
        logger.info(f"矩形检测成功: {self._corners}")
        return True

    def get_center(self) -> tuple[float, float] | None:
        """返回矩形中心坐标。"""
        if self._corners is None:
            return None
        cx = sum(p[0] for p in self._corners) / 4
        cy = sum(p[1] for p in self._corners) / 4
        return (cx, cy)

    def get_ordered_corners(self) -> list[tuple[float, float]] | None:
        """返回排序后的角点 [左上, 右上, 右下, 左下]。"""
        return self._corners

    def get_targets(self) -> list[tuple[float, float]]:
        """生成顺时针循迹子目标点列表。

        角点顺序: P0(左上)→P1(右上)→P2(右下)→P3(左下)→P0
        长边和短边分别使用不同的点数。

        返回: 子目标坐标列表，不包含起点（P0），包含终点（P0）
        """
        if self._corners is None:
            return []

        corners = self._corners
        targets = []

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]

            # 判断长边还是短边
            edge_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            n_points = self._get_edge_points(edge_len)

            for j in range(1, n_points + 1):
                t = j / n_points
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                targets.append((x, y))

        return targets

    def _get_edge_points(self, edge_len: float) -> int:
        """根据边长返回子目标点数。"""
        # 通过比较当前边长与所有边长的平均值来判断长/短边
        if self._corners is None:
            return EDGE_POINTS_SHORT

        corners = self._corners
        lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            lengths.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

        avg_len = sum(lengths) / len(lengths)
        return EDGE_POINTS_LONG if edge_len >= avg_len else EDGE_POINTS_SHORT

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """在画面上绘制矩形边框和角点标注。"""
        annotated = frame.copy()

        if self._corners is None:
            cv2.putText(annotated, "NO RECT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated

        corners = self._corners
        pts = np.array(corners, dtype=np.int32)

        # 矩形边框（蓝色）
        cv2.polylines(annotated, [pts], True, (255, 128, 0), 2)

        # 角点标注
        labels = ["P0(TopLeft)", "P1(TopRight)", "P2(BottomRight)", "P3(BottomLeft)"]
        for i, (x, y) in enumerate(corners):
            xi, yi = int(x), int(y)
            cv2.circle(annotated, (xi, yi), 6, (0, 0, 255), -1)
            cv2.putText(annotated, labels[i], (xi + 8, yi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # 中心点
        center = self.get_center()
        if center:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

        cv2.putText(annotated, "RECT OK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

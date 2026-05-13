"""激光识别模块 - 优化版。

基于vision_project的LAB算法，针对白纸场景进行优化：
1. 使用LAB颜色空间，更稳定
2. 增加自适应阈值，适应不同背景
3. 增加面积过滤和质心验证
"""

import cv2
import math
import numpy as np
from typing import Optional, Tuple, List


def process_laser_detection(
    frame: np.ndarray,
    debug: bool = False
) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """激光点检测主函数。

    输入：
    - frame: BGR图像帧
    - debug: 是否返回调试用的掩膜图像

    返回：
    - laser_pos: 激光点质心坐标 (x, y)，未检测到时为 None
    - annotated: 带标注的图像（或调试掩膜）
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # ====== LAB空间提取红激光 ======
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # 优化策略：使用相对阈值而非绝对阈值
    # 计算图像的平均亮度，动态调整阈值
    l_mean = np.mean(l_ch)
    
    # 核心检测条件（两种响应模式）
    # center_mask: 激光中心的高亮区域（L值很高，A值偏红）
    # halo_mask: 激光周围的光晕（L值中等，A值偏红）
    
    # 使用vision_project的原始固定阈值（已验证有效）
    # 根据实际测试数据优化：
    # 激光点中心: L=250-254, A=127-132
    # 边缘色差伪影: L=228, A=143-157
    # 添加 A 值上界，排除色差伪影
    # l_center_thresh = 180
    l_center_thresh = 230
    l_halo_thresh = 60
    a_thresh_center = 125      # A 值下界，检测红色激光
    a_thresh_center_max = 150  # A 值上界，排除色差伪影
    a_thresh_halo = 130        # 光晕 A 值下界
    a_thresh_halo_max = 145    # 光晕 A 值上界
    
    center_mask = (l_ch > l_center_thresh) & (a_ch > a_thresh_center) & (a_ch < a_thresh_center_max)
    halo_mask = (l_ch > l_halo_thresh) & (a_ch > a_thresh_halo) & (a_ch < a_thresh_halo_max)
    mask = center_mask | halo_mask
    mask_u8 = (mask.astype(np.uint8)) * 255

    # ====== ROI 排除边缘区域 ======
    # 排除画面边缘 10%，避免边缘干扰（如左下角误检）
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
    mask_u8 = cv2.bitwise_and(mask_u8, roi_mask)

    # ====== 形态学操作 ======
    # 先闭后开：修复光斑内部断裂，再去除小孤立噪声
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # ====== 带通滤波器 ======
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug:
            return None, mask_u8
        return None, annotated

    # 过滤面积：激光点通常在5-500像素之间
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 根据环境亮度调整面积阈值
        # 白纸环境：激光点可能更小，降低下限
        # 黑暗环境：激光点可能更大，提高上限
        min_area = 3 if l_mean > 150 else 5
        max_area = 800 if l_mean > 150 else 500
        
        if min_area <= area <= max_area:
            valid_contours.append(cnt)

    if not valid_contours:
        if debug:
            return None, mask_u8
        return None, annotated

    # 在合格的轮廓里，挑一个最大的（应对激光晕开的情况）
    largest = max(valid_contours, key=cv2.contourArea)
    
    # 计算质心
    m = cv2.moments(largest)
    if m["m00"] == 0:
        if debug:
            return None, mask_u8
        return None, annotated

    u = float(m["m10"] / m["m00"])
    v = float(m["m01"] / m["m00"])

    # ====== 质心验证 ======
    # 检查质心是否在有效区域内（排除边缘噪声）
    margin = 30  # 边缘区域宽度
    if u < margin or u > w - margin or v < margin or v > h - margin:
        if debug:
            return None, mask_u8
        return None, annotated
    
    # 额外验证：检查轮廓是否太靠近边缘（排除边角噪声）
    x, y, bw, bh = cv2.boundingRect(largest)
    if x < margin or x + bw > w - margin or y < margin or y + bh > h - margin:
        if debug:
            return None, mask_u8
        return None, annotated

    # ====== 可视化 ======
    if debug:
        # 调试模式：返回带标记的掩膜
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(final_mask, [largest], -1, 255, -1)
        cv2.circle(final_mask, (int(u), int(v)), 5, 128, -1)
        return (u, v), final_mask

    # 正常模式：在原图上绘制标注
    ui, vi = int(round(u)), int(round(v))
    
    # 十字准星
    cross_len = 12
    cv2.line(annotated, (ui - cross_len, vi), (ui + cross_len, vi), (0, 255, 0), 2)
    cv2.line(annotated, (ui, vi - cross_len), (ui, vi + cross_len), (0, 255, 0), 2)
    cv2.circle(annotated, (ui, vi), 6, (0, 0, 255), 1)

    # 坐标信息
    cv2.putText(
        annotated,
        f"({ui},{vi})",
        (ui + 10, vi - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    # 环境信息（调试用）
    cv2.putText(
        annotated,
        f"L_mean={l_mean:.0f} thresh={l_center_thresh:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    return (u, v), annotated


def process_init_mode(frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """INIT模式：检测四边形标定点。

    输入：
    - frame: BGR图像帧

    返回：
    - annotated: 带标注的图像
    - corners: 4个角点坐标，未检测到时为空列表
    """
    annotated = frame.copy()

    # 1) 预处理：灰度 + 高斯滤波
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # 2) 边缘检测：Canny
    edges = cv2.Canny(blur, 40, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3) 轮廓提取
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads = []

    # 4) 候选四边形筛选
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 1000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
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

    corners = []
    best_ordered = None

    # 5) 寻找稳定的内外双层配对
    if len(quads) >= 2:
        search_quads = quads[:5]
        matched_pair = False

        for i in range(len(search_quads) - 1):
            q_outer = search_quads[i]
            for j in range(i + 1, len(search_quads)):
                q_inner = search_quads[j]

                cx1, cy1 = float(q_outer["cx"]), float(q_outer["cy"])
                cx2, cy2 = float(q_inner["cx"]), float(q_inner["cy"])
                dist = math.hypot(cx1 - cx2, cy1 - cy2)

                area_outer = float(q_outer["area"])
                area_inner = float(q_inner["area"])
                if area_inner <= 0:
                    continue
                area_ratio = area_outer / area_inner

                if dist < 60.0 and 1.2 < area_ratio < 4.0:
                    p_outer = _order_quad_points(np.asarray(q_outer["pts"], dtype=np.float32))
                    p_inner = _order_quad_points(np.asarray(q_inner["pts"], dtype=np.float32))

                    best_ordered = (p_outer + p_inner) / 2.0

                    cv2.drawContours(annotated, [p_outer.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [p_inner.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [best_ordered.astype(np.int32)], -1, (0, 255, 0), 5)

                    matched_pair = True
                    break

            if matched_pair:
                break

    # 6) 未形成双层时，降级使用面积最大的一层
    if best_ordered is None and len(quads) >= 1:
        best_ordered = _order_quad_points(np.asarray(quads[0]["pts"], dtype=np.float32))
        cv2.drawContours(annotated, [best_ordered.astype(np.int32)], -1, (0, 255, 0), 3)

    if best_ordered is not None:
        corners = [(float(x), float(y)) for x, y in best_ordered]

        for i, (x, y) in enumerate(corners):
            xi, yi = int(x), int(y)
            cv2.circle(annotated, (xi, yi), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"P{i}({xi},{yi})",
                (xi + 10, yi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    return annotated, corners


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """将四边形点排序为 [左上, 右上, 右下, 左下]。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

"""激光点检测模块。

基于 LAB 色彩空间，通过 L/A 通道阈值提取红色激光点。
参数外置，支持动态调参。

用法::

    from tracker import LaserParams, process_laser_detection

    # 使用默认参数
    pos, annotated = process_laser_detection(frame)

    # 自定义参数
    params = LaserParams(l_center=200, a_center_min=120)
    pos, annotated = process_laser_detection(frame, params=params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaserParams:
    """激光检测参数。

    所有参数可直接通过 dataclass 构造修改，对应含义见注释。
    """

    # ---- L 通道 (亮度 0-255) ----
    l_center: int = 200       # 激光中心 L 下界
    l_halo: int = 200          # 激光光晕 L 下界

    # ---- A 通道 (红绿轴) ----
    a_center_min: int = 121   # 中心 A 下界
    a_center_max: int = 246   # 中心 A 上界（排除色差伪影）
    a_halo_min: int = 105     # 光晕 A 下界
    a_halo_max: int = 145     # 光晕 A 上界

    # ---- ROI 掩膜 ----
    roi_margin: float = 0.1   # 排除画面边缘比例

    # ---- 面积过滤 ----
    area_min_bright: int = 3     # 亮环境面积下限
    area_max_bright: int = 800   # 亮环境面积上限
    area_min_dark: int = 5       # 暗环境面积下限
    area_max_dark: int = 500     # 暗环境面积上限
    brightness_thresh: float = 150.0  # 亮/暗环境 L 均值分界

    # ---- 形态学 ----
    morph_kernel: int = 5    # 闭+开运算核尺寸

    # ---- 质心验证 ----
    centroid_margin: int = 30  # 质心/轮廓距画面边缘的距离下限


def process_laser_detection(
    frame: np.ndarray,
    params: LaserParams | None = None,
    debug: bool = False,
) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """激光点检测。

    参数:
        frame: BGR 图像帧
        params: 检测参数, None 则使用默认值
        debug: True 时返回二值掩膜而非标注图

    返回:
        laser_pos: 质心坐标 (x, y), 未检测到时为 None
        annotated: 标注图像（或调试掩膜）
    """
    p = params or LaserParams()
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # ====== LAB 空间提取 ======
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, _b_ch = cv2.split(lab)

    l_mean = float(np.mean(l_ch))

    # 中心掩膜
    center_mask = (
        (l_ch > p.l_center)
        & (a_ch > p.a_center_min)
        & (a_ch < p.a_center_max)
    )
    # 光晕掩膜
    halo_mask = (
        (l_ch > p.l_halo)
        & (a_ch > p.a_halo_min)
        & (a_ch < p.a_halo_max)
    )
    mask = center_mask | halo_mask

    mask_u8 = mask.astype(np.uint8) * 255

    # ====== ROI 排除边缘 ======
    margin_x = int(w * p.roi_margin)
    margin_y = int(h * p.roi_margin)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    mask_u8 = cv2.bitwise_and(mask_u8, roi_mask)

    # # ====== 形态学操作 ======
    # kernel = np.ones((p.morph_kernel, p.morph_kernel), dtype=np.uint8)
    # mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    # mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # ====== 轮廓提取 & 面积过滤 ======
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        if debug:
            return None, mask_u8
        return None, annotated

    min_area = p.area_min_bright if l_mean > p.brightness_thresh else p.area_min_dark
    max_area = p.area_max_bright if l_mean > p.brightness_thresh else p.area_max_dark

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            valid_contours.append(cnt)

    if not valid_contours:
        if debug:
            return None, mask_u8
        return None, annotated

    largest = max(valid_contours, key=cv2.contourArea)

    # ====== 质心计算 ======
    m = cv2.moments(largest)
    if m["m00"] == 0:
        if debug:
            return None, mask_u8
        return None, annotated

    u = float(m["m10"] / m["m00"])
    v = float(m["m01"] / m["m00"])

    # ====== 质心验证 ======
    mrg = p.centroid_margin
    if u < mrg or u > w - mrg or v < mrg or v > h - mrg:
        if debug:
            return None, mask_u8
        return None, annotated

    x_box, y_box, bw, bh = cv2.boundingRect(largest)
    if x_box < mrg or x_box + bw > w - mrg or y_box < mrg or y_box + bh > h - mrg:
        if debug:
            return None, mask_u8
        return None, annotated

    # ====== 调试输出 ======
    if debug:
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(final_mask, [largest], -1, 255, -1)
        cv2.circle(final_mask, (int(u), int(v)), 5, 128, -1)
        return (u, v), final_mask

    # ====== 可视化 ======
    ui, vi = int(round(u)), int(round(v))

    cross_len = 12
    cv2.line(annotated, (ui - cross_len, vi), (ui + cross_len, vi), (0, 255, 0), 2)
    cv2.line(annotated, (ui, vi - cross_len), (ui, vi + cross_len), (0, 255, 0), 2)
    cv2.circle(annotated, (ui, vi), 6, (0, 0, 255), 1)

    cv2.putText(
        annotated,
        f"({ui},{vi})",
        (ui + 10, vi - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
    )

    cv2.putText(
        annotated,
        f"L_mean={l_mean:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
    )

    return (u, v), annotated

"""云台坐标系全自动网格标定模块 (LUT Base)。

通过自动盲探获取摄像头视野边界，生成 3x3 绝对映射网格 (Look-Up Table)。
内置 LUTMapper 插值器，提供像素坐标到绝对角度的精确映射，彻底消除透视畸变。
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time

import cv2
import numpy as np

from camera import Camera
from motor import MotorController
from tracker import process_laser_detection

logger = logging.getLogger(__name__)

DEFAULT_CALIB_FILE = "calibration.json"


class LUTMapper:
    """空间插值器：基于 IDW (反距离权重) 将像素坐标精确映射为云台绝对角度。"""

    def __init__(self, lut_data: list[dict]) -> None:
        self.lut = lut_data

    def pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        """输入像素(px, py)，返回插值计算后的绝对角度(az, pt)。"""
        sum_w = 0.0
        sum_az = 0.0
        sum_pt = 0.0
        
        for item in self.lut:
            p_x, p_y = item["pixel"]
            az, pt = item["angle"]
            
            # 计算两点距离
            dist = math.hypot(px - p_x, py - p_y)
            # 如果靠得极近，直接返回
            if dist < 1e-3:
                return az, pt
            
            # 权重为距离平方的倒数（越近影响越大）
            w = 1.0 / (dist ** 2)
            sum_w += w
            sum_az += az * w
            sum_pt += pt * w
            
        return sum_az / sum_w, sum_pt / sum_w


def load_calibration(path: str = DEFAULT_CALIB_FILE) -> LUTMapper | None:
    """从文件加载 LUT 网格，并返回 Mapper 插值器。"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lut_data = data.get("lut_table")
        if not lut_data:
            return None
        logger.info(f"已加载标定文件: {path} (共 {len(lut_data)} 个网格点)")
        return LUTMapper(lut_data)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def save_calibration(lut_data: list[dict], path: str = DEFAULT_CALIB_FILE) -> None:
    """保存 LUT 网格数据到文件。"""
    data = {"lut_table": lut_data}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"标定网格已保存: {path}")


class Calibrator:
    """全自动网格探测标定器。"""

    def __init__(self, cam: Camera, motor: MotorController) -> None:
        self._cam = cam
        self._motor = motor
        self._lut_points: list[tuple[float, float]] = []
        self._trail_lock = threading.Lock()

        # 可视化状态
        self._current_pos: tuple[float, float] | None = None
        self._status_text: str = "Calib ready"
        
        # 记录线程
        self._recorder_thread: threading.Thread | None = None
        self._recording = False

    def run(self) -> list[dict] | None:
        """执行全自动盲探标定，返回 9 个点的 LUT 数据。"""
        self._recording = True
        self._recorder_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._recorder_thread.start()

        try:
            return self._do_calibration()
        finally:
            self._recording = False

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """绘制正在扫描的网格点。"""
        annotated = frame.copy()

        with self._trail_lock:
            points = list(self._lut_points)
            pos = self._current_pos

        # 画已经采样的网格点
        for p_x, p_y in points:
            cv2.circle(annotated, (int(p_x), int(p_y)), 5, (255, 128, 0), -1)

        # 当前激光位置
        if pos is not None:
            xi, yi = int(pos[0]), int(pos[1])
            cv2.drawMarker(annotated, (xi, yi), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        cv2.putText(annotated, self._status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"Grid Points: {len(points)}/9", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated

    def _record_loop(self) -> None:
        while self._recording:
            frame = self._cam.read()
            if frame is None:
                time.sleep(0.03)
                continue
            pos, _ = process_laser_detection(frame)
            with self._trail_lock:
                self._current_pos = pos
            time.sleep(0.03)

    def _send_position(self, azimuth_deg: float, pitch_deg: float) -> None:
        self._motor.set_position(azimuth_deg, pitch_deg)

    def _sample_pos(self, n: int = 5) -> tuple[float, float] | None:
        positions = []
        for _ in range(n):
            frame = self._cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            pos, _ = process_laser_detection(frame)
            if pos is not None:
                positions.append(pos)
            time.sleep(0.05)
        if len(positions) < n * 0.5:
            return None
        arr = np.array(positions)
        mean = arr.mean(axis=0)
        return float(mean[0]), float(mean[1])

    def _do_calibration(self) -> list[dict] | None:
        print("\n========== 阶段 1: 寻找原点 ==========")
        self._status_text = "Zeroing (0,0)..."
        self._send_position(0.0, 0.0)
        time.sleep(2.0)
        pos0 = self._sample_pos(10)
        if pos0 is None:
            print("❌ 错误：原点未检测到激光！请调整云台使激光处于画面内。")
            return None
        print(f"✅ 原点找到: {pos0[0]:.1f}, {pos0[1]:.1f}")

        margin_x = 640 * 0.10
        margin_y = 480 * 0.10

        def probe_boundary(d_az: float, d_pt: float) -> tuple[float, float]:
            """单步盲探边界，增加防黑斑容错逻辑"""
            az, pt = 0.0, 0.0
            last_valid_az, last_valid_pt = 0.0, 0.0
            lost_count = 0  # 容错计数器
            
            while True:
                az += d_az
                pt += d_pt
                self._send_position(az, pt)
                time.sleep(0.3)
                p = self._sample_pos(3)
                
                if p is not None:
                    # 1. 如果看到了激光，清零丢失计数器，并记录为安全位置
                    lost_count = 0
                    last_valid_az, last_valid_pt = az, pt
                    
                    # 2. 判断是否撞到了视野边缘 (触发真实物理边界)
                    if p[0] < margin_x or p[0] > 640 - margin_x or p[1] < margin_y or p[1] > 480 - margin_y:
                        last_valid_az, last_valid_pt = az - d_az, pt - d_pt # 安全退回一步
                        break
                else:
                    # 3. 没看到激光！可能是跨越了黑胶带，也可能是飞出了屏幕外
                    lost_count += 1
                    if lost_count >= 3: 
                        # 连续 3 度都没看到（也就是 3 次没检测到），确定是飞出去了，停止盲探
                        break
                        
                # 4. 机械极限保护，防止死循环
                if abs(az) >= 30.0 or abs(pt) >= 30.0:
                    break
                    
            return last_valid_az, last_valid_pt

        print("\n========== 阶段 2: 盲探安全边界 ==========")
        self._status_text = "Probing Boundaries..."
        
        az_max, _ = probe_boundary(1.0, 0.0)
        self._send_position(0.0, 0.0); time.sleep(1)
        
        az_min, _ = probe_boundary(-1.0, 0.0)
        self._send_position(0.0, 0.0); time.sleep(1)
        
        _, pt_max = probe_boundary(0.0, 1.0)
        self._send_position(0.0, 0.0); time.sleep(1)
        
        _, pt_min = probe_boundary(0.0, -1.0)
        self._send_position(0.0, 0.0); time.sleep(1)

        print(f"✅ 边界探明: Az[{az_min:.1f}, {az_max:.1f}] Pt[{pt_min:.1f}, {pt_max:.1f}]")

        print("\n========== 阶段 3: 记录 3x3 绝对网格 ==========")
        az_list = [az_min, 0.0, az_max]
        pt_list = [pt_min, 0.0, pt_max]
        
        lut_table = []
        for pt in pt_list:
            for az in az_list:
                self._status_text = f"Sampling Az:{az:.1f} Pt:{pt:.1f}"
                self._send_position(az, pt)
                time.sleep(0.8) # 给云台绝对移动留够时间
                p = self._sample_pos(10)
                if p:
                    lut_table.append({"angle": (az, pt), "pixel": p})
                    with self._trail_lock:
                        self._lut_points.append(p)
                    print(f"  网格点记录: 角度({az:.1f}, {pt:.1f}) -> 像素({p[0]:.1f}, {p[1]:.1f})")

        self._motor.stop()
        self._status_text = "Calibration Done!"
        print(f"\n✅ 网格标定完成，共记录 {len(lut_table)} 个节点。")
        return lut_table
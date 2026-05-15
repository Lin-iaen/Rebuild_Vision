"""交互式屏幕角点标定模块。

用户手动将激光照射屏幕四个角，通过 GPIO 按键记录像素坐标。
记录完成后自动排序为 [左上, 右上, 右下, 左下]。

通过全局变量 _screen_corners / _screen_progress 供 _frame_provider 渲染。

用法::

    sc = ScreenCalibrator(cam, keys)
    corners = sc.run()
    # 返回 [TL, TR, BR, BL] 或 None(取消)
"""

from __future__ import annotations

import time

import numpy as np

from camera import Camera
from gpio_keys import KeypadController
from tracker import process_laser_detection

# 角点标签（记录顺序不影响最终排序，显示时用）
CORNER_LABELS = ["P0 左上", "P1 右上", "P2 右下", "P3 左下"]


def _order_quad_points(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """将 4 个点排序为 [左上, 右上, 右下, 左下]。"""
    arr = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1)
    rect[0] = arr[np.argmin(s)]
    rect[2] = arr[np.argmax(s)]
    rect[1] = arr[np.argmin(diff)]
    rect[3] = arr[np.argmax(diff)]
    return [(float(x), float(y)) for x, y in rect]


def _detect_laser(cam: Camera) -> tuple[float, float] | None:
    """快速检测激光位置。"""
    for _ in range(5):
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pos, _ = process_laser_detection(frame)
        if pos is not None:
            return pos
        time.sleep(0.05)
    return None


class ScreenCalibrator:
    """交互式屏幕角点标定器。

    参数:
        cam:     摄像头实例
        keys:    KeypadController 实例
        corners: 外部可变列表 (写入已记录的点，供 _frame_provider 渲染)
        done:    bool 标志位 (写入标定完成状态)
    """

    def __init__(
        self,
        cam: Camera,
        keys: KeypadController,
        corners: list,      # 外部可变引用，用于 _frame_provider 实时渲染
        done_flag: list,     # [bool] 可变引用
        record_key: str = "enter",
        undo_key: str = "undo",
        cancel_key: str = "q",
    ) -> None:
        self._cam = cam
        self._keys = keys
        self._corners = corners
        self._done_flag = done_flag
        self._record_key = record_key
        self._undo_key = undo_key
        self._cancel_key = cancel_key

    def run(self) -> list[tuple[float, float]] | None:
        """运行标定流程。

        返回: 排序后的 4 个角点 [TL, TR, BR, BL]，用户取消返回 None。
        """
        raw: list[tuple[float, float]] = []
        self._corners.clear()
        self._done_flag[0] = False

        print()
        print("=" * 50)
        print("屏幕角点标定 (Phase 1)")
        print("=" * 50)
        print("请用激光笔依次照射屏幕的四个角")
        print("记录顺序任意，完成后自动排序")
        print()
        print("  按键:")
        print("    [enter]  记录当前激光位置")
        print("    [undo]   撤销上一个记录点")
        print("    [q]      取消退出")
        print("-" * 50)

        while len(raw) < 4:
            pos = _detect_laser(self._cam)

            # 实时反馈
            status_parts = ["■" if i < len(raw) else "□" for i in range(4)]
            progress = "".join(status_parts)
            target_label = CORNER_LABELS[len(raw)] if len(raw) < 4 else ""

            if pos is not None:
                pos_str = f"({pos[0]:.0f}, {pos[1]:.0f})"
            else:
                pos_str = "未检测到"

            print(f"\r  进度: {progress}  {target_label}  激光: {pos_str}   ", end="", flush=True)

            # 更新全局渲染数据
            self._corners[:] = raw

            key = self._keys.wait_key()
            if key is None:
                continue

            if key == self._record_key:
                if pos is not None:
                    raw.append(pos)
                    print(f"\n  [记录] P{len(raw)-1}: ({pos[0]:.0f}, {pos[1]:.0f})")
                else:
                    print("\n  [跳过] 未检测到激光，请重试")

            elif key == self._undo_key:
                if raw:
                    removed = raw.pop()
                    print(f"\n  [撤销] 已移除 ({removed[0]:.0f}, {removed[1]:.0f})")

            elif key == self._cancel_key:
                print("\n\n  标定已取消")
                self._corners.clear()
                return None

        print()
        print("-" * 50)

        # 排序并写入最终结果
        sorted_corners = _order_quad_points(raw)
        self._corners[:] = sorted_corners
        self._done_flag[0] = True

        print("屏幕标定完成!")
        for i, (x, y) in enumerate(sorted_corners):
            label = ["左上", "右上", "右下", "左下"][i]
            print(f"  P{i}({label}): ({x:.1f}, {y:.1f})")

        return sorted_corners

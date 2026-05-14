# AGENTS.md

## Project Overview

IBVS minimal laser-tracking vision system for Raspberry Pi + OV5647 camera. Tracks a laser point on surfaces using LAB color space detection, sends tracking errors over UART to a servo/motor controller.

Python 3, OpenCV, no formal build system or test framework.

## Running

```bash
# Install dependencies (use a venv)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the main loop
python3 main.py
```

No linter, formatter, or type checker is configured. There is no CI.

**Operational gotchas:**
- `main.py` runs `pkill -f flask` on startup — kills ALL Flask processes on the machine.
- Camera warmup: up to 5 seconds (50 iterations of 0.1s) before first valid frame. Timeout exits.
- Web stream always binds to port 5000 (hardcoded in `MjpegStream`).
- `pyserial` is required at runtime but **not listed in `requirements.txt`** — install manually.

## Architecture

Main entry point is `main.py` with a state machine:

```
INIT → 矩形检测 → CALIBRATE(可选) → READY → RESET / TRACK
```

### Core Modules

| Module | File | Status | Purpose |
|--------|------|--------|---------|
| Sensor | `camera.py` | Implemented | rpicam-vid 摄像头采集封装 |
| Vision | `tracker.py` | Implemented | LAB 色彩空间激光检测 |
| Stream | `web_stream.py` | Implemented | 通用 MJPEG 推流服务 |
| Serial | `uart.py` | Implemented | 通用串口通信模块 |
| Gimbal | `gimbal.py` | Implemented | 云台控制 + M⁻¹ 矩阵变换 |
| Rectangle | `rectangle.py` | Implemented | 矩形检测 + 子目标生成 |
| Calibration | `calibration.py` | Implemented | 云台坐标系标定 |
| Control | `control.py` | Implemented | 循迹控制逻辑 |
| Main | `main.py` | Implemented | 状态机 + 菜单 + Web监控 |

### Module Details

#### camera.py
`rpicam-vid` 子进程封装，提供线程安全的 `Camera` 类。硬件参数锁死：shutter=33239, gain=8.0, awb=auto, 640×480, 30fps。

#### tracker.py
核心检测代码。使用 **LAB 色彩空间**（不是 HSV）进行激光点检测。提供两个函数：
- `process_laser_detection(frame)` — 激光点检测，返回 (x, y) 坐标
- `process_init_mode(frame)` — 四边形标定点检测

**检测参数**：
- L 通道阈值：L_center=230, L_halo=60
- A 通道阈值：125 < A < 150（中心），130 < A < 145（光晕）
- A 值上界作用：排除边缘色差伪影（A=143-157）
- ROI 掩膜：排除画面边缘 10%

#### web_stream.py
通用 MJPEG 推流服务 `MjpegStream`。零耦合设计，接受回调函数 `frame_provider() -> bytes`，由调用方决定推什么画面。

#### uart.py
通用串口通信模块。提供 `UartController` 类，`send_raw(data)` 方法发送原始字节。不耦合任何协议格式。

#### gimbal.py
云台控制模块。封装新云台协议，通过 M⁻¹ 矩阵将像素误差转换为角度指令。
- 协议：`0x02 0x01/0x02 <angle_i16>`（X/Y轴，值 = 角度 × 100）
- `GimbalController.move(delta_px, delta_py)` — 像素误差 → 角度指令
- **注意:** `DEFAULT_M_INV` (单位矩阵) 在 `gimbal.py:24` 和 `main.py:32` 各定义了一份，修改时需要同步。

#### rectangle.py
矩形检测与管理模块。检测黑色电工胶带矩形框，生成循迹子目标点。
- `RectangleManager.detect(frame)` — 检测矩形
- `get_targets()` — 生成顺时针子目标点列表
- 角点顺序：[左上, 右上, 右下, 左下]

#### calibration.py
云台坐标系标定模块。通过发送已知角度指令，记录激光轨迹，反解出 2×2 变换矩阵。
- `Calibrator.run()` — 执行标定，返回 M⁻¹ 矩阵
- `load_calibration()` / `save_calibration()` — 文件读写
- 标定结果保存到 `calibration.json`

#### control.py
循迹控制模块。实现连续反馈循环：检测激光 → 计算误差 → M⁻¹ 变换 → 发送指令。
- `LaserTracker.reset_to_center()` — 复位到矩形中心
- `track_rectangle()` — 绕矩形循迹一圈
- 控制频率：20fps（50ms 延时）
- 噪声剔除：距离上次位置超过 50px 则判定为噪声，丢弃
- 检测失败：发送零指令，云台保持静止

#### main.py
主入口。状态机架构，集成所有模块：
- 矩形检测
- 标定（可选，可跳过）
- 复位到中心
- 绕矩形循迹
- Web 监控（浏览器实时查看）

## Testing

No pytest/unittest. Tests live in `sample/` as standalone scripts meant to run on the Raspberry Pi with real hardware.

All test scripts use `MjpegStream` for browser-based visualization. OpenCV GUI calls (`cv2.imshow`) are not used — tests work in headless environments.

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_camera_stream.py` | 摄像头采集验证 | `python3 sample/test_camera_stream.py` |
| `test_tracker_stream.py` | 激光追踪推流验证 | `python3 sample/test_tracker_stream.py` |
| `test_rectangle.py` | 黑色胶带矩形标定 | `python3 sample/test_rectangle.py` |
| `test_align.py` | 云台坐标系标定 | `python3 sample/test_align.py` |
| `test_camera.py` | 单帧拍照水印测试 | `python3 sample/test_camera.py` |
| `test_gain.py` | 增益参数调试 | `python3 sample/test_gain.py` |
| `test_lab_diagnostic.py` | LAB 值诊断 | `python3 sample/test_lab_diagnostic.py` |
| `test_optimized_lab.py` | 优化 LAB 算法测试 | `python3 sample/test_optimized_lab.py` |
| `test_can.py` | CAN 帧协议测试 | `python3 sample/test_can.py` |
| `test_uart.py` | 串口环回测试 | `python3 sample/test_uart.py` |
| `test_lab_vs_hsv.py` | LAB vs HSV 对比测试 | `python3 sample/test_lab_vs_hsv.py` |

`sample/test_laser.py` is an empty file (stale). `sample/test_stream.py` is a shorter duplicate of `test_camera_stream.py`.

## Key Hardware Constants (do not change without re-measurement)

From `README.md` and `tracker.py`:
- Shutter: 33239 us, Gain: 8.0, AWB: auto
- LAB thresholds in `tracker.py`: L_center=230, A_center=[125, 150], L_halo=60, A_halo=[130, 145]
- Laser area filter: 3–800 px (bright env) or 5–500 px (dark env)
- Morphology kernel: 5×5, close then open

## Communication Protocols

### 新云台协议（gimbal.py）

Serial 模式 — 单轴指令，4 字节帧：
- X轴：`0x02 0x01 <value_h> <value_l>`
- Y轴：`0x02 0x02 <value_h> <value_l>`
- 数据：有符号 int16 大端，值 = 角度 × 100

CAN 模式 — 双轴打包，10 字节帧：
- 帧结构：`[0x06 0x82] [0x00 0x00 0x00 0x00] [X_i16] [Y_i16]`
- X/Y 数据：有符号 int16 大端，值 = 角度 × 100
- 通过 `GimbalController(uart, mode='can')` 切换

### 标定结果（calibration.json）
M⁻¹ 矩阵保存为 JSON 格式，用于像素误差到角度指令的转换。

## Conventions

- Chinese comments and log messages throughout; match this style.
- `requirements.txt` lists runtime deps: opencv-python-headless, numpy, Flask. **Note: `pyserial` is also required** (`uart.py` imports `serial`) but is not listed in `requirements.txt` — install it manually with `pip install pyserial`.
- No dev dependencies exist.
- `.gitignore` excludes `*.jpg`, `*.png`, `logs/`, and `sample/align.txt`.
- The `venv/` directory is gitignored but present on disk.
- Video frame text uses English (OpenCV doesn't support Chinese in putText).
- Terminal output uses Chinese.

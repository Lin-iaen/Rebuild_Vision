# AGENTS.md

## Project Overview

IBVS minimal laser-tracking vision system for Raspberry Pi + OV5647 camera. Tracks a laser point on surfaces using LAB color space detection, sends tracking errors over UART to a servo/motor controller.

Python 3, OpenCV, no formal build system or test framework.

## Running

```bash
# Install dependencies (use a venv)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the main loop (currently a stub)
python3 main.py
```

No linter, formatter, or type checker is configured. There is no CI.

## Architecture

Four-layer pipeline described in `README.md`. Current state of implementation:

| Layer | File | Status |
|-------|------|--------|
| Sensor (camera) | `camera.py` | Implemented |
| Vision pipeline | `tracker.py` | Implemented |
| Tracking logic | `main.py` | Stub (`import cv2` only) |
| Actuator (UART) | `uart.py` | Implemented |
| Web debug stream | `web_stream.py` | Implemented |

### camera.py
`rpicam-vid` 子进程封装，提供线程安全的 `Camera` 类。硬件参数锁死：shutter=33239, gain=8.0, awb=auto, 640×480, 30fps。

### tracker.py
核心检测代码。使用 **LAB 色彩空间**（不是 HSV）进行激光点检测。提供两个函数：
- `process_laser_detection(frame)` — 激光点检测，返回 (x, y) 坐标
- `process_init_mode(frame)` — 四边形标定点检测

### web_stream.py
通用 MJPEG 推流服务 `MjpegStream`。零耦合设计，接受回调函数 `frame_provider() -> bytes`，由调用方决定推什么画面。

### uart.py
UART 通信模块。旧协议：`0xAA 0x55 <dx_i16> <dy_i16> <checksum> 0x0A`（大端）。新云台协议：`0x02 0x01/0x02 <value_i16>`（单轴指令，值 = 角度 × 100）。

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
| `test_uart.py` | 串口环回测试 | `python3 sample/test_uart.py` |

## Key Hardware Constants (do not change without re-measurement)

From `README.md` and `tracker.py`:
- Shutter: 33239 us, Gain: 8.0, AWB: auto
- LAB thresholds in `tracker.py`: L_center=180, A_center=125, L_halo=60, A_halo=130
- Laser area filter: 3–800 px (bright env) or 5–500 px (dark env)
- Morphology kernel: 5×5, close then open

## Communication Protocols

### 旧协议（uart.py — send_error）
`0xAA 0x55 <dx_i16> <dy_i16> <checksum> 0x0A`（大端，值 = 像素误差 × 10）

### 新云台协议（test_align.py）
单轴指令，4 字节帧：
- X轴：`0x02 0x01 <value_h> <value_l>`
- Y轴：`0x02 0x02 <value_h> <value_l>`
- 数据：有符号 int16 大端，值 = 角度 × 100

## Conventions

- Chinese comments and log messages throughout; match this style.
- `requirements.txt` lists runtime deps: opencv-python-headless, numpy, Flask, pyserial. No dev dependencies exist.
- `.gitignore` excludes `*.jpg`, `*.png`, and `sample/align.txt`.
- The `venv/` directory is gitignored but present on disk.

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
| Sensor (camera) | `camera.py` | Empty |
| Vision pipeline | `tracker.py` | Implemented |
| Tracking logic | `main.py` | Stub (`import cv2` only) |
| Actuator (UART) | `uart.py` | Implemented |
| Web debug stream | `web_stream.py` | Empty |

`tracker.py` is the real working code. It uses **LAB color space** (not HSV) for laser detection. The README's HSV/morphology description is outdated — trust the code.

`uart.py` sends a binary frame: `0xAA 0x55 <dx_i16> <dy_i16> <checksum> 0x0A` (big-endian).

## Testing

No pytest/unittest. Tests live in `sample/` as standalone scripts meant to run on the Raspberry Pi with real hardware. Most are currently empty stubs. `test_uart.py` is a serial loopback test that requires physical wire between GPIO pins 8 and 10.

Do not assume any test command will work in a headless environment — OpenCV GUI calls (`cv2.imshow`) will crash without a display.

## Key Hardware Constants (do not change without re-measurement)

From `README.md` and `tracker.py`:
- Shutter: 33239 us, Gain: 8.0, AWB: auto
- LAB thresholds in `tracker.py`: L_center=180, A_center=125, L_halo=60, A_halo=130
- Laser area filter: 3–800 px (bright env) or 5–500 px (dark env)
- Morphology kernel: 5×5, close then open

## Conventions

- Chinese comments and log messages throughout; match this style.
- `requirements.txt` lists only runtime deps (opencv-python-headless, numpy, Flask). No dev dependencies exist.
- `.gitignore` excludes `*.jpg` and `*.png` — do not commit test images.
- The `venv/` directory is gitignored but present on disk.

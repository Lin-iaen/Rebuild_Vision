"""Web 激光检测调试工具。

浏览器中实时拖动滑块调整 LAB 检测参数，所见即所得。
支持两种模式：检测模式（原图+掩膜）和诊断模式（LAB 通道）。

用法:
    python3 sample/test_laser_tuner.py
    浏览器访问 http://<树莓派IP>:5000
"""

import copy
import json
import subprocess
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import sys
sys.path.insert(0, ".")
from camera import Camera
from tracker import LaserParams, process_laser_detection

# ===== 全局状态 =====
app = Flask(__name__)
_params = LaserParams()
_params_lock = threading.Lock()
_mode = "detect"       # "detect" | "diagnostic"

# ===== 摄像头 =====
_cam = Camera()


def get_frame() -> bytes | None:
    """生成当前帧的 JPEG。"""
    frame = _cam.read()
    if frame is None:
        return None

    with _params_lock:
        p = copy.copy(_params)
    mode = _mode

    if mode == "diagnostic":
        # 诊断模式：用实际检测算法定位，同时显示 LAB 通道
        with _params_lock:
            p_diag = copy.copy(_params)

        pos, annotated = process_laser_detection(frame, params=p_diag)
        _, mask_u8 = process_laser_detection(frame, params=p_diag, debug=True)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # L/A/B 灰度图
        l_bgr = cv2.cvtColor(l_ch, cv2.COLOR_GRAY2BGR)
        a_bgr = cv2.cvtColor(a_ch, cv2.COLOR_GRAY2BGR)
        b_bgr = cv2.cvtColor(b_ch, cv2.COLOR_GRAY2BGR)

        # 在所有通道图上标记检测位置和 LAB 值
        if pos:
            mx, my = int(pos[0]), int(pos[1])
            l_val = int(l_ch[my, mx])
            a_val = int(a_ch[my, mx])
            b_val = int(b_ch[my, mx])
            label = f"L={l_val} A={a_val} B={b_val}"

            for img in [annotated, l_bgr, a_bgr, b_bgr]:
                cv2.circle(img, (mx, my), 12, (0, 0, 255), 2)
                cv2.putText(img, label, (mx + 15, my - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # HUD
            cv2.putText(annotated, f"DETECT ({mx},{my})  {label}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(annotated, "NO LASER", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 各通道标题
        cv2.putText(l_bgr, "L channel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(a_bgr, "A channel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(b_bgr, "B channel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 掩膜
        mask_bgr = cv2.cvtColor(
            mask_u8 if mask_u8 is not None else np.zeros((h, w), dtype=np.uint8),
            cv2.COLOR_GRAY2BGR,
        )
        mask_px = int(np.sum(mask_u8 > 0)) if mask_u8 is not None else 0
        cv2.putText(mask_bgr, f"MASK {mask_px}px", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 水平拼接：原图+准星 | L | A | 掩膜
        combined = np.hstack((annotated, l_bgr, a_bgr, mask_bgr))
        h_c, w_c = combined.shape[:2]
        if w_c > 1280:
            scale = 1280 / w_c
            combined = cv2.resize(combined, (1280, int(h_c * scale)))

    else:
        # 检测模式：原图(标注) + 掩膜
        pos, annotated = process_laser_detection(frame, params=p)
        _, mask_u8 = process_laser_detection(frame, params=p, debug=True)

        # 掩膜 → BGR
        mask_bgr = cv2.cvtColor(mask_u8 if mask_u8 is not None else np.zeros_like(annotated),
                                cv2.COLOR_GRAY2BGR)

        # HUD
        l_ch = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
        l_mean = float(np.mean(l_ch))
        if pos:
            cv2.putText(annotated, f"DETECT ({pos[0]:.0f},{pos[1]:.0f}) L_mean={l_mean:.0f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(annotated, f"NO LASER  L_mean={l_mean:.0f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 在掩膜上显示 mask 像素数
        mask_px = int(np.sum(mask_u8 > 0)) if mask_u8 is not None else 0
        cv2.putText(mask_bgr, f"MASK {mask_px}px", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        combined = np.hstack((annotated, mask_bgr))

    ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


# ===== Flask 路由 =====

@app.route("/")
def index():
    return _PAGE_HTML


@app.route("/params", methods=["POST"])
def update_params():
    global _params
    data = request.get_json(silent=True) or {}
    with _params_lock:
        for key, value in data.items():
            if hasattr(_params, key):
                current_type = type(getattr(_params, key))
                setattr(_params, key, current_type(value))
    return jsonify({"status": "ok"})


@app.route("/mode", methods=["POST"])
def set_mode():
    global _mode
    data = request.get_json(silent=True) or {}
    _mode = data.get("mode", "detect")
    return jsonify({"status": "ok", "mode": _mode})


@app.route("/params/get")
def get_params():
    with _params_lock:
        d = {f: getattr(_params, f) for f in _params.__dataclass_fields__}
    return jsonify(d)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            jpeg = get_frame()
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ===== HTML 模板 =====

_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>激光检测调参</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:monospace;padding:10px}
h1{font-size:16px;margin:5px 0;color:#00ff88}
#video{width:100%;max-width:1280px;border:2px solid #333;display:block;margin:0 auto}
.panel{max-width:1280px;margin:10px auto}
.modes{display:flex;gap:10px;margin-bottom:10px}
.modes button{padding:8px 20px;border:none;cursor:pointer;font-size:14px;font-family:monospace}
.modes button.active{background:#00ff88;color:#000}
.modes button.inactive{background:#333;color:#aaa}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.slider-group{background:#16213e;padding:8px 12px;border-radius:4px}
.slider-group label{display:block;font-size:11px;color:#aaa;margin-bottom:2px}
.slider-group .row{display:flex;align-items:center;gap:8px}
.slider-group input[type=range]{flex:1;height:6px;-webkit-appearance:none;appearance:none;background:#333;border-radius:3px;outline:none}
.slider-group input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:#00ff88;border-radius:50%;cursor:pointer}
.slider-group .val{font-size:12px;color:#00ff88;min-width:45px;text-align:right}
.btn-row{margin-top:10px;display:flex;gap:10px}
.btn-row button{padding:6px 16px;border:none;cursor:pointer;font-size:12px;font-family:monospace;background:#333;color:#aaa;border-radius:3px}
.btn-row button:hover{background:#444}
</style>
</head>
<body>
<h1>激光检测调参</h1>
<img id="video" src="/video_feed" alt="stream">

<div class="panel">
  <div class="modes">
    <button id="btn-detect" class="active" onclick="switchMode('detect')">检测模式</button>
    <button id="btn-diag" class="inactive" onclick="switchMode('diagnostic')">诊断模式</button>
  </div>

  <div class="grid">
    <div class="slider-group">
      <label>L center</label>
      <div class="row"><input type="range" id="l_center" min="80" max="255" value="230" oninput="update(this)"><span class="val" id="val_l_center">230</span></div>
    </div>
    <div class="slider-group">
      <label>L halo</label>
      <div class="row"><input type="range" id="l_halo" min="30" max="200" value="60" oninput="update(this)"><span class="val" id="val_l_halo">60</span></div>
    </div>
    <div class="slider-group">
      <label>A center min</label>
      <div class="row"><input type="range" id="a_center_min" min="100" max="170" value="125" oninput="update(this)"><span class="val" id="val_a_center_min">125</span></div>
    </div>
    <div class="slider-group">
      <label>A center max</label>
      <div class="row"><input type="range" id="a_center_max" min="125" max="255" value="150" oninput="update(this)"><span class="val" id="val_a_center_max">150</span></div>
    </div>
    <div class="slider-group">
      <label>A halo min</label>
      <div class="row"><input type="range" id="a_halo_min" min="100" max="170" value="130" oninput="update(this)"><span class="val" id="val_a_halo_min">130</span></div>
    </div>
    <div class="slider-group">
      <label>A halo max</label>
      <div class="row"><input type="range" id="a_halo_max" min="125" max="200" value="145" oninput="update(this)"><span class="val" id="val_a_halo_max">145</span></div>
    </div>
    <div class="slider-group">
      <label>ROI margin</label>
      <div class="row"><input type="range" id="roi_margin" min="0" max="0.5" step="0.01" value="0.1" oninput="update(this)"><span class="val" id="val_roi_margin">0.1</span></div>
    </div>
    <div class="slider-group">
      <label>Area min (bright)</label>
      <div class="row"><input type="range" id="area_min_bright" min="1" max="50" value="3" oninput="update(this)"><span class="val" id="val_area_min_bright">3</span></div>
    </div>
    <div class="slider-group">
      <label>Area max (bright)</label>
      <div class="row"><input type="range" id="area_max_bright" min="100" max="3000" value="800" oninput="update(this)"><span class="val" id="val_area_max_bright">800</span></div>
    </div>
    <div class="slider-group">
      <label>Morph kernel</label>
      <div class="row"><input type="range" id="morph_kernel" min="1" max="15" value="5" oninput="update(this)"><span class="val" id="val_morph_kernel">5</span></div>
    </div>
    <div class="slider-group">
      <label>Centroid margin</label>
      <div class="row"><input type="range" id="centroid_margin" min="5" max="100" value="30" oninput="update(this)"><span class="val" id="val_centroid_margin">30</span></div>
    </div>
    <div class="slider-group">
      <label>Brightness thresh</label>
      <div class="row"><input type="range" id="brightness_thresh" min="50" max="220" value="150" oninput="update(this)"><span class="val" id="val_brightness_thresh">150</span></div>
    </div>
  </div>

  <div class="btn-row">
    <button onclick="resetDefaults()">恢复默认</button>
    <button onclick="savePreset()">保存预设</button>
    <button onclick="loadPreset()">加载预设</button>
  </div>
</div>

<script>
var timer = null;

function update(slider) {
  document.getElementById('val_' + slider.id).textContent = slider.value;
  clearTimeout(timer);
  timer = setTimeout(function() {
    var data = {};
    data[slider.id] = slider.value;
    fetch('/params', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
  }, 200);
}

function switchMode(mode) {
  document.getElementById('btn-detect').className = mode === 'detect' ? 'active' : 'inactive';
  document.getElementById('btn-diag').className = mode === 'diagnostic' ? 'active' : 'inactive';
  fetch('/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: mode})
  });
}

function resetDefaults() {
  var defaults = {
    l_center:230,l_halo:60,a_center_min:125,a_center_max:150,a_halo_min:130,a_halo_max:145,
    roi_margin:0.1,area_min_bright:3,area_max_bright:800,morph_kernel:5,centroid_margin:30,brightness_thresh:150
  };
  Object.entries(defaults).forEach(function(e) {
    var el = document.getElementById(e[0]);
    if (el) { el.value = e[1]; document.getElementById('val_'+e[0]).textContent = e[1]; }
  });
  fetch('/params', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(defaults)
  });
}

function savePreset() {
  fetch('/params/get')
    .then(function(r) { return r.json(); })
    .then(function(p) {
      localStorage.setItem('laser_params', JSON.stringify(p));
      alert('预设已保存到浏览器');
    });
}

function loadPreset() {
  var raw = localStorage.getItem('laser_params');
  if (!raw) { alert('没有已保存的预设'); return; }
  var p = JSON.parse(raw);
  Object.entries(p).forEach(function(e) {
    var el = document.getElementById(e[0]);
    if (el) { el.value = e[1]; document.getElementById('val_'+e[0]).textContent = e[1]; }
  });
  fetch('/params', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(p)
  });
}

// 初始加载
window.onload = function() {
  fetch('/params/get')
    .then(function(r) { return r.json(); })
    .then(function(p) {
      Object.entries(p).forEach(function(e) {
        var el = document.getElementById(e[0]);
        if (el) { el.value = e[1]; document.getElementById('val_'+e[0]).textContent = e[1]; }
      });
    });
};
</script>
</body>
</html>
"""


def main():
    print("=" * 50)
    print("激光检测调参工具")
    print("=" * 50)

    subprocess.run(["pkill", "-f", "flask"], check=False)
    time.sleep(0.3)

    # ---- 摄像头 ----
    _cam.start()
    print("等待摄像头就绪...", end="", flush=True)
    for _ in range(50):
        if _cam.read() is not None:
            print(" OK")
            break
        time.sleep(0.1)
    else:
        print(" 超时！")
        _cam.release()
        return

    print(f"Web: http://0.0.0.0:5000")
    print("浏览器中拖动滑块实时调参")
    print("Ctrl+C 退出")

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _cam.release()


if __name__ == "__main__":
    main()

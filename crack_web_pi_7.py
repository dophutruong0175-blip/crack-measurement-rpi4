import json
import os
import sys
import time
import argparse
import threading
import io
from pathlib import Path

import cv2
import numpy as np

try:
    from flask import Flask, jsonify, send_file, render_template_string, make_response, Response, request, redirect
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

from crack_core import analyze_image_from_array
import crack_db


# Chessboard calibration
CHESSBOARD_CAMERA_MATRIX = np.array(
    [
        [1.48946887e+03, 0.00000000e+00, 6.97539281e+02],
        [0.00000000e+00, 1.45954884e+03, 3.58917468e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
    ],
    dtype=np.float64,
)
CHESSBOARD_DIST_COEFFS = np.array(
    [[ 0.13908682, -0.63114838, -0.01290731,  0.00876343,  2.50706406]],
    dtype=np.float64,
).reshape(-1, 1)


WINDOW_NAME = "Crack Measurement Pi4"
COLOR_OK    = (0, 230, 80)
COLOR_WARN  = (0, 200, 255)
COLOR_ERR   = (50,  50, 255)
OVERLAY_BG  = (20,  20,  20)

_shared = {
    "payload":      None,
    "overlay_jpg":  None,
    "binary_jpg":   None,
    "skeleton_jpg": None,
    "display_jpg":  None,   # live stream frame
    "command":      None,   # "capture" | "retake" -- set by web buttons
    "lock":         threading.Lock(),
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crack Measurement</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#0f0f1a;color:#eee;padding:12px}
.header{background:#111827;padding:8px 14px;border-radius:6px;margin-bottom:12px;
        display:flex;justify-content:space-between;align-items:center}
.header-title{color:#00e676;font-weight:bold;font-size:14px}
.header-sub{font-size:10px;color:#555}
.grid{display:grid;grid-template-columns:2fr 3fr;gap:12px}
.card{background:#1e1e2e;border-radius:6px;padding:10px;margin-bottom:10px}
.card-label{color:#555;font-size:9px;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:8px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.metric{background:#12122a;padding:8px;border-radius:4px}
.mname{color:#555;font-size:8px}
.mval{font-size:16px;font-weight:bold}
.munit{font-size:10px;font-weight:normal}
.green{color:#00e676}.yellow{color:#ffc800}
.blue{color:#42a5f5}.purple{color:#ce93d8}
.meta{margin-top:6px;padding-top:6px;border-top:1px solid #2a2a3e;
      display:flex;justify-content:space-between;align-items:center;
      font-size:9px;color:#555}
.badge-ok{background:#0d2a0d;color:#66bb6a;padding:2px 7px;border-radius:3px;font-size:8px}
.badge-err{background:#2a0d0d;color:#ef5350;padding:2px 7px;border-radius:3px;font-size:8px}
table{width:100%;border-collapse:collapse;font-size:9px}
th{text-align:left;padding:3px 4px;font-weight:normal;color:#444}
td{padding:4px}
tr:nth-child(even) td{background:#12122a}
.btn{display:inline-block;padding:8px 20px;border-radius:4px;font-size:11px;
     text-decoration:none;margin-right:8px;border:none;cursor:pointer;font-weight:bold}
.btn-capture{background:#1565c0;color:#fff;font-size:13px;padding:10px 28px}
.btn-retake{background:#333;color:#aaa;border:1px solid #555}
.btn-blue{background:#0d2137;border:1px solid #1565c0;color:#42a5f5}
.btn-green{background:#0d1f0d;border:1px solid #2e7d32;color:#66bb6a}
.img-wrap{background:#1e1e2e;border-radius:6px;padding:10px;margin-bottom:10px}
.img-wrap img{width:100%;border-radius:4px;display:block}
.img-pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.val-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:8px}
.val-ok{background:#0d2a0d;border-radius:3px;padding:5px;color:#66bb6a}
.val-err{background:#2a0d0d;border-radius:3px;padding:5px;color:#ef5350}
.no-data{color:#666;font-style:italic;padding:20px 0}
.stream-wrap{background:#000;border-radius:6px;overflow:hidden;margin-bottom:10px;
             text-align:center;max-height:320px}
.stream-wrap img{width:100%;max-height:320px;object-fit:contain;display:block}
.ctrl{padding:8px 0 4px 0}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="header">
  <span class="header-title">📷 Crack Measurement Pi</span>
  <span style="font-size:10px;color:#555" id="ts">--</span>
</div>

<!-- LIVE STREAM + CONTROLS -->
<div class="stream-wrap">
  <div class="card-label" style="padding:6px 10px 4px">📡 Live Camera</div>
  <img src="/stream" alt="live stream">
</div>
<div class="ctrl">
  <form action="/capture" method="post" style="display:inline">
    <button class="btn btn-capture" type="submit">📸 Chụp &amp; Phân tích</button>
  </form>
  <form action="/retake" method="post" style="display:inline">
    <button class="btn btn-retake" type="submit">🔄 Chụp lại</button>
  </form>
</div>

<br>

{% if d %}
<div class="grid">

  <!-- CỘT TRÁI -->
  <div>
    <div class="card">
      <div class="card-label">Kết quả mới nhất</div>
      <div class="metrics">
        <div class="metric">
          <div class="mname">Tổng chiều dài</div>
          <div class="mval green">{{ d.total_length_mm }}<span class="munit"> mm</span></div>
        </div>
        <div class="metric">
          <div class="mname">Rộng lớn nhất</div>
          <div class="mval yellow">{{ d.max_width_mm }}<span class="munit"> mm</span></div>
        </div>
        <div class="metric">
          <div class="mname">Diện tích</div>
          <div class="mval blue">{{ d.area_mm2 }}<span class="munit"> mm²</span></div>
        </div>
        <div class="metric">
          <div class="mname">Calibration</div>
          <div class="mval purple">{{ (d.pixels_per_mm)|round(1) }}<span class="munit"> px/mm</span></div>
        </div>
      </div>
      <div class="meta">
        <span>🕐 {{ d.processed_at }} &mdash; {{ d.segment_count }} đoạn</span>
        {% if d.input_validation and d.input_validation.passed %}
          <span class="badge-ok">✓ Hợp lệ</span>
        {% else %}
          <span class="badge-err">✗ Lỗi ảnh</span>
        {% endif %}
      </div>
    </div>

    <!-- Lịch sử -->
    <div class="card">
      <div class="card-label">Lịch sử đo ({{ history|length }} lần)</div>
      {% if history %}
      <table>
        <tr>
          <th>Giờ</th>
          <th style="text-align:right">Dài</th>
          <th style="text-align:right">Rộng</th>
          <th style="text-align:right">DT</th>
          <th style="text-align:right">Đoạn</th>
        </tr>
        {% for r in history %}
        <tr>
          <td>{{ r.captured_at[11:16] }}</td>
          <td style="text-align:right">{{ r.length_mm }}mm</td>
          <td style="text-align:right;color:#ffc800">{{ r.max_width_mm }}mm</td>
          <td style="text-align:right;color:#42a5f5">{{ r.area_mm2 }}</td>
          <td style="text-align:right">{{ r.segment_count }}</td>
        </tr>
        {% endfor %}
      </table>
      {% else %}
      <p class="no-data">Chưa có lịch sử.</p>
      {% endif %}
    </div>

    <!-- Segments -->
    {% set valid_segs = d.segments | selectattr("passes_width") | list %}
    {% if valid_segs %}
    <div class="card">
      <div class="card-label">Chi tiết từng đoạn ({{ valid_segs|length }} đoạn hợp lệ)</div>
      <table>
        <tr>
          <th>#</th>
          <th style="text-align:right">Dài (mm)</th>
          <th style="text-align:right">Rộng (mm)</th>
        </tr>
        {% for seg in valid_segs %}
        <tr>
          <td>{{ seg.index }}</td>
          <td style="text-align:right">{{ seg.length_mm }}</td>
          <td style="text-align:right;color:#ffc800">{{ seg.max_width_mm }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    {% endif %}

    <!-- Export -->
    <div class="card">
      <div class="card-label">Xuất dữ liệu</div>
      <a href="/export/csv" class="btn btn-blue">📥 CSV</a>
      <a href="/api" class="btn btn-green" target="_blank">📋 JSON</a>
    </div>
  </div>

  <!-- CỘT PHẢI -->
  <div>
    <div class="img-wrap">
      <div class="card-label">Kết quả – Overlay</div>
      <img src="/overlay" alt="overlay">
    </div>
    <div class="img-pair">
      <div class="img-wrap">
        <div class="card-label">Binary</div>
        <img src="/binary" alt="binary">
      </div>
      <div class="img-wrap">
        <div class="card-label">Skeleton</div>
        <img src="/skeleton" alt="skeleton">
      </div>
    </div>
    {% if d.input_validation %}
    {% set iv = d.input_validation %}
    <div class="card">
      <div class="card-label">Kiểm tra ảnh đầu vào</div>
      <div class="val-grid">
        {% if iv.background_median >= 115 %}
          <div class="val-ok">✓ Đủ sáng ({{ iv.background_median|round(0)|int }})</div>
        {% else %}
          <div class="val-err">✗ Thiếu sáng</div>
        {% endif %}
        {% if iv.laplacian_variance >= 0 %}
          <div class="val-ok">✓ Nét (Lap: {{ iv.laplacian_variance|round(1) }})</div>
        {% else %}
          <div class="val-err">✗ Bị mờ</div>
        {% endif %}
        {% if iv.extreme_exposure_ratio <= 0.35 %}
          <div class="val-ok">✓ Không cháy sáng</div>
        {% else %}
          <div class="val-err">✗ Cháy/thiếu sáng</div>
        {% endif %}
        {% if iv.dark_background_ratio <= 0.30 %}
          <div class="val-ok">✓ Nền sáng đạt</div>
        {% else %}
          <div class="val-err">✗ Nền quá tối</div>
        {% endif %}
      </div>
    </div>
    {% endif %}
  </div>

</div>
{% else %}
<div class="card">
  <p class="no-data">Chưa có dữ liệu. Nhấn "Chụp & Phân tích" để bắt đầu.</p>
</div>
{% endif %}

<script>
  document.getElementById('ts').textContent = new Date().toLocaleTimeString();
  setTimeout(() => location.reload(), 5000);
</script>
</body>
</html>"""


def draw_reference_overlay(image: np.ndarray, calibration):
    overlay = image.copy()
    if calibration is None:
        return overlay
    corners_int = np.asarray(calibration.reference_corners).astype(np.int32)
    cv2.polylines(overlay, [corners_int], True, (0, 255, 255), 2, cv2.LINE_AA)
    center = tuple(np.mean(corners_int, axis=0).astype(int))
    cv2.putText(overlay, f"Square: {calibration.pixels_per_mm:.2f} px/mm",
                (center[0] - 72, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _encode_jpg(img: np.ndarray, quality: int = 80) -> bytes | None:
    if img is None or (hasattr(img, "size") and img.size == 0):
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def update_display(img: np.ndarray):
    """Cập nhật frame hiển thị live stream."""
    jpg = _encode_jpg(img, quality=70)
    if jpg:
        with _shared["lock"]:
            _shared["display_jpg"] = jpg


def create_flask_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        with _shared["lock"]:
            payload = _shared["payload"]
        history = crack_db.get_history(limit=50)
        return render_template_string(HTML_PAGE, d=payload, history=history)

    @app.route("/api")
    def api():
        with _shared["lock"]:
            payload = _shared["payload"]
        if payload is None:
            return jsonify({"status": "no_data"}), 204
        return jsonify(payload)

    @app.route("/stream")
    def stream():
        """MJPEG live stream."""
        def generate():
            while True:
                with _shared["lock"]:
                    jpg = _shared["display_jpg"]
                if jpg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.05)   # ~20 fps
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/capture", methods=["POST"])
    def web_capture():
        with _shared["lock"]:
            _shared["command"] = "capture"
        return redirect("/")

    @app.route("/retake", methods=["POST"])
    def web_retake():
        with _shared["lock"]:
            _shared["command"] = "retake"
        return redirect("/")

    @app.route("/overlay")
    def overlay():
        with _shared["lock"]:
            jpg = _shared["overlay_jpg"]
        if jpg is None:
            return "", 204
        return send_file(io.BytesIO(jpg), mimetype="image/jpeg")

    @app.route("/binary")
    def binary_img():
        with _shared["lock"]:
            jpg = _shared["binary_jpg"]
        if jpg is None:
            return "", 204
        return send_file(io.BytesIO(jpg), mimetype="image/jpeg")

    @app.route("/skeleton")
    def skeleton_img():
        with _shared["lock"]:
            jpg = _shared["skeleton_jpg"]
        if jpg is None:
            return "", 204
        return send_file(io.BytesIO(jpg), mimetype="image/jpeg")

    @app.route("/history")
    def history_api():
        return jsonify(crack_db.get_history(limit=50))

    @app.route("/overlay/<int:record_id>")
    def overlay_record(record_id: int):
        jpg = crack_db.get_overlay_jpg(record_id)
        if jpg is None:
            return "", 404
        return send_file(io.BytesIO(jpg), mimetype="image/jpeg")

    @app.route("/export/csv")
    def export_csv_route():
        csv_content = crack_db.export_csv()
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = "attachment; filename=crack_history.csv"
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        return response

    return app


def start_web_server(port=5000):
    if not FLASK_OK:
        print("[WARN] Flask chua cai. Chay: pip install flask --break-system-packages")
        return
    app = create_flask_app()
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    print("[WEB] Truy cap: http://<IP_Pi>:" + str(port))


def update_web(analysis: dict, result_img: np.ndarray) -> None:
    if analysis is None:
        return
    overlay_jpg  = _encode_jpg(result_img, 88)
    binary_jpg   = _encode_jpg(analysis.get("binary"), 88)
    skeleton_jpg = _encode_jpg(analysis.get("skeleton"), 88)
    payload = {k: v for k, v in analysis["payload"].items() if k != "artifacts"}
    try:
        crack_db.save_measurement(payload, result_img)
    except Exception as exc:
        print(f"[WARN] DB save failed: {exc}")
    with _shared["lock"]:
        _shared["payload"]      = payload
        _shared["overlay_jpg"]  = overlay_jpg
        _shared["binary_jpg"]   = binary_jpg
        _shared["skeleton_jpg"] = skeleton_jpg


def apply_rectification(frame: np.ndarray, config):
    if config is None or frame is None:
        return frame
    rectified = frame
    if config.get("camera_matrix") is not None:
        h, w = rectified.shape[:2]
        # Tinh vung anh hop le sau undistort (khong bi meo goc)
        _, roi = cv2.getOptimalNewCameraMatrix(
            config["camera_matrix"], config["dist_coeffs"], (w, h), alpha=0
        )
        # Pad bang BORDER_REPLICATE de undistort khong tao vung den/gương o rìa
        PAD = 100
        padded = cv2.copyMakeBorder(rectified, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        cam_adj = config["camera_matrix"].copy()
        cam_adj[0, 2] += PAD
        cam_adj[1, 2] += PAD
        undistorted = cv2.undistort(padded, cam_adj, config["dist_coeffs"])
        full = undistorted[PAD:PAD + h, PAD:PAD + w]
        # Crop theo ROI de bo vung goc bi meo
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            rectified = full[y:y + rh, x:x + rw]
        else:
            rectified = full
    if config.get("homography") is not None:
        out_size = config.get("output_size") or (rectified.shape[1], rectified.shape[0])
        rectified = cv2.warpPerspective(rectified, config["homography"], out_size)
    return rectified


def open_usb_camera(device=0, width=1920, height=1080):
    candidates = [(width, height), (1280, 720), (640, 480)]
    for req_w, req_h in candidates:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Camera /dev/video{device} @ {actual_w}x{actual_h}")
            return cap
        cap.release()
    return None


def grab_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None


def resize_for_screen(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def draw_preview_hud(frame):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.putText(out, "PREVIEW -- [SPACE] Chup",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, "PREVIEW -- [SPACE] Chup",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OK, 1, cv2.LINE_AA)
    return out


def draw_result_hud(result_img, analysis, error):
    out = result_img.copy()
    h, w = out.shape[:2]
    font, fs, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 24
    if error:
        lines = [("ERR: " + error[:65], COLOR_ERR)]
    else:
        lines = [
            ("Total length: " + str(round(analysis["total_length_mm"], 2)) + " mm", COLOR_OK),
            ("Max width   : " + str(round(analysis["widest_segment_max_mm"], 2)) + " mm", COLOR_WARN),
        ]
    for i, (text, color) in enumerate(lines):
        y = 14 + 20 + i * lh
        cv2.putText(out, text, (14, y), font, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, text, (14, y), font, fs, color, th, cv2.LINE_AA)
    cv2.putText(out, "[R] Retake  [Q] Quit",
                (14, h - 10), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, "[R] Retake  [Q] Quit",
                (14, h - 10), font, 0.45, COLOR_WARN, 1, cv2.LINE_AA)
    return out


def draw_widest_highlight(result_img, analysis):
    out = result_img.copy()
    if not analysis:
        return out
    ws = analysis.get("widest_segment")
    if ws is None:
        return out
    mid_y, mid_x = ws.points[len(ws.points) // 2]
    cv2.circle(out, (mid_x, mid_y), 8, (0, 0, 0), -1)
    cv2.circle(out, (mid_x, mid_y), 6, (0, 0, 255), -1)
    label = "W=" + str(round(ws.max_width_mm, 2)) + "mm"
    lx, ly = mid_x + 10, mid_y - 4
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def main():
    os.chdir(Path(__file__).parent.resolve())
    crack_db.init_db()

    parser = argparse.ArgumentParser(description="Crack Measurement Pi4 - Web Display")
    parser.add_argument("--device",     type=int,  default=0)
    parser.add_argument("--width",      type=int,  default=1920)
    parser.add_argument("--height",     type=int,  default=1080)
    parser.add_argument("--port",       type=int,  default=5000)
    parser.add_argument("--headless",   action="store_true",
                        help="Khong hien thi cua so OpenCV (dung khi khong co man hinh)")
    args = parser.parse_args()

    rectify_config = {
        "camera_matrix": CHESSBOARD_CAMERA_MATRIX,
        "dist_coeffs":   CHESSBOARD_DIST_COEFFS,
        "homography":    None,
        "output_size":   None,
    }

    cap = open_usb_camera(args.device, args.width, args.height)
    if cap is None:
        for dev in range(1, 5):
            cap = open_usb_camera(dev, args.width, args.height)
            if cap is not None:
                break
    if cap is None:
        print("[ERROR] Khong mo duoc camera.")
        sys.exit(1)

    start_web_server(port=args.port)

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[INFO] Phim: SPACE=Chup  R=Chup_lai  Q=Thoat")
    print("[WEB]  Nut:  Chup & Phan tich | Chup lai")

    state = "preview"
    analysis = error_msg = result_img = None

    while True:
        # --- Doc lenh tu web ---
        with _shared["lock"]:
            cmd = _shared["command"]
            _shared["command"] = None

        # --- Phim ban phim (neu co man hinh) ---
        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")) or cmd == "retake":
                state = "preview"
                analysis = error_msg = result_img = None
                cmd = None
            if state == "preview" and (key in (ord(" "), 13) or cmd == "capture"):
                state = "capture"
        else:
            time.sleep(0.03)
            if cmd == "retake":
                state = "preview"
                analysis = error_msg = result_img = None
            elif cmd == "capture" and state == "preview":
                state = "capture"

        # --- PREVIEW ---
        if state == "preview":
            frame = grab_frame(cap)
            if frame is not None:
                frame = apply_rectification(frame, rectify_config)
                preview_display = resize_for_screen(draw_preview_hud(frame))
                update_display(preview_display)
                if not args.headless:
                    cv2.imshow(WINDOW_NAME, preview_display)
            continue

        # --- CAPTURE ---
        if state == "capture":
            frame = grab_frame(cap)
            if frame is None:
                error_msg  = "Khong lay duoc anh tu camera."
                result_img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = apply_rectification(frame, rectify_config)

                # Hien thi loading
                proc = frame.copy()
                hp, wp = proc.shape[:2]
                cv2.rectangle(proc, (0, hp // 2 - 40), (wp, hp // 2 + 40), (20, 20, 20), -1)
                cv2.putText(proc, "Dang phan tich, vui long cho...",
                            (wp // 2 - 260, hp // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WARN, 2, cv2.LINE_AA)
                loading_display = resize_for_screen(proc)
                update_display(loading_display)
                if not args.headless:
                    cv2.imshow(WINDOW_NAME, loading_display)
                    cv2.waitKey(1)

                print("[INFO] Dang phan tich...")
                try:
                    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5,
                                             interpolation=cv2.INTER_AREA)
                    analysis  = analyze_image_from_array(frame_small,
                                                         image_name="camera_capture.jpg")
                    error_msg = None
                    print("[OK] Tong chieu dai: " + str(round(analysis["total_length_mm"], 2)) + " mm")
                    print("[OK] Rong lon nhat : " + str(round(analysis["widest_segment_max_mm"], 2)) + " mm")
                    result_img = draw_widest_highlight(analysis["overlay"], analysis)
                    result_img = draw_reference_overlay(result_img, analysis.get("calibration"))
                    update_web(analysis, result_img)
                except ValueError as exc:
                    print("[WARN] " + str(exc))
                    error_msg  = str(exc)
                    result_img = frame.copy()
                    analysis   = None
            state = "result"

        # --- RESULT ---
        if state == "result" and result_img is not None:
            display = draw_result_hud(result_img, analysis, error_msg)
            result_display = resize_for_screen(display)
            update_display(result_display)
            if not args.headless:
                cv2.imshow(WINDOW_NAME, result_display)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("[INFO] Da thoat.")


if __name__ == "__main__":
    main()

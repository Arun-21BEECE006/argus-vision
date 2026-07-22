"""
Argus Vision - Flask backend
Fixed: _detect_small scale factor bug removed entirely.
Fixed: yolo11s-safe video processing (320p resize, 60 frames max).
"""

import gc
import json
import os
import time
import uuid
import traceback

import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, Response)

from detector import detector, DEFAULT_CONF

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VID = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ── Pages ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model=detector.model_name,
                           n_classes=len(detector.names))

@app.route("/image")
def page_image():
    return render_template("image.html")

@app.route("/video")
def page_video():
    return render_template("video.html")

@app.route("/webcam")
def page_webcam():
    return render_template("webcam.html")

@app.route("/rtsp")
def page_rtsp():
    return render_template("rtsp.html")


# ── Image detection ────────────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No image uploaded."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_IMG:
            return jsonify({"error": f"Unsupported type: {ext}"}), 400

        conf = float(request.form.get("conf", DEFAULT_CONF))
        data = np.frombuffer(file.read(), np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not read image."}), 400

        annotated, summary, details = detector.detect(img, conf=conf)
        gc.collect()

        out_name = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), annotated)

        return jsonify({
            "output":  f"/outputs/{out_name}",
            "summary": summary,
            "total":   sum(summary.values()),
            "details": details,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Video detection ────────────────────────────────────────────────────
def _fit_to(frame, max_dim):
    """Resize frame so longest edge = max_dim, keeping aspect ratio."""
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame
    s = max_dim / max(w, h)
    return cv2.resize(frame, (int(w * s), int(h * s)))


@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    in_path  = None
    out_path = None
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No video uploaded."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_VID:
            return jsonify({"error": f"Unsupported type: {ext}"}), 400

        conf    = float(request.form.get("conf", DEFAULT_CONF))
        in_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        file.save(in_path)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video."}), 400

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── 320p max for video — keeps yolo11s under 512MB RAM ────────
        MAX_DIM = 320
        scale   = min(1.0, MAX_DIM / max(orig_w, orig_h))
        out_w   = int(orig_w * scale)
        out_h   = int(orig_h * scale)

        out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (out_w, out_h))

        if not writer.isOpened():
            cap.release()
            return jsonify({"error": "VideoWriter failed."}), 500

        from collections import Counter
        totals  = Counter()
        written = 0
        MAX_FRAMES = 60      # ~2 sec at 30fps — safe for yolo11s RAM

        while written < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize keeping correct aspect ratio — no squishing!
            frame = cv2.resize(frame, (out_w, out_h))

            # Direct detection — no separate small-frame logic needed
            annotated, summary, _ = detector.detect(frame, conf=conf)

            for k, v in summary.items():
                totals[k] = max(totals[k], v)

            writer.write(annotated)
            written += 1

            # Aggressive memory cleanup for yolo11s
            if written % 5 == 0:
                gc.collect()

        cap.release()
        writer.release()
        gc.collect()

        _cleanup(in_path); in_path = None

        if not os.path.exists(out_path) or written == 0:
            return jsonify({"error": "Output video empty."}), 500

        final = _try_h264(out_path)

        return jsonify({
            "output":  f"/outputs/{os.path.basename(final)}",
            "fps":     round(fps, 2),
            "summary": dict(totals),
            "total":   sum(totals.values()),
            "frames":  written,
        })

    except Exception:
        _cleanup(in_path); _cleanup(out_path)
        return jsonify({"error": traceback.format_exc()}), 500


# ── Single-frame detection (browser webcam) ────────────────────────────
@app.route("/api/detect/frame", methods=["POST"])
def api_detect_frame():
    try:
        conf  = float(request.args.get("conf", DEFAULT_CONF))
        data  = np.frombuffer(request.get_data(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return "Bad frame", 400

        # Keep aspect ratio — critical for correct detection
        frame = _fit_to(frame, max_dim=480)

        annotated, summary, _ = detector.detect(frame, conf=conf)
        gc.collect()

        _, buf = cv2.imencode(".jpg", annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        resp = Response(buf.tobytes(), mimetype="image/jpeg")
        resp.headers["X-Summary"]                     = json.dumps(summary)
        resp.headers["Access-Control-Expose-Headers"] = "X-Summary"
        return resp
    except Exception:
        return "Error", 500


# ── Helpers ────────────────────────────────────────────────────────────
def _cleanup(path):
    if path and os.path.exists(path):
        try: os.remove(path)
        except OSError: pass


def _try_h264(src):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True, capture_output=True, timeout=60)
        if os.path.exists(dst):
            _cleanup(src); return dst
    except Exception:
        pass
    return src


# ── Live streaming (RTSP) ──────────────────────────────────────────────
def _open_source(source):
    if source == "webcam": return cv2.VideoCapture(0)
    if source.isdigit():   return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _mjpeg(source, conf):
    cap = _open_source(source)
    if not cap.isOpened():
        err = np.full((360, 640, 3), 24, np.uint8)
        cv2.putText(err, "Cannot open source", (60, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 240), 2)
        _, buf = cv2.imencode(".jpg", err)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05); continue
            annotated = detector.annotate_frame(frame, conf=conf)
            ok, buf   = cv2.imencode(".jpg", annotated,
                                     [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/api/stream")
def api_stream():
    source = request.args.get("source", "webcam")
    conf   = float(request.args.get("conf", DEFAULT_CONF))
    return Response(_mjpeg(source, conf),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
"""
Argus Vision - AI Object Detection Studio
Flask backend: image / video / live-webcam / RTSP detection.
"""

import gc
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
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload


# ── Pages ────────────────────────────────────────────────────────────────────
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


# ── Image detection ──────────────────────────────────────────────────────────
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


# ── Video detection ──────────────────────────────────────────────────────────
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

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize to max 480p — critical for free-tier RAM + speed
        if max(w, h) > 480:
            s = 480 / max(w, h)
            w = int(w * s)
            h = int(h * s)

        out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if not writer.isOpened():
            cap.release()
            return jsonify({"error": "VideoWriter failed to open."}), 500

        from collections import Counter
        totals      = Counter()
        frame_idx   = 0
        written     = 0
        MAX_FRAMES  = 150        # max frames to detect (~5 sec at 30fps)
        SKIP        = 2          # run detection every 2nd frame, copy others
        last_ann    = None

        while written < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize frame to our output size
            frame = cv2.resize(frame, (w, h))

            if frame_idx % SKIP == 0:
                last_ann, summary, _ = detector.detect(frame, conf=conf)
                for k, v in summary.items():
                    totals[k] = max(totals[k], v)
                if frame_idx % 20 == 0:   # free memory periodically
                    gc.collect()
            else:
                if last_ann is None:
                    last_ann = frame

            writer.write(last_ann)
            frame_idx += 1
            written   += 1

        cap.release()
        writer.release()
        gc.collect()

        _cleanup(in_path)
        in_path = None

        if not os.path.exists(out_path) or written == 0:
            return jsonify({"error": "Output video empty."}), 500

        # Try H.264 re-encode for browser playback
        final_path = _try_h264(out_path)
        final_name = os.path.basename(final_path)

        return jsonify({
            "output":  f"/outputs/{final_name}",
            "fps":     round(fps, 2),
            "summary": dict(totals),
            "total":   sum(totals.values()),
            "frames":  written,
        })

    except Exception:
        _cleanup(in_path)
        _cleanup(out_path)
        return jsonify({"error": traceback.format_exc()}), 500


def _cleanup(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


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
            check=True, capture_output=True, timeout=120)
        if os.path.exists(dst):
            _cleanup(src)
            return dst
    except Exception:
        pass
    return src


# ── Live streaming ────────────────────────────────────────────────────────────
def _open_source(source):
    if source == "webcam":  return cv2.VideoCapture(0)
    if source.isdigit():    return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _mjpeg(source, conf):
    cap = _open_source(source)
    if not cap.isOpened():
        err = np.full((360, 640, 3), 24, np.uint8)
        cv2.putText(err, "Cannot open source", (60, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 240), 2)
        _, buf = cv2.imencode(".jpg", err)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            annotated  = detector.annotate_frame(frame, conf=conf)
            ok, buf    = cv2.imencode(".jpg", annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
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
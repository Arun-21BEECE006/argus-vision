"""
Argus Vision - AI Object Detection Studio
Flask backend: image / video / live-webcam / RTSP detection.
"""

import gc
import os
import time
import uuid

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
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


# ── Pages ───────────────────────────────────────────────────────────────────
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


# ── Image detection ─────────────────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No image uploaded."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_IMG:
            return jsonify({"error": f"Unsupported image type: {ext}"}), 400

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
    except Exception as e:
        return jsonify({"error": f"Image detection failed: {str(e)}"}), 500


# ── Video detection ─────────────────────────────────────────────────────────
@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    in_path = None
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No video uploaded."}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_VID:
            return jsonify({"error": f"Unsupported video type: {ext}"}), 400

        conf = float(request.form.get("conf", DEFAULT_CONF))

        in_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        file.save(in_path)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file."}), 400

        fps    = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:
            fps = 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # ── Resize large videos to max 720p for faster inference ──────────
        max_dim = 720
        if max(width, height) > max_dim:
            scale  = max_dim / max(width, height)
            width  = int(width  * scale)
            height = int(height * scale)

        # ── For long videos, skip frames so processing finishes in time ───
        # Target: process at most 300 frames (≈ 30s at 10fps equivalent)
        MAX_PROCESS_FRAMES = 300
        frame_skip = max(1, total // MAX_PROCESS_FRAMES) if total > 0 else 1

        output_name = f"{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Try mp4v first, fallback to MJPG if it fails
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            output_name = f"{uuid.uuid4().hex}.avi"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            return jsonify({"error": "Could not create output video."}), 500

        from collections import Counter
        totals      = Counter()
        frame_count = 0
        last_ann    = None   # reuse last annotated frame for skipped frames

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize frame if needed
            fh, fw = frame.shape[:2]
            if max(fw, fh) > max_dim:
                frame = cv2.resize(frame, (width, height))

            if frame_count % frame_skip == 0:
                # Run detection on this frame
                last_ann, summary, _ = detector.detect(frame, conf=conf)
                for k, v in summary.items():
                    totals[k] = max(totals[k], v)
            else:
                # Reuse last annotated frame for speed
                last_ann = frame if last_ann is None else last_ann

            writer.write(last_ann)
            frame_count += 1

        cap.release()
        writer.release()
        gc.collect()

        try:
            os.remove(in_path)
            in_path = None
        except OSError:
            pass

        if not os.path.exists(output_path) or frame_count == 0:
            return jsonify({"error": "Output video not generated."}), 500

        # Try H.264 conversion for browser compatibility
        playable_path = _try_h264(output_path)
        playable_name = os.path.basename(playable_path)

        return jsonify({
            "output":  f"/outputs/{playable_name}",
            "fps":     round(fps, 3),
            "summary": dict(totals),
            "total":   sum(totals.values()),
            "frames":  frame_count,
        })

    except Exception as e:
        if in_path and os.path.exists(in_path):
            try:
                os.remove(in_path)
            except OSError:
                pass
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500


def _try_h264(input_video):
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        return input_video
    output_video = input_video.rsplit(".", 1)[0] + "_h264.mp4"
    command = ["ffmpeg", "-y", "-i", input_video,
               "-c:v", "libx264", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               output_video]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        if os.path.exists(output_video):
            os.remove(input_video)
            return output_video
        return input_video
    except Exception:
        return input_video


# ── Live streaming ───────────────────────────────────────────────────────────
def _open_source(source: str):
    if source == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _mjpeg(source: str, conf: float):
    cap = _open_source(source)
    if not cap.isOpened():
        err = np.full((360, 640, 3), 24, np.uint8)
        cv2.putText(err, "Cannot open source", (60, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 240), 2)
        ok, buf = cv2.imencode(".jpg", err)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            annotated = detector.annotate_frame(frame, conf=conf)
            ok, buf   = cv2.imencode(".jpg", annotated,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
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


# ── Serve outputs ───────────────────────────────────────────────────────────
@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    print("=" * 60)
    print(" Argus Vision  |  model:", detector.model_name)
    print(" Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
"""
Argus Vision - AI Object Detection Studio
Flask backend: image / video / live-webcam / RTSP detection.
"""

import os
import time
import uuid
import threading
import gc
import torch
import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, Response, abort)

from detector import detector, DEFAULT_CONF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VID = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads


# --------------------------------------------------------------------------- #
#  Pages                                                                       #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Image detection                                                             #
# --------------------------------------------------------------------------- #
@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMG:
        return jsonify({"error": f"Unsupported image type: {ext}"}), 400

    conf = float(request.form.get("conf", DEFAULT_CONF))

    # --- AFTER (replace with this) ---
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image."}), 400

    # Resize large images before inference to save memory
    MAX_DIM = 640
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    annotated, summary, details = detector.detect(img, conf=conf)
    gc.collect()   # ← free memory after every detection
    out_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), annotated)
    free_memory()
    return jsonify({
        "output": f"/outputs/{out_name}",
        "summary": summary,
        "total": sum(summary.values()),
        "details": details,
    })

def _try_h264(input_video):
    """
    Re-encode with ffmpeg to H.264/yuv420p so the file plays inline in every
    browser. OpenCV's own mp4v writer produces MPEG-4 Part 2, which Chrome,
    Safari and most browsers refuse to play back in a <video> tag.
    Falls back to the original file (still downloadable) if ffmpeg is missing
    or the conversion fails.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print("[Argus] ffmpeg not found on PATH — serving raw mp4v file. "
              "Install ffmpeg for guaranteed in-browser playback.")
        return input_video

    output_video = input_video.replace(".mp4", "_h264.mp4")

    command = [
        "ffmpeg", "-y", "-i", input_video,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        if os.path.exists(output_video):
            os.remove(input_video)
            return output_video
        return input_video
    except Exception as e:
        print("[Argus] ffmpeg conversion failed:", e)
        return input_video


# --------------------------------------------------------------------------- #
#  Video detection — writes a real annotated .mp4 and serves it as a file     #
#  so the front-end can play it back in a normal <video> element.             #
# --------------------------------------------------------------------------- #
@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
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
        return jsonify({"error": "Could not open video."}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # also guards against NaN
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return jsonify({"error": "Failed to create output video."}), 500

    from collections import Counter
    totals = Counter()
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, summary, _ = detector.detect(frame, conf=conf)
        writer.write(annotated)
        frame_count += 1
        for k, v in summary.items():
            totals[k] = max(totals[k], v)  # peak simultaneous count per class

    cap.release()
    writer.release()

    try:
        os.remove(in_path)
    except OSError:
        pass

    if not os.path.exists(output_path) or frame_count == 0:
        return jsonify({"error": "Output video not generated."}), 500

    playable_path = _try_h264(output_path)
    playable_name = os.path.basename(playable_path)
    free_memory()
    return jsonify({
        "output": f"/outputs/{playable_name}",
        "fps": round(fps, 3),
        "summary": dict(totals),
        "total": sum(totals.values()),
        "frames": frame_count,
    })


# --------------------------------------------------------------------------- #
#  Live streaming (webcam + RTSP) via MJPEG                                    #
# --------------------------------------------------------------------------- #
def _open_source(source: str):
    """Resolve 'webcam' -> device 0, otherwise treat as RTSP/HTTP/file URL."""
    if source == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)  # rtsp:// or http:// url


def _mjpeg(source: str, conf: float):
    cap = _open_source(source)
    if not cap.isOpened():
        # Emit a single error frame so the browser shows a message.
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
            ok, buf = cv2.imencode(".jpg", annotated,
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
    conf = float(request.args.get("conf", DEFAULT_CONF))
    return Response(_mjpeg(source, conf),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/detect/frame", methods=["POST"])
def api_detect_frame():
    """
    Receives a single JPEG frame from the user's browser webcam,
    runs detection, and returns the annotated frame as JPEG.
    Used by the frontend webcam page in production.
    """
    data = request.get_data()
    if not data:
        return "No frame received", 400
    
    conf = float(request.args.get("conf", DEFAULT_CONF))
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid frame", 400
    
    annotated = detector.annotate_frame(frame, conf=conf)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")
# --------------------------------------------------------------------------- #
#  Serve annotated outputs                                                     #
# --------------------------------------------------------------------------- #
@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)

def free_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 60)
    print(" Argus Vision  |  model:", detector.model_name)
    print(" Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
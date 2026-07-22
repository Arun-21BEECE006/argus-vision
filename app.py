"""
Argus Vision — Gradio edition
Runs on Hugging Face Spaces (free, 16 GB RAM)
detector.py is unchanged — only this file replaces the Flask app.py
"""

import gc
import tempfile

import cv2
import numpy as np
import gradio as gr

from detector import detector, DEFAULT_CONF


# ------------------------------------------------------------------ #
#  Detection functions                                                 #
# ------------------------------------------------------------------ #

def detect_image(image, conf):
    """Image upload → annotated image + summary text."""
    if image is None:
        return None, "Please upload an image first."

    # Gradio gives RGB numpy array → convert to BGR for OpenCV / YOLO
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize very large images to save memory
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    annotated_bgr, summary, details = detector.detect(img_bgr, conf=conf)
    gc.collect()

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    total = sum(summary.values())
    lines = [f"Total objects detected: {total}", ""]
    for label, count in sorted(summary.items(), key=lambda x: -x[1]):
        lines.append(f"  {label}: {count}")
    if details:
        lines.append("")
        lines.append("All detections:")
        for d in details:
            lines.append(f"  {d['label']}  conf={d['conf']}  box={d['box']}")

    return annotated_rgb, "\n".join(lines)


def detect_video(video_path, conf):
    """Video upload → fully annotated video file."""
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:   # guard NaN / 0
        fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, _, _ = detector.detect(frame, conf=conf)
        writer.write(annotated)

    cap.release()
    writer.release()
    gc.collect()
    return out_path


def detect_webcam_frame(image, conf):
    """
    Called on every webcam frame (streaming=True).
    Gradio sends RGB numpy array, returns RGB numpy array.
    """
    if image is None:
        return None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = detector.annotate_frame(img_bgr, conf=conf)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ #
#  Gradio UI                                                           #
# ------------------------------------------------------------------ #

css = """
#header { text-align: center; margin-bottom: 8px; }
footer  { display: none !important; }
"""

with gr.Blocks(title="Argus Vision", css=css) as demo:

    gr.Markdown(
        f"""
# 👁 Argus Vision
**Real-time AI Object Detection** &nbsp;·&nbsp; YOLO11 &nbsp;·&nbsp;
{len(detector.names)} COCO classes
        """,
        elem_id="header"
    )

    # ── Image tab ──────────────────────────────────────────────────
    with gr.Tab("🖼️ Image"):
        with gr.Row():
            img_in = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            img_out = gr.Image(label="Detection Result")

        conf_img = gr.Slider(
            0.05, 0.95, value=DEFAULT_CONF, step=0.05,
            label="Confidence Threshold"
        )
        btn_img = gr.Button("Detect Objects", variant="primary", size="lg")
        summary_img = gr.Textbox(
            label="Detection Summary",
            lines=8,
            interactive=False,
        )

        btn_img.click(
            detect_image,
            inputs=[img_in, conf_img],
            outputs=[img_out, summary_img],
        )

    # ── Video tab ──────────────────────────────────────────────────
    with gr.Tab("🎬 Video"):
        gr.Markdown("Upload a video — every frame is annotated and returned as a playable file.")
        with gr.Row():
            vid_in = gr.Video(label="Upload Video", sources=["upload"])
            vid_out = gr.Video(label="Annotated Video")

        conf_vid = gr.Slider(
            0.05, 0.95, value=DEFAULT_CONF, step=0.05,
            label="Confidence Threshold"
        )
        btn_vid = gr.Button("Process Video", variant="primary", size="lg")

        btn_vid.click(
            detect_video,
            inputs=[vid_in, conf_vid],
            outputs=[vid_out],
        )

    # ── Live Camera tab ────────────────────────────────────────────
    with gr.Tab("📷 Live Camera"):
        gr.Markdown(
            "Uses **your browser's camera** — frames are sent to the server "
            "for detection and streamed back in real time."
        )
        with gr.Row():
            cam_in = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Camera Feed",
                mirror_webcam=False,
            )
            cam_out = gr.Image(label="Live Detection")

        conf_cam = gr.Slider(
            0.05, 0.95, value=DEFAULT_CONF, step=0.05,
            label="Confidence Threshold"
        )

        cam_in.stream(
            detect_webcam_frame,
            inputs=[cam_in, conf_cam],
            outputs=[cam_out],
            time_limit=600,         # keep stream alive up to 10 min
            stream_every=0.1,       # process ~10 fps
        )

    # ── RTSP tab ───────────────────────────────────────────────────
    with gr.Tab("📡 RTSP"):
        gr.Markdown(
            "**RTSP live stream detection.**\n\n"
            "Enter a publicly accessible RTSP URL. "
            "Private-network cameras (192.168.x.x) are not reachable from the cloud."
        )
        rtsp_url = gr.Textbox(
            label="RTSP URL",
            placeholder="rtsp://your-camera-ip:554/stream",
        )
        conf_rtsp = gr.Slider(
            0.05, 0.95, value=DEFAULT_CONF, step=0.05,
            label="Confidence Threshold"
        )
        rtsp_out = gr.Image(label="Stream Frame")
        rtsp_btn = gr.Button("Grab Frame", variant="primary")

        def grab_rtsp_frame(url, conf):
            if not url.strip():
                return None
            cap = cv2.VideoCapture(url.strip())
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            annotated = detector.annotate_frame(frame, conf=conf)
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        rtsp_btn.click(
            grab_rtsp_frame,
            inputs=[rtsp_url, conf_rtsp],
            outputs=[rtsp_out],
        )

    gr.Markdown(
        f"Model: `{detector.model_name}` &nbsp;·&nbsp; "
        "Argus Vision · YOLO11 Object Detection",
        elem_id="header"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
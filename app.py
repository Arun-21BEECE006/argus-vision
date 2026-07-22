"""
Argus Vision — Gradio edition with original custom design
Dark theme: #0b0f1a background, #14b8a6 teal, #f59e0b gold
Runs on Hugging Face Spaces free tier (16GB RAM)
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
    if image is None:
        return None, "Please upload an image first."
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        for d in details:
            lines.append(f"  {d['label']}  conf={d['conf']}  box={d['box']}")
    return annotated_rgb, "\n".join(lines)


def detect_video(video_path, conf):
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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
    if image is None:
        return None
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = detector.annotate_frame(img_bgr, conf=conf)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def grab_rtsp_frame(url, conf):
    if not url or not url.strip():
        return None
    cap = cv2.VideoCapture(url.strip())
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    annotated = detector.annotate_frame(frame, conf=conf)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ #
#  Custom CSS — your original Argus Vision dark theme                  #
# ------------------------------------------------------------------ #

ARGUS_CSS = """
body, .gradio-container, .main {
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
footer, .footer, #share-btn-container { display: none !important; }

.tab-nav {
    background: #0d1520 !important;
    border-bottom: 1px solid #1e2d3d !important;
    padding: 0 8px !important;
}
.tab-nav button {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    border-radius: 0 !important;
}
.tab-nav button:hover { color: #14b8a6 !important; }
.tab-nav button.selected {
    color: #14b8a6 !important;
    border-bottom: 2px solid #14b8a6 !important;
    background: transparent !important;
}

.block, .form, .panel {
    background: #0d1520 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 8px !important;
}

label > span, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

input[type="text"], input[type="number"], textarea {
    background: #0a1018 !important;
    border: 1px solid #1e2d3d !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20,184,166,0.15) !important;
    outline: none !important;
}

input[type="range"] { accent-color: #14b8a6 !important; }
.slider-container .head span { color: #f59e0b !important; font-weight: 600 !important; }

button.primary {
    background: #f59e0b !important;
    color: #0b0f1a !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 12px 28px !important;
    border-radius: 6px !important;
}
button.primary:hover { background: #d97706 !important; }

button.secondary {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
button.secondary:hover { border-color: #14b8a6 !important; color: #14b8a6 !important; }

.image-container, .upload-container, [data-testid="image"] {
    background: #0a1018 !important;
    border: 1px solid #14b8a6 !important;
    border-radius: 8px !important;
}
.upload-container svg { stroke: #334155 !important; }

textarea[readonly], .textbox {
    background: #0a1018 !important;
    border: 1px solid #1e2d3d !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

.prose p, .md p { color: #94a3b8 !important; font-size: 14px !important; }
.progress-bar { background: #14b8a6 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #14b8a6; }
"""


# ------------------------------------------------------------------ #
#  HTML blocks                                                         #
# ------------------------------------------------------------------ #

LOGO_SVG = """<svg width="36" height="36" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <rect width="32" height="32" rx="6" fill="#0b0f1a"/>
  <path d="M3,16 C5,9 11,7 16,7 C21,7 27,9 29,16 C27,23 21,25 16,25 C11,25 5,23 3,16 Z" fill="#0b1a24" stroke="#14b8a6" stroke-width="1.5"/>
  <circle cx="16" cy="16" r="5.5" fill="#0d9488"/>
  <circle cx="16" cy="16" r="3" fill="#0b0f1a"/>
  <circle cx="17.5" cy="14.5" r="1" fill="white" opacity="0.85"/>
  <path d="M3,7 L3,3 L7,3" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M25,3 L29,3 L29,7" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M3,25 L3,29 L7,29" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M29,25 L29,29 L25,29" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

NAVBAR = f"""<nav style="background:#0d1520;border-bottom:1px solid #1e2d3d;padding:14px 32px;display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:14px;">
    {LOGO_SVG}
    <span style="font-size:16px;font-weight:700;letter-spacing:0.12em;color:#e2e8f0;">ARGUS <span style="color:#f59e0b;">VISION</span></span>
  </div>
  <div style="display:flex;gap:32px;">
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">IMAGE</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">VIDEO</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">LIVE CAMERA</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">RTSP</span>
  </div>
</nav>"""

FOOTER = f"""<div style="background:#0d1520;border-top:1px solid #1e2d3d;padding:14px 32px;display:flex;align-items:center;gap:10px;margin-top:8px;">
  {LOGO_SVG}
  <span style="color:#f59e0b;font-size:12px;font-weight:700;letter-spacing:0.1em;">ARGUS VISION</span>
  <span style="color:#1e2d3d;">·</span>
  <span style="color:#475569;font-size:12px;">Real-time object detection · YOLO11 · {len(detector.names)} COCO classes</span>
</div>"""


def heading(title, sub=""):
    s = f'<p style="color:#64748b;font-size:13px;margin:6px 0 0;">{sub}</p>' if sub else ""
    return f"""<div style="padding:24px 4px 12px;">
      <h2 style="color:#e2e8f0;font-size:20px;font-weight:700;letter-spacing:0.08em;margin:0;
                 text-transform:uppercase;border-bottom:1px solid #14b8a6;padding-bottom:10px;">{title}</h2>{s}
    </div>"""


# ------------------------------------------------------------------ #
#  Gradio UI                                                           #
# ------------------------------------------------------------------ #

with gr.Blocks(title="Argus Vision", css=ARGUS_CSS) as demo:

    gr.HTML(NAVBAR)

    with gr.Tab("🖼️ Image"):
        gr.HTML(heading("Image Detection",
            "Upload an image — bounding boxes and labels are drawn on every detected object."))
        with gr.Row():
            img_in  = gr.Image(label="Upload Image", type="numpy",
                               sources=["upload", "clipboard"])
            img_out = gr.Image(label="Detection Result")
        conf_img    = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                                label="Confidence Threshold")
        btn_img     = gr.Button("⚡ Detect Objects", variant="primary", size="lg")
        summary_img = gr.Textbox(label="Detection Summary", lines=7, interactive=False)
        btn_img.click(detect_image, inputs=[img_in, conf_img],
                      outputs=[img_out, summary_img])

    with gr.Tab("🎬 Video"):
        gr.HTML(heading("Video Detection",
            "Upload a video — every frame is annotated and returned as a playable file."))
        with gr.Row():
            vid_in  = gr.Video(label="Upload Video", sources=["upload"])
            vid_out = gr.Video(label="Annotated Video")
        conf_vid = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                             label="Confidence Threshold")
        btn_vid  = gr.Button("⚡ Process Video", variant="primary", size="lg")
        btn_vid.click(detect_video, inputs=[vid_in, conf_vid], outputs=[vid_out])

    with gr.Tab("📷 Live Camera"):
        gr.HTML(heading("Live Camera Detection",
            "Stream from your browser camera. Boxes and a live object count on every frame."))
        with gr.Row():
            cam_in  = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                               label="Camera Feed", mirror_webcam=False)
            cam_out = gr.Image(label="Live Detection")
        conf_cam = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                             label="Confidence Threshold")
        cam_in.stream(detect_webcam_frame, inputs=[cam_in, conf_cam],
                      outputs=[cam_out], time_limit=600, stream_every=0.1)

    with gr.Tab("📡 RTSP"):
        gr.HTML(heading("RTSP Stream",
            "Enter a publicly accessible RTSP URL. Private-network cameras (192.168.x.x) not reachable from cloud."))
        rtsp_url  = gr.Textbox(label="RTSP URL",
                               placeholder="rtsp://your-camera-ip:554/stream")
        conf_rtsp = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                              label="Confidence Threshold")
        rtsp_btn  = gr.Button("⚡ Grab Frame", variant="primary")
        rtsp_out  = gr.Image(label="Stream Frame")
        rtsp_btn.click(grab_rtsp_frame, inputs=[rtsp_url, conf_rtsp], outputs=[rtsp_out])

    gr.HTML(FOOTER)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
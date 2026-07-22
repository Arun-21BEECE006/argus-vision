"""
Argus Vision — Streamlit edition
Runs on Streamlit Community Cloud (free, 1GB RAM)
detector.py is completely unchanged.
"""

import gc
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from detector import detector, DEFAULT_CONF

# ------------------------------------------------------------------ #
#  Page config — must be first Streamlit call                          #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Argus Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
#  Custom CSS — keeps your original dark teal/gold design              #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

.argus-nav {
    background: #0d1520;
    border-bottom: 1px solid #1e2d3d;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}
.argus-footer {
    background: #0d1520;
    border-top: 1px solid #1e2d3d;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 24px;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #e2e8f0;
    border-bottom: 1px solid #14b8a6;
    padding-bottom: 10px;
    margin: 8px 0 14px;
}
.section-sub { color: #64748b; font-size: 13px; margin-bottom: 16px; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d1520 !important;
    border-bottom: 1px solid #1e2d3d !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 22px !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #14b8a6 !important; }
.stTabs [aria-selected="true"] {
    color: #14b8a6 !important;
    border-bottom: 2px solid #14b8a6 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 20px 0 !important; }

[data-testid="stFileUploader"] {
    border: 1px solid #14b8a6 !important;
    border-radius: 8px !important;
    background: #0a1018 !important;
}
[data-testid="stCameraInput"] section {
    border: 1px solid #14b8a6 !important;
    border-radius: 8px !important;
    background: #0a1018 !important;
}
[data-testid="stTextInput"] input {
    background: #0a1018 !important;
    border: 1px solid #1e2d3d !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 0 2px rgba(20,184,166,0.15) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #14b8a6 !important;
    border-color: #14b8a6 !important;
}
.stSlider label { color: #94a3b8 !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important; }

button[kind="primary"] {
    background: #f59e0b !important;
    color: #0b0f1a !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
}
button[kind="primary"]:hover { background: #d97706 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #14b8a6; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Shared HTML fragments                                               #
# ------------------------------------------------------------------ #
LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <rect width="32" height="32" rx="6" fill="#0b0f1a"/>
  <path d="M3,16 C5,9 11,7 16,7 C21,7 27,9 29,16 C27,23 21,25 16,25 C11,25 5,23 3,16 Z"
        fill="#0b1a24" stroke="#14b8a6" stroke-width="1.5"/>
  <circle cx="16" cy="16" r="5.5" fill="#0d9488"/>
  <circle cx="16" cy="16" r="3" fill="#0b0f1a"/>
  <circle cx="17.5" cy="14.5" r="1" fill="white" opacity="0.85"/>
  <path d="M3,7 L3,3 L7,3"    fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M25,3 L29,3 L29,7" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M3,25 L3,29 L7,29" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M29,25 L29,29 L25,29" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

# Navbar
st.markdown(f"""
<nav class="argus-nav">
  <div style="display:flex;align-items:center;gap:14px;">
    {LOGO_SVG}
    <span style="font-size:16px;font-weight:700;letter-spacing:0.12em;color:#e2e8f0;">
      ARGUS <span style="color:#f59e0b;">VISION</span>
    </span>
  </div>
  <div style="display:flex;gap:32px;">
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">IMAGE</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">VIDEO</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">LIVE CAMERA</span>
    <span style="font-size:12px;letter-spacing:0.08em;color:#94a3b8;">RTSP</span>
  </div>
</nav>
""", unsafe_allow_html=True)


def heading(title, sub=""):
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(
        f'<div class="section-title">{title}</div>{sub_html}',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------ #
#  Tabs                                                                #
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️  Image",
    "🎬  Video",
    "📷  Live Camera",
    "📡  RTSP",
])

# ── IMAGE ──────────────────────────────────────────────────────────────────
with tab1:
    heading("Image Detection",
            "Upload an image — bounding boxes and labels are drawn on every detected object.")
    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_up"
        )
        conf_img = st.slider("Confidence Threshold", 0.05, 0.95,
                             DEFAULT_CONF, 0.05, key="img_conf")
        btn_img  = st.button("⚡ Detect Objects", key="img_btn",
                             type="primary", use_container_width=True)

    with col2:
        out_img     = st.empty()
        out_summary = st.empty()

    if uploaded:
        raw   = np.frombuffer(uploaded.read(), np.uint8)
        img   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w  = img.shape[:2]
        if max(h, w) > 1280:
            s = 1280 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))

        if btn_img:
            with st.spinner("Detecting objects..."):
                annotated, summary, details = detector.detect(img, conf=conf_img)
                gc.collect()
            out_img.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          caption="Detection Result", use_container_width=True)
            total = sum(summary.values())
            lines = [f"**Total: {total} objects detected**", ""]
            for lbl, cnt in sorted(summary.items(), key=lambda x: -x[1]):
                lines.append(f"- {lbl}: {cnt}")
            out_summary.markdown("\n".join(lines))
        else:
            out_img.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                          caption="Uploaded Image", use_container_width=True)

# ── VIDEO ──────────────────────────────────────────────────────────────────
with tab2:
    heading("Video Detection",
            "Upload a video — every frame is annotated and returned as a playable file.")
    vid_up   = st.file_uploader("Upload Video",
                                type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    conf_vid = st.slider("Confidence Threshold", 0.05, 0.95,
                         DEFAULT_CONF, 0.05, key="vid_conf")
    btn_vid  = st.button("⚡ Process Video", key="vid_btn",
                         type="primary", use_container_width=True)

    if btn_vid and vid_up:
        with st.spinner("Processing video — this may take a few minutes..."):
            tmp_in  = tempfile.mktemp(suffix=".mp4")
            tmp_out = tempfile.mktemp(suffix=".mp4")
            with open(tmp_in, "wb") as f:
                f.write(vid_up.read())
            cap = cv2.VideoCapture(tmp_in)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, _, _ = detector.detect(frame, conf=conf_vid)
                writer.write(annotated)
            cap.release()
            writer.release()
            gc.collect()
            try:
                os.remove(tmp_in)
            except OSError:
                pass
        with open(tmp_out, "rb") as f:
            st.video(f.read())
        st.success("Video processed successfully!")

# ── LIVE CAMERA ────────────────────────────────────────────────────────────
with tab3:
    heading("Live Camera Detection",
            "Capture a frame from your browser camera — detection runs on every snapshot.")
    col1, col2  = st.columns(2)
    with col1:
        cam_frame = st.camera_input("📷 Point your camera and capture", key="cam")
        conf_cam  = st.slider("Confidence Threshold", 0.05, 0.95,
                              DEFAULT_CONF, 0.05, key="cam_conf")
    with col2:
        if cam_frame:
            raw    = np.frombuffer(cam_frame.getvalue(), np.uint8)
            frame  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            result = detector.annotate_frame(frame, conf=conf_cam)
            gc.collect()
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption="Detection Result", use_container_width=True)
        else:
            st.markdown(
                '<p style="color:#475569;padding-top:40px;text-align:center;">'
                'Capture a photo on the left to see detection here.</p>',
                unsafe_allow_html=True
            )

# ── RTSP ───────────────────────────────────────────────────────────────────
with tab4:
    heading("RTSP Stream",
            "Enter a publicly accessible RTSP URL. "
            "Private-network cameras (192.168.x.x) are not reachable from cloud.")
    rtsp_url  = st.text_input("RTSP URL",
                              placeholder="rtsp://your-camera-ip:554/stream")
    conf_rtsp = st.slider("Confidence Threshold", 0.05, 0.95,
                          DEFAULT_CONF, 0.05, key="rtsp_conf")
    btn_rtsp  = st.button("⚡ Grab Frame", key="rtsp_btn", type="primary")

    if btn_rtsp:
        if not rtsp_url.strip():
            st.warning("Please enter an RTSP URL first.")
        else:
            with st.spinner("Connecting to stream..."):
                cap  = cv2.VideoCapture(rtsp_url.strip())
                ok, frame = cap.read()
                cap.release()
            if ok:
                annotated = detector.annotate_frame(frame, conf=conf_rtsp)
                gc.collect()
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="Stream Frame", use_container_width=True)
            else:
                st.error("Could not connect to stream. "
                         "Check the URL and ensure it is publicly accessible.")

# Footer
st.markdown(f"""
<div class="argus-footer">
  {LOGO_SVG}
  <span style="color:#f59e0b;font-size:12px;font-weight:700;letter-spacing:0.1em;">ARGUS VISION</span>
  <span style="color:#1e2d3d;">·</span>
  <span style="color:#475569;font-size:12px;">
    Real-time object detection · YOLO11 · {len(detector.names)} COCO classes
  </span>
</div>
""", unsafe_allow_html=True)
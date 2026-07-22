"""
Argus Vision — Streamlit
Uses your EXACT original CSS, HTML structure, class names, fonts, and JS logic.
Templates folder is rebuilt in Python using the same HTML from your originals.
"""

import base64
import gc
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import detector, DEFAULT_CONF

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argus Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load your original style.css ───────────────────────────────────
original_css = Path("static/css/style.css").read_text()

def file_to_b64(path, mime="image/png"):
    p = Path(path)
    if not p.exists():
        return ""
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

logo_src = file_to_b64("static/images/logo2.png")
logo_tag  = (f'<img src="{logo_src}" class="brand-logo" alt="Argus Vision" />'
             if logo_src else "")

def img_to_b64_src(bgr_img):
    """Convert OpenCV BGR image → data URI for inline HTML."""
    _, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"

# ── CSS: inject your original + minimal Streamlit overrides ────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

{original_css}

/* ── Streamlit chrome removal ── */
#MainMenu, footer, header {{ visibility: hidden !important; }}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: var(--ink) !important;
}}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* ── Tabs styled like nav-links ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid var(--line-soft) !important;
    gap: 0 !important;
    padding: 0 34px !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .8rem !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid transparent !important;
    padding: 14px 0 !important;
    margin-right: 26px !important;
    letter-spacing: .03em !important;
    transition: .18s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--teal-soft) !important;
    border-bottom-color: var(--teal) !important;
}}
.stTabs [aria-selected="true"] {{
    color: var(--teal-soft) !important;
    border-bottom: 1px solid var(--teal) !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding: 0 !important;
    background: transparent !important;
}}

/* ── Primary button → .btn .btn-primary ── */
button[kind="primary"] {{
    background: var(--gold) !important;
    color: var(--ink) !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius) !important;
    font-size: .95rem !important;
    transition: .18s !important;
    width: 100% !important;
}}
button[kind="primary"]:hover {{
    background: var(--gold-soft) !important;
    transform: translateY(-1px) !important;
}}

/* ── Secondary button → .btn .btn-ghost ── */
button[kind="secondary"] {{
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    font-family: 'Chakra Petch', sans-serif !important;
    border-radius: var(--radius) !important;
    transition: .18s !important;
}}
button[kind="secondary"]:hover {{
    border-color: var(--teal) !important;
    color: var(--teal-soft) !important;
}}

/* ── Slider ── */
input[type="range"] {{ accent-color: var(--teal) !important; }}
.stSlider label, [data-testid="stWidgetLabel"] p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .82rem !important;
    color: var(--muted) !important;
    letter-spacing: .04em !important;
}}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {{
    color: var(--gold) !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ── File uploader → .drop style ── */
[data-testid="stFileUploader"] section {{
    background: var(--ink-2) !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: var(--radius) !important;
    color: var(--muted) !important;
    transition: .18s !important;
    padding: 34px 18px !important;
    text-align: center !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: var(--teal) !important;
    color: var(--teal-soft) !important;
    background: var(--ink-2) !important;
}}
[data-testid="stFileUploader"] section * {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .85rem !important;
}}

/* ── Camera input ── */
[data-testid="stCameraInput"] section {{
    background: var(--ink-2) !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: var(--radius) !important;
}}

/* ── Text input → input[type=url] style ── */
[data-testid="stTextInput"] input {{
    background: var(--ink-2) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .88rem !important;
    border-radius: var(--radius) !important;
    padding: 11px 13px !important;
    width: 100% !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: var(--teal) !important;
    outline: none !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] p {{
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--muted) !important;
    letter-spacing: .06em !important;
}}

/* ── Markdown inside components ── */
[data-testid="stMarkdownContainer"] p {{
    color: var(--muted) !important;
    font-size: .92rem !important;
    line-height: 1.5 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: var(--ink); }}
::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--teal); }}
</style>
""", unsafe_allow_html=True)


# ── Navbar — exact base.html structure ─────────────────────────────
st.markdown(f"""
<div class="grid-bg" aria-hidden="true"></div>
<header class="nav">
  <a class="brand" href="#">
    {logo_tag}
    <span class="brand-name">ARGUS<span class="brand-name-2">VISION</span></span>
  </a>
  <nav class="nav-links">
    <a href="#">Image</a>
    <a href="#">Video</a>
    <a href="#">Live Camera</a>
    <a href="#">RTSP</a>
  </nav>
</header>
""", unsafe_allow_html=True)


# ── Reusable HTML snippets matching your templates ─────────────────

def tool_head(eyebrow, h1, desc=""):
    d = f"<p>{desc}</p>" if desc else ""
    st.markdown(f"""
<div class="tool-head">
  <div class="eyebrow mono"><span class="dot"></span> {eyebrow}</div>
  <h1>{h1}</h1>{d}
</div>""", unsafe_allow_html=True)


def stage_placeholder(icon_svg, msg):
    return f"""
<div class="stage">
  <span class="corner tl"></span><span class="corner tr"></span>
  <span class="corner bl"></span><span class="corner br"></span>
  <div class="placeholder">
    <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.4">{icon_svg}</svg>
    <div>{msg}</div>
  </div>
</div>"""


def stage_image(bgr_img):
    src = img_to_b64_src(bgr_img)
    return f"""
<div class="stage">
  <span class="corner tl"></span><span class="corner tr"></span>
  <span class="corner bl"></span><span class="corner br"></span>
  <img src="{src}" style="max-width:100%;max-height:72vh;display:block;">
</div>"""


def readout_html(summary, details=None):
    total  = sum(summary.values())
    chips  = f'<span class="chip total">TOTAL {total}</span>'
    chips += "".join(
        f'<span class="chip">{k} <b>{v}</b></span>'
        for k, v in sorted(summary.items(), key=lambda x: -x[1])
    )
    if details:
        rows = "".join(
            f'<div class="row"><span class="lbl">{d["label"]}</span>'
            f'<span class="cf">{d["conf"]*100:.1f}%</span></div>'
            for d in sorted(details, key=lambda x: -x["conf"])
        ) or '<div class="row"><span>No objects detected.</span></div>'
        det = f'<div class="det-list" style="display:block">{rows}</div>'
    else:
        det = ""
    return f'<div class="readout"><div class="summary-chips">{chips}</div>{det}</div>'


ICONS = {
    "image":  '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.8"/><path d="M21 15l-5-5L5 21"/>',
    "video":  '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M10 9l5 3-5 3z" fill="currentColor" stroke="none"/>',
    "webcam": '<circle cx="12" cy="10" r="7"/><circle cx="12" cy="10" r="2.5"/><path d="M6 20h12"/>',
    "rtsp":   '<path d="M4 6h16v10H4z"/><path d="M8 20h8M12 16v4"/><path d="M15 9l3 2-3 2z" fill="currentColor" stroke="none"/>',
}


# ── Tabs ───────────────────────────────────────────────────────────
t_home, t_img, t_vid, t_cam, t_rtsp = st.tabs([
    "Home", "Image", "Video", "Live Camera", "RTSP"
])


# ════════════════════════════════════════════════════════════════════
#  HOME — exact index.html content
# ════════════════════════════════════════════════════════════════════
with t_home:
    st.markdown(f"""
<section class="hero">
  <div class="hero-frame">
    <span class="conf-tag mono">argus 0.99</span>
    <span class="corner tl"></span><span class="corner tr"></span>
    <span class="corner bl"></span><span class="corner br"></span>
    <span class="scanline"></span>
    <div class="eyebrow mono">
      <span class="dot"></span> LIVE · {len(detector.names)} CLASSES · {detector.model_name}
    </div>
    <h1>The all-seeing<br><span class="accent">detection</span> console.</h1>
    <p class="lead">
      Argus Vision runs state-of-the-art YOLO detection on your images, videos,
      webcam, and any RTSP camera — finding people, vehicles, and 70+ other
      objects in real time. Point it at a source and watch it draw the boxes.
    </p>
    <div class="hero-cta">
      <span class="btn btn-primary">Detect an image →</span>
      <span class="btn btn-ghost">Connect an RTSP camera</span>
    </div>
    <div class="stats">
      <div class="stat"><b>80</b><span>COCO CLASSES</span></div>
      <div class="stat"><b>4</b><span>INPUT SOURCES</span></div>
      <div class="stat"><b>YOLO11</b><span>DETECTION MODEL</span></div>
      <div class="stat"><b>Real-time</b><span>LIVE OVERLAY</span></div>
    </div>
  </div>
</section>

<section class="section" id="modes">
  <div class="section-head">
    <span class="idx mono">// SOURCES</span>
    <h2>Choose what to watch</h2>
    <span class="rule"></span>
  </div>
  <div class="modes">

    <div class="mode-card">
      <span class="mc-corner tl"></span><span class="mc-corner br"></span>
      <span class="src mono">SRC_01 · STILL</span>
      <svg class="mode-icon" width="44" height="44" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.6">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="1.8"/>
        <path d="M21 15l-5-5L5 21"/>
      </svg>
      <h3>Image</h3>
      <p>Upload a photo and get an annotated result with every detected object and its confidence.</p>
      <span class="go mono">OPEN →</span>
    </div>

    <div class="mode-card">
      <span class="mc-corner tl"></span><span class="mc-corner br"></span>
      <span class="src mono">SRC_02 · CLIP</span>
      <svg class="mode-icon" width="44" height="44" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.6">
        <rect x="2" y="4" width="20" height="16" rx="2"/>
        <path d="M10 9l5 3-5 3z" fill="currentColor" stroke="none"/>
      </svg>
      <h3>Video</h3>
      <p>Process a recorded clip frame by frame and play back the fully annotated result inline.</p>
      <span class="go mono">OPEN →</span>
    </div>

    <div class="mode-card">
      <span class="mc-corner tl"></span><span class="mc-corner br"></span>
      <span class="src mono">SRC_03 · WEBCAM</span>
      <svg class="mode-icon" width="44" height="44" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.6">
        <circle cx="12" cy="10" r="7"/>
        <circle cx="12" cy="10" r="2.5"/>
        <path d="M6 20h12"/>
      </svg>
      <h3>Live Camera</h3>
      <p>Capture from your browser camera with live bounding boxes on every frame.</p>
      <span class="go mono">OPEN →</span>
    </div>

    <div class="mode-card">
      <span class="mc-corner tl"></span><span class="mc-corner br"></span>
      <span class="src mono">SRC_04 · NETWORK</span>
      <svg class="mode-icon" width="44" height="44" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.6">
        <path d="M4 6h16v10H4z"/>
        <path d="M8 20h8M12 16v4"/>
        <path d="M15 9l3 2-3 2z" fill="currentColor" stroke="none"/>
      </svg>
      <h3>RTSP Camera</h3>
      <p>Paste an <span class="mono">rtsp://</span> URL to connect an IP camera and detect the live feed.</p>
      <span class="go mono">OPEN →</span>
    </div>

  </div>
</section>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  IMAGE — exact image.html structure
# ════════════════════════════════════════════════════════════════════
with t_img:
    st.markdown('<section class="tool">', unsafe_allow_html=True)
    tool_head("SRC_01 · STILL IMAGE", "Image detection",
              "Upload a photo. Argus draws a box around every object it finds "
              "and lists each one with its confidence.")

    st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="panel"><h4>Input</h4>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop an image here or click to browse",
                                    type=["jpg","jpeg","png","bmp","webp"],
                                    key="img_up", label_visibility="collapsed")
        conf_img = st.slider("CONFIDENCE THRESHOLD", 0.05, 0.95,
                             DEFAULT_CONF, 0.05, key="img_c",
                             format="%.2f")
        run_img  = st.button("Run detection", key="img_run",
                             type="primary", use_container_width=True)
        st.markdown(
            '<p class="hint">Detects all 80 COCO classes — people, cars, '
            'buses, trucks, motorcycles, bicycles, and more.</p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        stage_ph   = st.empty()
        readout_ph = st.empty()
        stage_ph.markdown(
            stage_placeholder(ICONS["image"], "Annotated result appears here"),
            unsafe_allow_html=True)

    if uploaded:
        raw  = np.frombuffer(uploaded.read(), np.uint8)
        img  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            s = 1280 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))

        if run_img:
            with st.spinner("DETECTING…"):
                annotated, summary, details = detector.detect(img, conf=conf_img)
                gc.collect()
            stage_ph.markdown(stage_image(annotated), unsafe_allow_html=True)
            readout_ph.markdown(readout_html(summary, details), unsafe_allow_html=True)
        else:
            stage_ph.markdown(stage_image(img), unsafe_allow_html=True)

    st.markdown('</div></section>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  VIDEO — exact video.html structure
# ════════════════════════════════════════════════════════════════════
with t_vid:
    st.markdown('<section class="tool">', unsafe_allow_html=True)
    tool_head("SRC_02 · VIDEO CLIP", "Video detection",
              "Upload a clip. Argus processes it frame by frame and plays back "
              "the annotated result here.")

    st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="panel"><h4>Input</h4>', unsafe_allow_html=True)
        vid_up   = st.file_uploader("Drop a video here or click to browse",
                                    type=["mp4","avi","mov","mkv","webm"],
                                    key="vid_up", label_visibility="collapsed")
        conf_vid = st.slider("CONFIDENCE THRESHOLD", 0.05, 0.95,
                             DEFAULT_CONF, 0.05, key="vid_c", format="%.2f")
        run_vid  = st.button("Run detection", key="vid_run",
                             type="primary", use_container_width=True)
        st.markdown(
            '<p class="hint">Longer clips take longer — the whole video is '
            'processed before playback. Formats: mp4, mov, avi, mkv, webm.</p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        vid_stage_ph = st.empty()
        vid_result_ph = st.empty()
        vid_meta_ph   = st.empty()
        vid_stage_ph.markdown(
            stage_placeholder(ICONS["video"], "Annotated video appears here"),
            unsafe_allow_html=True)

    if run_vid and vid_up:
        with st.spinner("PROCESSING VIDEO…"):
            tmp_in  = tempfile.mktemp(suffix=".mp4")
            tmp_out = tempfile.mktemp(suffix=".mp4")
            with open(tmp_in, "wb") as f:
                f.write(vid_up.read())
            cap     = cv2.VideoCapture(tmp_in)
            fps     = cap.get(cv2.CAP_PROP_FPS) or 25
            w, h    = int(cap.get(3)), int(cap.get(4))
            writer  = cv2.VideoWriter(tmp_out,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w, h))
            from collections import Counter
            totals = Counter(); frames = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                ann, summary, _ = detector.detect(frame, conf=conf_vid)
                writer.write(ann)
                frames += 1
                for k, v in summary.items():
                    totals[k] = max(totals[k], v)
            cap.release(); writer.release(); gc.collect()
            try: os.remove(tmp_in)
            except OSError: pass

        vid_stage_ph.empty()
        with open(tmp_out, "rb") as f:
            vid_result_ph.video(f.read())

        total = sum(totals.values())
        chips = (f'<span class="chip total">TOTAL {total}</span>' +
                 "".join(f'<span class="chip">{k} <b>{v}</b></span>'
                         for k, v in sorted(totals.items(), key=lambda x:-x[1])))
        vid_meta_ph.markdown(
            f'<div class="readout"><div class="summary-chips">{chips}</div>'
            f'<p class="hint">Processed {frames} frames · counts show peak '
            f'simultaneous objects per class.</p></div>',
            unsafe_allow_html=True)

    st.markdown('</div></section>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  LIVE CAMERA — exact webcam.html structure
# ════════════════════════════════════════════════════════════════════
with t_cam:
    st.markdown('<section class="tool">', unsafe_allow_html=True)
    tool_head("SRC_03 · LIVE WEBCAM", "Live camera detection",
              "Capture from your browser camera. Boxes and a live object count "
              "are drawn on every frame.")

    st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="panel"><h4>Controls</h4>', unsafe_allow_html=True)
        cam_frame = st.camera_input("Camera", key="cam",
                                    label_visibility="collapsed")
        conf_cam  = st.slider("CONFIDENCE THRESHOLD", 0.05, 0.95,
                              DEFAULT_CONF, 0.05, key="cam_c", format="%.2f")
        st.markdown(
            '<p class="hint">Each capture is sent for detection. '
            'Take a new photo to refresh the result.</p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        cam_ph = st.empty()
        cam_ph.markdown(
            stage_placeholder(ICONS["webcam"], 'Press "Take Photo" to begin'),
            unsafe_allow_html=True)

        if cam_frame:
            raw   = np.frombuffer(cam_frame.getvalue(), np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            res   = detector.annotate_frame(frame, conf=conf_cam)
            gc.collect()
            cam_ph.markdown(stage_image(res), unsafe_allow_html=True)

    st.markdown('</div></section>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  RTSP — exact rtsp.html structure
# ════════════════════════════════════════════════════════════════════
with t_rtsp:
    st.markdown('<section class="tool">', unsafe_allow_html=True)
    tool_head("SRC_04 · NETWORK CAMERA", "RTSP camera detection",
              "Connect an IP / CCTV camera by its stream URL. Argus opens the "
              "feed and detects people, vehicles, and more in real time.")

    st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="panel"><h4>Connection</h4>', unsafe_allow_html=True)
        st.markdown('<label class="field">RTSP / HTTP STREAM URL</label>',
                    unsafe_allow_html=True)
        rtsp_url  = st.text_input("rtsp", key="rtsp_url",
                                  placeholder="rtsp://user:pass@192.168.1.10:554/stream",
                                  label_visibility="collapsed")
        conf_rtsp = st.slider("CONFIDENCE THRESHOLD", 0.05, 0.95,
                              DEFAULT_CONF, 0.05, key="rtsp_c", format="%.2f")
        c1, c2    = st.columns(2)
        connect   = c1.button("Connect",    key="rtsp_conn", type="primary")
        _         = c2.button("Disconnect", key="rtsp_disc", type="secondary")
        st.markdown(
            '<p class="hint">Typical URL: '
            '<code>rtsp://user:pass@camera-ip:554/stream1</code>. '
            'Check your camera\'s manual for the exact path.</p>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        rtsp_ph = st.empty()
        rtsp_ph.markdown(
            stage_placeholder(ICONS["rtsp"],
                              'Enter a stream URL and press "Connect"'),
            unsafe_allow_html=True)

    if connect:
        if not rtsp_url or not rtsp_url.strip():
            st.markdown('<p class="err">Enter a stream URL first.</p>',
                        unsafe_allow_html=True)
        else:
            with st.spinner("CONNECTING…"):
                cap       = cv2.VideoCapture(rtsp_url.strip())
                ok, frame = cap.read(); cap.release()
            if ok:
                ann = detector.annotate_frame(frame, conf=conf_rtsp)
                gc.collect()
                rtsp_ph.markdown(stage_image(ann), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p class="err">Could not connect. Check the URL and '
                    'ensure it is publicly accessible.</p>',
                    unsafe_allow_html=True)

    st.markdown('</div></section>', unsafe_allow_html=True)


# ── Footer — exact base.html footer ────────────────────────────────
st.markdown("""
<footer class="footer">
  <span class="mono">ARGUS VISION</span>
  <span class="footer-dot">·</span>
  <span>Real-time object detection · YOLO11 · 80 COCO classes</span>
</footer>
""", unsafe_allow_html=True)
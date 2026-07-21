# Argus Vision — AI Object Detection Studio

Real-time object detection for **images, videos, your webcam, and any RTSP / IP camera**,
powered by **Ultralytics YOLO11**. Detects all 80 COCO classes — people, cars, buses,
trucks, motorcycles, bicycles, and more — and draws the boxes right on the page.

No login, no sign-up. Just run it and point it at a source.

```
Argus-Vision/
├── app.py              # Flask server + all routes
├── detector.py         # YOLO model wrapper
├── requirements.txt
├── yolo11n.pt          # bundled model weights (runs offline)
├── templates/          # landing + one page per source (image/video/webcam/rtsp)
├── static/             # css + js
├── uploads/            # temp input files (auto-created)
└── outputs/            # annotated results (auto-created)
```

## 1. Setup

You need **Python 3.9+**. From inside the `Argus-Vision` folder:

```bash
# (recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## 2. Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

That's it. The landing page links to four detection pages:

| Page          | What it does                                                        |
|---------------|---------------------------------------------------------------------|
| **Image**     | Upload a photo → annotated result + per-object confidence list.     |
| **Video**     | Upload a clip → processed frame-by-frame → annotated video inline.  |
| **Live Camera** | Streams the webcam on the host machine with live boxes + counts.  |
| **RTSP**      | Paste an `rtsp://` URL → live detection on an IP / CCTV feed.        |

## 3. Choosing accuracy vs. speed

The app ships with `yolo11n.pt` (nano) — the fastest model, so live webcam and RTSP
run smoothly even on a CPU. For **higher accuracy**, switch the model with one
environment variable (the weights download automatically on first use):

```bash
# macOS / Linux
ARGUS_MODEL=yolo11x.pt python app.py     # most accurate (best on a GPU)
ARGUS_MODEL=yolo11m.pt python app.py     # balanced

# Windows (PowerShell)
$env:ARGUS_MODEL="yolo11x.pt"; python app.py
```

Model options, fastest → most accurate:
`yolo11n.pt` · `yolo11s.pt` · `yolo11m.pt` · `yolo11l.pt` · `yolo11x.pt`

You can also change the default confidence threshold:

```bash
ARGUS_CONF=0.5 python app.py
```

> **A note on accuracy:** no detection model is "100% correct" — YOLO11 is
> state of the art, but lighting, distance, motion blur, and unusual angles can
> still cause misses or false positives. The bigger models (`l` / `x`) and a
> higher-resolution source give the best results.

## Tips & troubleshooting

- **Live Camera shows nothing** → the machine running `app.py` has no webcam, or another
  app is using it. Live/webcam detection uses the *server host's* camera (device 0).
- **RTSP won't connect** → double-check the URL (often
  `rtsp://user:pass@camera-ip:554/stream1`); confirm the camera is reachable on your
  network. First connection can take a few seconds.
- **Video won't play inline** → install `ffmpeg` for guaranteed H.264 output. Without it
  the file still saves; download it from the player if it doesn't preview.
- **GPU** → install the CUDA build of PyTorch and Ultralytics will use it automatically,
  making the larger models fast.
```

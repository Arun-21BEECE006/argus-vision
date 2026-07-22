"""
Argus Vision - detection engine.

Thin wrapper around Ultralytics YOLO. Loads the model once and exposes
helpers to detect on a single image/frame and return both the annotated
image and a structured summary of what was found.
"""

import os
from collections import Counter

import cv2
from ultralytics import YOLO

# ---- Configuration (override with environment variables if you like) --------
# Swap the model for higher accuracy at the cost of speed:
#   yolo11n.pt  -> fastest, great for live/RTSP (DEFAULT)
#   yolo11s.pt  -> a little slower, a little more accurate
#   yolo11m.pt  -> balanced
#   yolo11l.pt  -> accurate, needs a decent CPU/GPU
#   yolo11x.pt  -> most accurate, best on a GPU
MODEL_NAME = os.environ.get("ARGUS_MODEL", "yolo11n.pt")

# Minimum confidence for a detection to be kept (0-1).
DEFAULT_CONF = float(os.environ.get("ARGUS_CONF", "0.35"))


class Detector:
    """Loads a YOLO model and runs detection on images / frames."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = YOLO(model_name)          # auto-downloads weights on first run
        self.names = self.model.names          # {id: class_name}

    def class_list(self):
        return list(self.names.values())

    def detect(self, image, conf: float = DEFAULT_CONF):
        """
        Run detection on a single image (numpy BGR array or file path).

        Returns:
            annotated (np.ndarray): image with boxes + labels drawn
            summary  (dict): {class_name: count}
            details  (list): [{label, conf, box:[x1,y1,x2,y2]}, ...]
        """
        result = self.model.predict(image, conf=conf, verbose=False, imgsz=640)[0]
        annotated = result.plot()  # BGR image with boxes drawn

        details = []
        counter = Counter()
        for box in result.boxes:
            label = self.names[int(box.cls)]
            confidence = round(float(box.conf), 3)
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            details.append({"label": label, "conf": confidence, "box": xyxy})
            counter[label] += 1

        return annotated, dict(counter), details

    def annotate_frame(self, frame, conf: float = DEFAULT_CONF):
        """
        Detect on a video/stream frame and draw a live count banner on top.
        Returns the annotated frame (BGR).
        """
        annotated, summary, _ = self.detect(frame, conf=conf)
        total = sum(summary.values())

        # Compact live banner in the top-left corner.
        label = f"Objects: {total}"
        if summary:
            top = sorted(summary.items(), key=lambda kv: -kv[1])[:4]
            label += "  |  " + "  ".join(f"{k}:{v}" for k, v in top)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (0, 0), (tw + 20, th + 20), (18, 18, 24), -1)
        cv2.putText(annotated, label, (10, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 230, 160), 2, cv2.LINE_AA)
        return annotated


# A single shared instance for the whole app (model loads only once).
detector = Detector()

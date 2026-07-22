"""
Argus Vision — detection engine (ONNX Runtime edition)
Replaces PyTorch (~200MB) with onnxruntime (~15MB).
Total idle RAM drops from ~500MB to ~180MB — fits Render free tier.
Box thickness and font scale automatically adapt to image size.
"""

import os
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH   = os.environ.get("ARGUS_MODEL",  "yolo11n.onnx")
DEFAULT_CONF = float(os.environ.get("ARGUS_CONF", "0.35"))
IMGSZ        = 640

COCO_NAMES = {
    0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"airplane",
    5:"bus",6:"train",7:"truck",8:"boat",9:"traffic light",
    10:"fire hydrant",11:"stop sign",12:"parking meter",13:"bench",
    14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",19:"cow",
    20:"elephant",21:"bear",22:"zebra",23:"giraffe",24:"backpack",
    25:"umbrella",26:"handbag",27:"tie",28:"suitcase",29:"frisbee",
    30:"skis",31:"snowboard",32:"sports ball",33:"kite",34:"baseball bat",
    35:"baseball glove",36:"skateboard",37:"surfboard",38:"tennis racket",
    39:"bottle",40:"wine glass",41:"cup",42:"fork",43:"knife",
    44:"spoon",45:"bowl",46:"banana",47:"apple",48:"sandwich",
    49:"orange",50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",
    54:"donut",55:"cake",56:"chair",57:"couch",58:"potted plant",
    59:"bed",60:"dining table",61:"toilet",62:"tv",63:"laptop",
    64:"mouse",65:"remote",66:"keyboard",67:"cell phone",68:"microwave",
    69:"oven",70:"toaster",71:"sink",72:"refrigerator",73:"book",
    74:"clock",75:"vase",76:"scissors",77:"teddy bear",78:"hair drier",
    79:"toothbrush"
}

_PALETTE = [
    (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
    (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
    (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
    (132,56,255),(82,0,133),(203,56,255),(255,149,200),(255,55,199),
]
def _color(cls_id): return _PALETTE[int(cls_id) % len(_PALETTE)]


class Detector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_name = model_path
        self.session    = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.in_name  = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        self.names    = COCO_NAMES

    def _preprocess(self, img_bgr):
        h, w   = img_bgr.shape[:2]
        scale  = IMGSZ / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        ph = (IMGSZ - nh) // 2
        pw = (IMGSZ - nw) // 2
        padded = cv2.copyMakeBorder(
            resized, ph, IMGSZ - nh - ph, pw, IMGSZ - nw - pw,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])
        return tensor, scale, ph, pw

    def _postprocess(self, raw, scale, ph, pw, orig_h, orig_w, conf_thresh):
        pred  = raw[0].T
        cx, cy, bw, bh = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
        scores    = pred[:, 4:].max(axis=1)
        class_ids = pred[:, 4:].argmax(axis=1)

        keep = scores >= conf_thresh
        if not keep.any():
            return np.empty((0,4)), np.empty(0), np.empty(0, dtype=int)

        cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
        scores    = scores[keep]
        class_ids = class_ids[keep]

        x1 = ((cx - bw / 2) - pw) / scale
        y1 = ((cy - bh / 2) - ph) / scale
        x2 = ((cx + bw / 2) - pw) / scale
        y2 = ((cy + bh / 2) - ph) / scale

        x1 = np.clip(x1, 0, orig_w); x2 = np.clip(x2, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h); y2 = np.clip(y2, 0, orig_h)

        boxes_xywh = np.stack(
            [x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)
        idx = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), conf_thresh, 0.45)

        if len(idx) == 0:
            return np.empty((0,4)), np.empty(0), np.empty(0, dtype=int)

        idx = idx.flatten()
        xyxy = np.stack([x1[idx], y1[idx], x2[idx], y2[idx]], axis=1)
        return xyxy, scores[idx], class_ids[idx]

    def detect(self, image, conf: float = DEFAULT_CONF):
        orig_h, orig_w = image.shape[:2]
        tensor, scale, ph, pw = self._preprocess(image)
        raw = self.session.run([self.out_name], {self.in_name: tensor})[0]
        boxes, scores, class_ids = self._postprocess(
            raw, scale, ph, pw, orig_h, orig_w, conf)

        annotated = image.copy()
        counter   = Counter()
        details   = []

        # ── Scale thickness and font to image size so boxes are
        #    clearly visible on both small and large photos ──────────
        img_diag   = (orig_h ** 2 + orig_w ** 2) ** 0.5
        lw         = max(2, int(img_diag / 500))   # line width
        font_scale = max(0.6, img_diag / 2000)      # text size
        font_thick = max(1, lw // 2)                # text thickness
        pad        = max(6, lw * 3)                 # label padding

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label      = self.names.get(int(cls_id), str(cls_id))
            confidence = round(float(score), 3)
            color      = _color(cls_id)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)

            # Label background + text
            txt = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            label_y = max(y1, th + pad)
            cv2.rectangle(annotated,
                          (x1, label_y - th - pad),
                          (x1 + tw + pad, label_y),
                          color, -1)
            cv2.putText(annotated, txt,
                        (x1 + pad // 2, label_y - pad // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), font_thick, cv2.LINE_AA)

            counter[label] += 1
            details.append({
                "label": label,
                "conf":  confidence,
                "box":   [round(float(v), 1) for v in [x1, y1, x2, y2]],
            })

        return annotated, dict(counter), details

    def annotate_frame(self, frame, conf: float = DEFAULT_CONF):
        annotated, summary, _ = self.detect(frame, conf=conf)
        total = sum(summary.values())
        label = f"Objects: {total}"
        if summary:
            top    = sorted(summary.items(), key=lambda kv: -kv[1])[:4]
            label += "  |  " + "  ".join(f"{k}:{v}" for k, v in top)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (0, 0), (tw + 20, th + 20), (18, 18, 24), -1)
        cv2.putText(annotated, label, (10, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 230, 160), 2,
                    cv2.LINE_AA)
        return annotated


detector = Detector()
"""
pothole_ml_detector.py — real ML inference for pothole localization.

Trained with Ultralytics YOLOv8 on the michelpf/dataset-pothole dataset
(https://github.com/michelpf/dataset-pothole — YOLO-format annotations:
`class cx cy w h`, all normalized 0-1, single class "pothole").
See ../ml/README.md for the full training pipeline.

Training happens offline (needs torch/ultralytics + a GPU, and network
access to pull the dataset — none of which this server needs). The trained
model is exported to ONNX (`ml/export_onnx.py`), and *this* module is all
the deployed server needs to run inference: onnxruntime + OpenCV + NumPy,
no torch, no GPU, no network at serve time.

If no trained model has been placed at MODEL_PATH yet, `detect_pothole_bbox_ml`
returns None and the caller (`cv_verification.detect_pothole_bbox`) falls
back to the classical contour-based heuristic — training a model is a
strict upgrade you can drop in later, not a hard requirement to run the app.
"""
import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "pothole_yolo.onnx")
INPUT_SIZE = 640          # must match imgsz in ml/train.py and ml/export_onnx.py
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

_session = None
_session_load_attempted = False


def _get_session():
    """Lazily loads the ONNX Runtime session at most once per process.
    Any failure (missing file, missing onnxruntime package, corrupt model)
    is swallowed here and reported to the caller as "no model available" —
    this function must never raise, since it sits on the request path."""
    global _session, _session_load_attempted
    if _session is not None or _session_load_attempted:
        return _session
    _session_load_attempted = True
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[pothole_ml_detector] could not load '{MODEL_PATH}' ({e}); "
              f"falling back to the classical contour-based detector.")
        _session = None
    return _session


def model_available() -> bool:
    return _get_session() is not None


def _letterbox(img_bgr, size=INPUT_SIZE):
    """Resize+pad to a square `size`x`size` canvas, preserving aspect ratio
    (standard YOLO preprocessing). Returns the canvas plus the scale factor
    and padding offsets needed to map predicted boxes back to the original
    image's pixel coordinates."""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _nms(boxes_xyxy, scores, iou_threshold):
    """Greedy non-max suppression. boxes_xyxy: (N,4) array of x1,y1,x2,y2."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def _decode_yolov8_output(raw):
    """Normalizes an Ultralytics-exported YOLOv8 ONNX output — shape
    (1, 4+num_classes, num_anchors), e.g. (1, 5, 8400) for a single class —
    into (num_anchors, 4+num_classes)."""
    arr = raw[0]
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def detect_pothole_bbox_ml(img_bgr, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """
    Runs the trained YOLOv8 pothole detector on a BGR OpenCV image.

    Returns (x, y, w, h, confidence) in the ORIGINAL image's pixel
    coordinates for the single highest-confidence detection, or None if no
    model is loaded, or nothing scored above `conf_threshold`.
    """
    session = _get_session()
    if session is None:
        return None

    h0, w0 = img_bgr.shape[:2]
    if h0 == 0 or w0 == 0:
        return None

    canvas, scale, pad_left, pad_top = _letterbox(img_bgr, INPUT_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])  # NCHW

    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: blob})
    except Exception as e:
        print(f"[pothole_ml_detector] inference failed ({e}); skipping ML detection for this image.")
        return None

    preds = _decode_yolov8_output(outputs[0])
    if preds.ndim != 2 or preds.shape[1] < 5:
        return None

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    scores = class_scores[:, 0] if class_scores.shape[1] == 1 else class_scores.max(axis=1)

    mask = scores >= conf_threshold
    if not np.any(mask):
        return None
    boxes_cxcywh, scores = boxes_cxcywh[mask], scores[mask]

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    keep = _nms(boxes_xyxy, scores, iou_threshold)
    if not keep:
        return None
    boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]

    best_i = int(np.argmax(scores))
    bx1, by1, bx2, by2 = boxes_xyxy[best_i]
    conf = float(scores[best_i])

    # map letterboxed model-space coords back to the original image
    bx1 = (bx1 - pad_left) / scale
    by1 = (by1 - pad_top) / scale
    bx2 = (bx2 - pad_left) / scale
    by2 = (by2 - pad_top) / scale
    bx1, bx2 = sorted((max(0.0, min(w0, bx1)), max(0.0, min(w0, bx2))))
    by1, by2 = sorted((max(0.0, min(h0, by1)), max(0.0, min(h0, by2))))

    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return None

    return (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1), conf)

"""
pothole_svm_detector.py — real ML inference using the HOG+SVM model trained
by ml/train_classical_detector.py.

This is a genuine (if classical, non-deep-learning) machine learning
detector: multi-scale sliding window over the image, HOG feature extraction
per window, classification by a real trained scikit-learn SVM (with
cross-validated probability calibration), then non-max suppression over the
windows that score above threshold. It ships pre-trained (see
backend/ml_models/pothole_hog_svm.joblib) so it works immediately — no
training step required to use the app.

Position in the detection fallback chain (see cv_verification.detect_pothole_bbox):
  1. Trained YOLOv8 ONNX model, if you've run the ml/ deep-learning pipeline
     on your own machine (best accuracy, needs that training step)
  2. This HOG+SVM detector — ships pre-trained, real ML, works out of the box
  3. Classical contour/thresholding heuristic — the final fallback, needs no
     model file at all
"""
import os

import cv2
import numpy as np
from skimage.feature import hog

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_models", "pothole_hog_svm.joblib")
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.35
SCAN_MAX_WIDTH = 480          # image is downscaled to this width before scanning, for speed
WINDOW_SCALES = (0.55, 0.85, 1.2)  # window side length as a fraction of scan-image height
STRIDE_FRACTION = 0.22        # window stride as a fraction of the window size
ROI_Y_FRACTION = 0.30         # only scan the lower (1 - ROI_Y_FRACTION) of the image (road, not sky)

_model = None
_model_load_attempted = False


def _get_model():
    global _model, _model_load_attempted
    if _model is not None or _model_load_attempted:
        return _model
    _model_load_attempted = True
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[pothole_svm_detector] could not load '{MODEL_PATH}' ({e}); "
              f"falling back to the classical contour-based detector.")
        _model = None
    return _model


def model_available() -> bool:
    return _get_model() is not None


def _nms(boxes_xyxy, scores, iou_threshold):
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


def detect_pothole_bbox_svm(img_bgr, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """
    Real multi-scale sliding-window HOG+SVM detection.
    Returns (x, y, w, h, confidence) in ORIGINAL image pixel coordinates for
    the highest-confidence detection, or None if no model is available or
    nothing scored above `conf_threshold`.
    """
    model = _get_model()
    if model is None:
        return None

    clf = model["classifier"]
    scaler = model["scaler"]
    patch_size = model["patch_size"]
    hog_params = model["hog_params"]

    h0, w0 = img_bgr.shape[:2]
    if h0 == 0 or w0 == 0:
        return None

    scale = min(1.0, SCAN_MAX_WIDTH / w0)
    scan_w, scan_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    scan_img = cv2.resize(img_bgr, (scan_w, scan_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)

    roi_y0 = int(scan_h * ROI_Y_FRACTION)

    windows, patches = [], []
    for scale_frac in WINDOW_SCALES:
        win = max(16, int(scan_h * scale_frac * 0.5))  # window side length in scan-image pixels
        stride = max(4, int(win * STRIDE_FRACTION))
        for y in range(roi_y0, scan_h - win, stride):
            for x in range(0, scan_w - win, stride):
                patch = gray[y:y + win, x:x + win]
                if patch.shape[0] != win or patch.shape[1] != win:
                    continue
                windows.append((x, y, win, win))
                patches.append(cv2.resize(patch, (patch_size, patch_size)))

    if not patches:
        return None

    feats = np.array([hog(p, **hog_params) for p in patches], dtype=np.float32)
    feats_s = scaler.transform(feats)
    proba = clf.predict_proba(feats_s)[:, 1]  # P(pothole)

    mask = proba >= conf_threshold
    if not np.any(mask):
        return None

    kept_windows = [w for w, m in zip(windows, mask) if m]
    kept_scores = proba[mask]
    boxes_xyxy = np.array([[x, y, x + w, y + h] for (x, y, w, h) in kept_windows], dtype=np.float32)

    keep = _nms(boxes_xyxy, kept_scores, iou_threshold)
    if not keep:
        return None
    boxes_xyxy, kept_scores = boxes_xyxy[keep], kept_scores[keep]

    best_i = int(np.argmax(kept_scores))
    bx1, by1, bx2, by2 = boxes_xyxy[best_i]
    conf = float(kept_scores[best_i])

    # map scan-image coords back to the original image
    bx1, bx2 = bx1 / scale, bx2 / scale
    by1, by2 = by1 / scale, by2 / scale
    bx1 = max(0, min(w0, bx1)); bx2 = max(0, min(w0, bx2))
    by1 = max(0, min(h0, by1)); by2 = max(0, min(h0, by2))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return None

    return (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1), conf)

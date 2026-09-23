Drop your exported model here as `pothole_yolo.onnx` (this is exactly what
`ml/export_onnx.py` does by default — you shouldn't need to touch this
folder manually).

Until a model is present here, `backend/pothole_ml_detector.py` reports
"no model available" and `cv_verification.detect_pothole_bbox` transparently
falls back to the classical contour-based heuristic — the app works fine
either way.

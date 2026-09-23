#!/usr/bin/env python3
"""
export_onnx.py — exports a trained YOLOv8 checkpoint (best.pt) to ONNX and
places it exactly where backend/pothole_ml_detector.py expects it, so the
deployed server (which only needs onnxruntime, not torch) can use it
immediately.

Usage:
    python3 export_onnx.py --weights runs/pothole_yolov8/weights/best.pt

By default this writes to ../backend/ml_models/pothole_yolo.onnx — restart
the server (or it'll pick it up on next request, since the model is loaded
lazily) and pothole localization automatically switches from the classical
heuristic to your trained model.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Path to a trained YOLOv8 .pt checkpoint (e.g. best.pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="Must match INPUT_SIZE in backend/pothole_ml_detector.py")
    ap.add_argument(
        "--out", default=None,
        help="Destination .onnx path (default: ../backend/ml_models/pothole_yolo.onnx)",
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run:\n  pip install -r requirements-ml.txt"
        )

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(f"Checkpoint not found: {weights_path}")

    model = YOLO(str(weights_path))
    # opset 12 keeps compatibility with a wide range of onnxruntime versions;
    # simplify=True folds constant ops so inference in pothole_ml_detector.py is faster.
    exported_path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)

    dest = Path(args.out) if args.out else (Path(__file__).parent.parent / "backend" / "ml_models" / "pothole_yolo.onnx")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, dest)

    print(f"\nExported ONNX model -> {dest}")
    print("Restart the backend (or just wait for the next request) to start using the trained model.")


if __name__ == "__main__":
    main()

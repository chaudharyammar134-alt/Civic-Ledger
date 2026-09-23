#!/usr/bin/env python3
"""
train.py — trains a YOLOv8 pothole detector on the prepared dataset.

This is real training code using Ultralytics' YOLOv8, not a stub — but it
genuinely needs `torch` (+ ideally a CUDA GPU) and is meant to be run on
your own machine, not inside the lightweight server environment. Install
ml/requirements-ml.txt first:

    pip install -r ../requirements.txt -r requirements-ml.txt
    python3 prepare_dataset.py --source /path/to/dataset-pothole/dataset --out ./yolo_dataset
    python3 train.py --data ./yolo_dataset/data.yaml --epochs 80 --model yolov8n.pt

On an RTX 3050 (4/6GB laptop GPU), `yolov8n` (nano) at imgsz=640 with a
batch size of 16 is a good starting point — bump batch down if you hit an
out-of-memory error, or use `yolov8s` for a bit more accuracy once nano
trains cleanly.

Ultralytics downloads the pretrained COCO checkpoint (e.g. yolov8n.pt) on
first use if it isn't already cached locally — that's the one network
dependency in this whole pipeline, and it only happens on your own machine
at training time, never on the deployed server.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to data.yaml from prepare_dataset.py")
    ap.add_argument("--model", default="yolov8n.pt", help="Base checkpoint to fine-tune (yolov8n.pt/yolov8s.pt/...)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640, help="Must match INPUT_SIZE in backend/pothole_ml_detector.py")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU, default: auto-detect")
    ap.add_argument("--project", default="runs", help="Where Ultralytics writes training runs")
    ap.add_argument("--name", default="pothole_yolov8")
    ap.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs with no improvement)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run:\n"
            "  pip install -r requirements-ml.txt\n"
            "(this needs a machine with real disk/network access — not the deployed server)."
        )

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # single-class "pothole" detector — the dataset itself is single-class,
        # nothing extra needed here (nc/names already come from data.yaml).
    )

    best_weights = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nTraining finished. Best weights: {best_weights}")
    print(f"Next: python3 export_onnx.py --weights {best_weights}")


if __name__ == "__main__":
    main()

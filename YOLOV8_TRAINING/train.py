from ultralytics import YOLO
from pathlib import Path

# Put your YOLO-format dataset inside:
# dataset/images/train
# dataset/images/val
# dataset/labels/train
# dataset/labels/val

DATA = "dataset/data.yaml"

# YOLOv8 Nano is lightweight and suitable for a normal laptop.
# Change to yolov8s.pt if your laptop/GPU can handle it.
model = YOLO("yolov8n.pt")

model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=8,
    project="runs",
    name="pothole_yolov8",
    patience=20,
    plots=True
)

print("\nTraining finished.")
print("Your trained model should be:")
print("runs/pothole_yolov8/weights/best.pt")

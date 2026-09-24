from ultralytics import YOLO
from pathlib import Path
import shutil

best = Path("runs/pothole_yolov8/weights/best.pt")

if not best.exists():
    raise FileNotFoundError(
        f"{best} not found. Train the model first using: python train.py"
    )

model = YOLO(str(best))

# Export trained YOLO model to ONNX.
exported = model.export(format="onnx", imgsz=640)

exported = Path(exported)
target = Path("pothole_yolo.onnx")

if exported.resolve() != target.resolve():
    shutil.copy2(exported, target)

print("\nONNX model created:")
print(target.resolve())

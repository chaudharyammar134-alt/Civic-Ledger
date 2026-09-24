from ultralytics import YOLO
import sys

model_path = "runs/pothole_yolov8/weights/best.pt"

if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    raise SystemExit(1)

image = sys.argv[1]

model = YOLO(model_path)
results = model.predict(
    source=image,
    conf=0.25,
    save=True
)

print("\nPrediction complete.")
print("Check the generated folder under runs/detect/")

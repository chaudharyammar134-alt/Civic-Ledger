# YOLOv8 Pothole Training Package

This package trains a custom YOLOv8 pothole detector and exports the trained
model to `pothole_yolo.onnx`.

## 1. Install

Open PowerShell in this folder:

    python -m pip install -r requirements.txt

If you use a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Then:

    python -m pip install -r requirements.txt

## 2. Prepare the dataset

The training dataset must use YOLO object-detection labels.

Expected structure:

    dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

For every image, its label file should have the same filename:

    image001.jpg
    image001.txt

A YOLO detection label contains:

    class_id x_center y_center width height

For this project there is one class:

    0 = pothole

## 3. Train

Run:

    python train.py

The important trained file will be:

    runs/pothole_yolov8/weights/best.pt

## 4. Export to ONNX

After training:

    python export.py

This creates:

    pothole_yolo.onnx

This is the file you can copy into your main pothole application.

## 5. Test a real image

Run:

    python predict.py "C:\path\to\real_pothole.jpg"

The prediction output is saved by Ultralytics under `runs/detect/`.

## Recommended workflow

1. Download a pothole dataset with bounding-box annotations.
2. Convert/organize it into the YOLO structure above if necessary.
3. Put training images/labels into `dataset/images/train` and `dataset/labels/train`.
4. Put validation images/labels into `dataset/images/val` and `dataset/labels/val`.
5. Run `python train.py`.
6. Take `runs/pothole_yolov8/weights/best.pt`.
7. Run `python export.py`.
8. Copy `pothole_yolo.onnx` into your main application.

IMPORTANT:
Do not put the raw dataset into this ZIP. The dataset can be very large.
This ZIP contains the complete training code/configuration, not the trained
weights themselves.

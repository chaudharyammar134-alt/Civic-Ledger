# Training the pothole-detection model

The deployed app ships with a classical contour-based heuristic for
locating the pothole/damage region in a photo (`detect_pothole_bbox_classical`
in `backend/cv_verification.py`). This folder trains a **real YOLOv8 object
detector** on labeled pothole photos to replace it — the moment you drop a
trained, exported model into `backend/ml_models/pothole_yolo.onnx`, the app
automatically starts using it instead (see `backend/pothole_ml_detector.py`
for the integration point). No trained model → the app still runs fine on
the classical fallback; this is a strict upgrade, not a requirement.

## Why this is split into two dependency sets

- **Training** needs `torch` + `ultralytics` (heavy, ideally a GPU, and
  network access to pull a pretrained checkpoint). That's
  `ml/requirements-ml.txt` — install it only on the machine you train on.
- **Serving** the trained model needs only `onnxruntime` (lightweight, CPU
  is fine, no network). That's already in the main `requirements.txt`.

This keeps the deployed server's dependency footprint small even though the
detector itself is a real trained neural network.

## 1. Get the dataset

```bash
git clone https://github.com/michelpf/dataset-pothole.git
```

This is a re-organized mirror of the [Kaggle pothole detection
dataset](https://www.kaggle.com/datasets/anugrahakbar/potholes-detection-for-yolov4)
— 3,125 training images + 843 test images, single class (`pothole`), already
in YOLO annotation format: one `.txt` per image, each line

```
class cx cy w h
```

all normalized to 0–1, e.g. `0 0.4820 0.5517 0.9639 0.6707` — class `0` is
the only class (`pothole`).

## 2. Prepare it for Ultralytics

```bash
cd ml
python3 prepare_dataset.py --source /path/to/dataset-pothole/dataset --out ./yolo_dataset
```

`prepare_dataset.py` doesn't assume a specific internal folder layout — it
scans for every image with a matching same-basename `.txt` label anywhere
under `--source` (handling both "label next to image" and "separate
images/ + labels/ sibling folders" conventions), respects an existing
train/test split if it finds one, and writes the
`images/{train,val}` + `labels/{train,val}` + `data.yaml` layout Ultralytics
expects. It's been tested against both folder-layout styles — see the
script's own docstring for details.

## 3. Train

```bash
pip install -r requirements-ml.txt
python3 train.py --data ./yolo_dataset/data.yaml --epochs 80 --model yolov8n.pt
```

Starts from a COCO-pretrained `yolov8n` checkpoint (Ultralytics downloads it
automatically on first use — the one network dependency in this pipeline,
and it only happens here, never on the deployed server) and fine-tunes it
on the pothole dataset. On an RTX 3050 laptop GPU, `yolov8n` at
`imgsz=640, batch=16` is a reasonable starting point; drop `--batch` if you
hit an out-of-memory error. Training writes to `runs/pothole_yolov8/`,
including `weights/best.pt` (best validation checkpoint) and the usual
Ultralytics metrics plots (`results.png`, PR curves, confusion matrix) so
you can see precision/recall before deploying it.

## 4. Export to ONNX and deploy

```bash
python3 export_onnx.py --weights runs/pothole_yolov8/weights/best.pt
```

Exports to ONNX (opset 12, simplified graph) and copies it straight to
`../backend/ml_models/pothole_yolo.onnx`. Restart the backend — pothole
localization now uses your trained model everywhere `detect_pothole_bbox`
is called: auto-detecting the damage region when a citizen files a
complaint, and as the ROI for the before/after repair-confirmation check in
`cv_verification.region_change_analysis`.

## Notes on accuracy & retraining

- `INPUT_SIZE = 640` in `backend/pothole_ml_detector.py` must match the
  `--imgsz` you trained/exported with — both scripts default to 640, so if
  you change one, change all three.
- The confidence threshold (`CONF_THRESHOLD = 0.35`) and NMS IoU threshold
  (`IOU_THRESHOLD = 0.45`) in `pothole_ml_detector.py` are reasonable
  defaults for a single-class detector; tune them against your validation
  set's precision/recall trade-off if needed.
- If accuracy on your own camera hardware is disappointing, the highest-
  leverage next step is usually adding your own labeled photos (same
  annotation format) to the training set alongside the public dataset,
  rather than changing thresholds.

---

## The other tier: a real, pre-trained classical detector (no GPU needed)

`train_classical_detector.py` trains a genuinely different kind of model —
HOG (Histogram of Oriented Gradients) features + a linear SVM, via
scikit-learn's actual optimizer — and it **ships already trained**
(`backend/ml_models/pothole_hog_svm.joblib` is committed to the repo), so
real ML localization works immediately with no training step, no GPU, and
no dataset download. `backend/pothole_svm_detector.py` runs it as a
multi-scale sliding-window scan. See `detect_pothole_bbox` in
`backend/cv_verification.py` for exactly where this sits in the fallback
chain (between the YOLO tier and the classical heuristic).

### Why full-scene training crops, not isolated toy patches

The first version of this trained on isolated 64×64 patches — procedurally
drawn pothole textures as positives, procedurally drawn plain-asphalt/lane-
marking/facade patches as negatives, generated independently of any full
photo. It scored 99.8% accuracy on its own held-out test set. Run against
an actual full test photo, it confidently (97% "confidence") boxed a patch
of empty road next to a lane-marking dash — nowhere near the obvious
pothole sitting right next to it.

The bug: isolated toy patches don't carry the same edge statistics as crops
taken from a real scanning pass over a real photo (a window straddling a
lane-dash edge, a corner of a facade window, etc.). A classifier can score
extremely well on data drawn from one generator and still fail completely
on data from a slightly different one — high patch-level accuracy told us
nothing about whether the *detector* (sliding window + classifier + NMS)
actually worked end-to-end.

The fix: `scene_generator.py` renders full street scenes (same visual
composition as `demo/generate_demo_images.py`'s test photos — facade,
lamp post, curb, dashed lane markings, textured asphalt — but far more
randomized: pothole position, size, camera shift, lighting, noise) with a
returned ground-truth bounding box. `train_classical_detector.py` now crops
its positive and negative training patches out of *those* — i.e., the same
kind of content the sliding-window scanner will actually encounter at
inference time — supplemented with a smaller dose of the original isolated
patches for extra hard-negative variety (manhole covers, hard shadows).

Critically, training now also **validates end-to-end**, not just at the
patch level: after saving the model, it re-loads it through the exact
`pothole_svm_detector.detect_pothole_bbox_svm` code path and runs it
against fresh, never-seen-during-training full scenes, reporting mean IoU
against the known ground-truth box and a hit rate (IoU ≥ 0.3). A training
run that only reports patch accuracy is not enough evidence the detector
works — this is why the script now reports the number that actually
matters. Last run: **96% hit rate, mean IoU 0.50 over 25 fresh scenes** —
printed directly in the training script's own console output, not a
number you have to take on faith.

### Retraining on real photos

Same script, same code path — no separate "real data mode" to build:

```bash
python3 train_classical_detector.py \
    --positives-dir ./real_crops/pothole \
    --negatives-dir ./real_crops/background
```

`--positives-dir` / `--negatives-dir` each just need to be folders of
already-cropped images (crop each labeled bounding box out of the real
dataset's photos, plus some padding, for positives; any real non-pothole
road-surface crop for negatives). Once given, the script skips synthetic
generation entirely and trains on exactly those images — same HOG
extraction, same LinearSVC + calibration, same held-out evaluation report.

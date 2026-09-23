#!/usr/bin/env python3
"""
prepare_dataset.py — turns a checkout of

    https://github.com/michelpf/dataset-pothole

into the images/{train,val} + labels/{train,val} + data.yaml layout that
Ultralytics YOLOv8 expects for training.

That dataset is already in YOLO-annotation format — one .txt per image,
each line `class cx cy w h` normalized to 0-1, e.g.:

    0 0.48197115384615385 0.5516826923076923 0.9639423076923077 0.6706730769230769

— single class (0 = pothole), which is exactly what this script assumes
(`nc: 1`, `names: ['pothole']`).

This script does NOT assume a specific internal folder name (Kaggle-derived
dataset mirrors vary — "train/test", "images/labels" nesting order, etc.).
Instead it:
  1. Recursively scans the source path for every image file that has a
     same-basename `.txt` label file next to it (that pairing IS a labeled
     example, regardless of what folder it lives in).
  2. If it finds top-level splits literally named train/ and test/ (or
     val/), it respects that split faithfully (test -> val, since
     Ultralytics' convention is train/val).
  3. Otherwise, it makes its own reproducible train/val split.
  4. Copies (not moves — the source checkout is left untouched) every pair
     into the Ultralytics layout, and writes data.yaml.

Usage:
    python3 prepare_dataset.py --source /path/to/dataset-pothole/dataset --out ./yolo_dataset
    python3 prepare_dataset.py --source /path/to/dataset-pothole/dataset --out ./yolo_dataset --val-fraction 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_labeled_pairs(root: Path):
    """
    Yields (image_path, label_path) for every image that has a matching
    same-basename .txt label file SOMEWHERE under `root` — not necessarily
    in the same folder. Real YOLO-format dataset mirrors commonly put
    images/ and labels/ as separate sibling trees (e.g.
    train/images/x.jpg + train/labels/x.txt), so matching purely by
    same-directory .txt (as a naive glob would) misses most real datasets.
    Basenames are matched within this root only, first-match-wins on
    duplicates (a warning is printed so you can check that's a real
    collision, not a scan bug).
    """
    label_by_stem = {}
    for label_path in root.rglob("*.txt"):
        if label_path.stem in label_by_stem:
            print(f"  [warn] duplicate label basename '{label_path.stem}' under {root} "
                  f"— keeping {label_by_stem[label_path.stem]}, ignoring {label_path}")
            continue
        label_by_stem[label_path.stem] = label_path

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = label_by_stem.get(img_path.stem)
        if label_path is not None:
            yield img_path, label_path


def detect_existing_split(root: Path):
    """
    Returns {"train": [...], "val": [...]} if the source already has clear
    train/ and test-or-val/ top-level (or one-level-nested) folders,
    else None to signal "do your own split".
    """
    def subdir(*names):
        for name in names:
            for candidate in [root / name, *root.glob(f"*/{name}")]:
                if candidate.is_dir():
                    return candidate
        return None

    train_dir = subdir("train", "training")
    val_dir = subdir("val", "valid", "validation", "test", "testing")
    if train_dir is None or val_dir is None:
        return None

    train_pairs = list(find_labeled_pairs(train_dir))
    val_pairs = list(find_labeled_pairs(val_dir))
    if not train_pairs or not val_pairs:
        return None
    print(f"Detected existing split: train/ -> {train_dir} ({len(train_pairs)} labeled images), "
          f"val/ -> {val_dir} ({len(val_pairs)} labeled images)")
    return {"train": train_pairs, "val": val_pairs}


def make_own_split(root: Path, val_fraction: float, seed: int):
    pairs = list(find_labeled_pairs(root))
    if not pairs:
        raise SystemExit(f"No labeled image/.txt pairs found under {root} — check --source path.")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    return {"train": pairs[n_val:], "val": pairs[:n_val]}


def write_split(pairs, images_dir: Path, labels_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    seen_names = set()
    written = 0
    for img_path, label_path in pairs:
        # de-duplicate basenames across different source subfolders
        name = img_path.stem
        suffix = img_path.suffix.lower()
        candidate = f"{name}{suffix}"
        i = 1
        while candidate in seen_names:
            candidate = f"{name}_{i}{suffix}"
            i += 1
        seen_names.add(candidate)
        dest_img = images_dir / candidate
        dest_label = labels_dir / (Path(candidate).stem + ".txt")
        shutil.copy2(img_path, dest_img)
        shutil.copy2(label_path, dest_label)
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Path to the cloned dataset-pothole repo (or its 'dataset' subfolder)")
    ap.add_argument("--out", default="./yolo_dataset", help="Output directory for the prepared dataset")
    ap.add_argument("--val-fraction", type=float, default=0.15, help="Used only if no existing train/val split is detected")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"--source path does not exist or is not a directory: {source}")

    split = detect_existing_split(source)
    if split is None:
        print("No existing train/val folder structure detected — creating a reproducible split instead.")
        split = make_own_split(source, args.val_fraction, args.seed)

    n_train = write_split(split["train"], out / "images" / "train", out / "labels" / "train")
    n_val = write_split(split["val"], out / "images" / "val", out / "labels" / "val")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"train: {(out / 'images' / 'train').as_posix()}\n"
        f"val: {(out / 'images' / 'val').as_posix()}\n"
        f"nc: 1\n"
        f"names: ['pothole']\n"
    )

    print(f"\nDone.")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val}")
    print(f"  data.yaml:    {data_yaml}")
    print(f"\nNext: python3 train.py --data {data_yaml}")


if __name__ == "__main__":
    main()

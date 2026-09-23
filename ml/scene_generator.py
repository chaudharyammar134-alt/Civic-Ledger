"""
scene_generator.py — renders varied synthetic street scenes WITH ground-
truth pothole bounding boxes, for generating real training crops for
train_classical_detector.py.

This intentionally shares the visual "family" (sky/facade/lamp-post/curb/
lane-dash/asphalt-texture composition) with demo/generate_demo_images.py's
test photos, but randomizes position, size, camera shift, and noise far
more — because a classifier trained only on isolated toy patches (no
surrounding lane-dash edges, facade grids, etc.) turned out NOT to
generalize to full scenes: it fired on a lane-marking edge instead of the
actual pothole (see ml/README.md, "Why full-scene training crops"). Training
on crops taken from full rendered scenes — the same composition style the
detector will actually run against — closes that gap for real, rather than
just tuning a threshold.
"""
import cv2
import numpy as np


def render_scene(rng, w=1000, h=750, has_pothole=True,
                  pothole_fx=None, pothole_fy=None, rx=None, ry=None,
                  shift_px=0, noise_sigma=None, lamp=True):
    """
    Renders one scene. Returns (img_bgr, pothole_bbox) where pothole_bbox is
    (x, y, w, h) tightly around the pothole/patch region, or None if
    has_pothole is False.
    Randomized parameters are drawn from `rng` when not explicitly given, so
    repeated calls with different `rng` state produce meaningfully different
    scenes (position, size, lighting, camera shift).
    """
    if pothole_fx is None:
        pothole_fx = rng.uniform(0.20, 0.75)
    if pothole_fy is None:
        pothole_fy = rng.uniform(0.42, 0.68)
    if rx is None:
        rx = int(rng.integers(45, 110))
    if ry is None:
        ry = int(rng.integers(30, 70))
    if noise_sigma is None:
        noise_sigma = rng.uniform(4, 12)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (int(rng.integers(195, 230)), int(rng.integers(175, 200)), int(rng.integers(145, 175)))
    horizon = int(h * rng.uniform(0.30, 0.40))
    img[horizon:, :] = (70, 80, 85)

    facade_top, facade_bottom = int(h * 0.05), horizon
    facade_color = (int(rng.integers(105, 135)), int(rng.integers(115, 145)), int(rng.integers(125, 155)))
    cv2.rectangle(img, (0, facade_top), (w, facade_bottom), facade_color, -1)
    win_w, win_h, gap = int(rng.integers(55, 85)), int(rng.integers(70, 105)), int(rng.integers(30, 50))
    win_color = (int(rng.integers(70, 110)), int(rng.integers(150, 190)), int(rng.integers(180, 220)))
    x = 20 + shift_px
    while x < w - win_w:
        cv2.rectangle(img, (int(x), facade_top + 30), (int(x + win_w), facade_top + 30 + win_h), win_color, -1)
        cv2.rectangle(img, (int(x), facade_top + 30), (int(x + win_w), facade_top + 30 + win_h), (40, 40, 40), 3)
        x += win_w + gap

    if lamp:
        lamp_x = int(w * rng.uniform(0.65, 0.85)) + shift_px
        cv2.rectangle(img, (lamp_x, horizon - 260), (lamp_x + 14, horizon + 40), (50, 50, 50), -1)
        cv2.circle(img, (lamp_x + 7, horizon - 270), 22, (60, 200, 230), -1)

    curb_y = horizon + 40
    cv2.rectangle(img, (0, horizon), (w, curb_y), (150, 150, 150), -1)
    cv2.line(img, (0, curb_y), (w, curb_y), (30, 30, 30), 4)

    road_gray = int(rng.integers(52, 70))
    img[curb_y:, :] = (road_gray, road_gray, road_gray + 2)
    dash_y = curb_y + int((h - curb_y) * rng.uniform(0.35, 0.65))
    dash_len = int(rng.integers(40, 60))
    dash_gap = int(rng.integers(70, 110))
    for dx in range(0, w, dash_gap):
        cv2.rectangle(img, (dx + shift_px, dash_y - 6), (dx + dash_len + shift_px, dash_y + 6), (210, 210, 210), -1)

    tex = rng.normal(0, noise_sigma, size=img[curb_y:, :].shape[:2])
    for c in range(3):
        seg = img[curb_y:, :, c].astype(np.float32) + tex
        img[curb_y:, :, c] = np.clip(seg, 0, 255).astype(np.uint8)

    px = int(w * pothole_fx) + shift_px
    py = curb_y + int((h - curb_y) * pothole_fy)
    pothole_center = (px, py)
    bbox = None
    if has_pothole:
        depth = int(rng.integers(10, 35))
        cv2.ellipse(img, pothole_center, (rx, ry), int(rng.integers(0, 25)), 0, 360, (depth,) * 3, -1)
        cv2.ellipse(img, pothole_center, (rx, ry), int(rng.integers(0, 25)), 0, 360, (max(0, depth - 15),) * 3, 4)
        n_cracks = int(rng.integers(3, 8))
        for _ in range(n_cracks):
            ang = rng.uniform(0, 2 * np.pi)
            r1 = max(rx, ry) * rng.uniform(0.85, 1.0)
            r2 = r1 + rng.uniform(10, 35)
            x1, y1 = int(px + r1 * np.cos(ang)), int(py + r1 * np.sin(ang))
            x2, y2 = int(px + r2 * np.cos(ang)), int(py + r2 * np.sin(ang))
            cv2.line(img, (x1, y1), (x2, y2), (max(0, depth - 5),) * 3, 2)
        pad = int(0.25 * max(rx, ry))
        bx0, by0 = max(0, px - rx - pad), max(0, py - ry - pad)
        bx1, by1 = min(w, px + rx + pad), min(h, py + ry + pad)
        bbox = (bx0, by0, bx1 - bx0, by1 - by0)
    elif rng.random() < 0.5:
        # sometimes render an already-patched (repaired) surface instead of leaving it blank —
        # useful as a hard negative too (a repaired patch should NOT be classified as a pothole)
        px0, py0 = px - rx, py - ry
        px1, py1 = px + rx, py + ry
        patch_gray = road_gray + int(rng.integers(15, 30))
        cv2.rectangle(img, (px0, py0), (px1, py1), (patch_gray,) * 3, -1)
        cv2.rectangle(img, (px0, py0), (px1, py1), (patch_gray + 10,) * 3, 2)

    return img, bbox

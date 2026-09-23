"""
generate_demo_images.py — builds a small set of synthetic test photos so the
whole platform (and specifically the anti-gaming verification pipeline) can
be exercised end-to-end without needing real phone photos.

Produces, in demo/photos/:
  scene_a_before.jpg   - "reported pothole" photo, scene A, with EXIF GPS
  scene_a_after_honest.jpg
                        - same scene A, same viewpoint, pothole patched,
                          EXIF GPS ~5m away (normal GPS drift) -> should VERIFY
  scene_a_after_gamed.jpg
                        - a DIFFERENT scene (B) that already shows a patched
                          road, dropped in as if it were the "after" photo for
                          scene A, EXIF GPS ~600m away -> the exact attack
                          described in the brief -> should be REJECTED
  scene_a_after_same_spot_ai.jpg
                        - correct scene/GPS, but rendered with an unnaturally
                          flat/denoised look and no camera EXIF, to exercise
                          the AI-generated-image heuristic -> should FLAG

Run: python3 generate_demo_images.py
"""
import os
import numpy as np
from PIL import Image
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(7)


def _camera_noise(img, sigma=6.0):
    noise = RNG.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_street_scene(w=1000, h=750, pothole=True, shift_px=0, jitter_seed=0):
    """A stylised street scene: sky, building facade with windows (landmarks),
    a lamp post, sidewalk, and a road with lane markings. `shift_px` simulates
    a slightly different camera position (the contractor standing a bit off
    from the citizen's exact spot, which is realistic and should still pass)."""
    rng = np.random.default_rng(100 + jitter_seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # sky
    img[:, :] = (215, 190, 160)  # BGR: warm overcast sky
    horizon = int(h * 0.35)
    img[horizon:, :] = (70, 80, 85)  # road/ground base

    # building facade (upper area) — the key set of "background landmarks"
    facade_top, facade_bottom = int(h * 0.05), horizon
    cv2.rectangle(img, (0, facade_top), (w, facade_bottom), (120, 130, 140), -1)
    win_w, win_h, gap = 70, 90, 40
    x = 20 + shift_px
    while x < w - win_w:
        cv2.rectangle(img, (int(x), facade_top + 30), (int(x + win_w), facade_top + 30 + win_h), (200, 170, 90), -1)
        cv2.rectangle(img, (int(x), facade_top + 30), (int(x + win_w), facade_top + 30 + win_h), (40, 40, 40), 3)
        x += win_w + gap

    # a lamp post (distinct vertical landmark)
    lamp_x = int(w * 0.78) + shift_px
    cv2.rectangle(img, (lamp_x, horizon - 260), (lamp_x + 14, horizon + 40), (50, 50, 50), -1)
    cv2.circle(img, (lamp_x + 7, horizon - 270), 22, (60, 200, 230), -1)

    # sidewalk / curb line
    curb_y = horizon + 40
    cv2.rectangle(img, (0, horizon), (w, curb_y), (150, 150, 150), -1)
    cv2.line(img, (0, curb_y), (w, curb_y), (30, 30, 30), 4)

    # road with lane markings
    road = img[curb_y:, :]
    road[:, :] = (60, 60, 62)
    dash_y = curb_y + int((h - curb_y) * 0.5)
    for dx in range(0, w, 90):
        cv2.rectangle(img, (dx + shift_px, dash_y - 6), (dx + 50 + shift_px, dash_y + 6), (210, 210, 210), -1)

    # some road texture noise so ORB has real corners/edges to find
    tex = rng.integers(-10, 10, size=img[curb_y:, :].shape[:2])
    for c in range(3):
        seg = img[curb_y:, :, c].astype(np.int16) + tex
        img[curb_y:, :, c] = np.clip(seg, 0, 255).astype(np.uint8)

    pothole_center = (int(w * 0.42) + shift_px, curb_y + int((h - curb_y) * 0.55))
    if pothole:
        cv2.ellipse(img, pothole_center, (90, 55), 8, 0, 360, (25, 25, 30), -1)
        cv2.ellipse(img, pothole_center, (90, 55), 8, 0, 360, (10, 10, 12), 4)
        # jagged crack lines radiating out, typical of real pothole edges
        for ang in range(0, 360, 40):
            rad = np.radians(ang)
            x2 = int(pothole_center[0] + 130 * np.cos(rad))
            y2 = int(pothole_center[1] + 70 * np.sin(rad))
            cv2.line(img, pothole_center, (x2, y2), (20, 20, 22), 2)
    else:
        # freshly patched asphalt: smoother, slightly lighter, well-defined rectangle
        px0, py0 = pothole_center[0] - 100, pothole_center[1] - 62
        px1, py1 = pothole_center[0] + 100, pothole_center[1] + 62
        cv2.rectangle(img, (px0, py0), (px1, py1), (82, 82, 84), -1)
        cv2.rectangle(img, (px0, py0), (px1, py1), (95, 95, 97), 2)

    return img, pothole_center


def _pil_save_with_exif(bgr_img, path, gps_latlon=None, make_model=True, software=None):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif = Image.Exif()
    if make_model:
        exif[271] = "DemoPhone Inc."   # Make
        exif[272] = "DemoPhone X"       # Model
        exif[306] = "2026:09:10 11:32:00"  # DateTime
    if software:
        exif[305] = software
    if gps_latlon:
        lat, lon = gps_latlon
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"
        lat = abs(lat); lon = abs(lon)
        def to_dms(v):
            d = int(v); m_full = (v - d) * 60; m = int(m_full); s = (m_full - m) * 60
            return (float(d), float(m), round(s, 3))
        exif[0x8825] = {1: lat_ref, 2: to_dms(lat), 3: lon_ref, 4: to_dms(lon)}
    pil_img.save(path, quality=88, exif=exif.tobytes())
    print(f"  wrote {path}  (gps={gps_latlon}, exif={'yes' if make_model else 'NO'})")


def main():
    BASE_LAT, BASE_LON = 19.1876, 72.9789  # a point in Mumbai, purely illustrative

    print("Generating scene A (the real pothole)...")
    before_img, center = _draw_street_scene(pothole=True, shift_px=0, jitter_seed=1)
    before_img = _camera_noise(before_img, sigma=6.0)
    _pil_save_with_exif(before_img, os.path.join(OUT_DIR, "scene_a_before.jpg"),
                         gps_latlon=(BASE_LAT, BASE_LON))

    print("Generating HONEST after-photo (same scene, ~5m GPS drift, contractor standing slightly off)...")
    after_honest, _ = _draw_street_scene(pothole=False, shift_px=6, jitter_seed=2)
    after_honest = _camera_noise(after_honest, sigma=6.0)
    drift_lat = BASE_LAT + (5.0 / 111_000)   # ~5m north
    _pil_save_with_exif(after_honest, os.path.join(OUT_DIR, "scene_a_after_honest.jpg"),
                         gps_latlon=(drift_lat, BASE_LON))

    print("Generating GAMED after-photo (a totally different, already-fixed pothole elsewhere)...")
    gamed_img, _ = _draw_street_scene(pothole=False, shift_px=-40, jitter_seed=99)
    # scramble the "landmarks" so it's a genuinely different location, not just shifted
    gamed_img = np.roll(gamed_img, shift=250, axis=1)
    gamed_img = cv2.flip(gamed_img, 1)
    gamed_img = _camera_noise(gamed_img, sigma=6.0)
    far_lat = BASE_LAT + (600.0 / 111_000)   # ~600m away — a different street entirely
    _pil_save_with_exif(gamed_img, os.path.join(OUT_DIR, "scene_a_after_gamed.jpg"),
                         gps_latlon=(far_lat, BASE_LON + 0.002))

    print("Generating SAME-SPOT-BUT-SYNTHETIC-LOOKING after-photo (to trip the AI-authenticity heuristic)...")
    ai_img, _ = _draw_street_scene(pothole=False, shift_px=4, jitter_seed=3)
    # Unnaturally smooth: heavy denoise removes normal sensor noise texture,
    # and we deliberately strip camera EXIF/Make/Model - both are classic
    # (heuristic, not certain) signals of a non-camera-original image.
    ai_img = cv2.bilateralFilter(ai_img, d=15, sigmaColor=90, sigmaSpace=90)
    ai_img = cv2.bilateralFilter(ai_img, d=15, sigmaColor=90, sigmaSpace=90)
    ai_img = cv2.GaussianBlur(ai_img, (5, 5), 0)  # kill remaining sensor-noise-like texture
    _pil_save_with_exif(ai_img, os.path.join(OUT_DIR, "scene_a_after_same_spot_ai.jpg"),
                         gps_latlon=(drift_lat, BASE_LON), make_model=False,
                         software="Stable Diffusion 3.5 (demo-only synthetic marker)")

    print("\nDone. Suggested demo flow:")
    print("  1. Register a citizen, POST scene_a_before.jpg as a new complaint at "
          f"lat={BASE_LAT}, lon={BASE_LON}.")
    print("  2. Log in as admin, assign it to a contractor.")
    print("  3. As the contractor, submit scene_a_after_honest.jpg -> expect VERIFIED.")
    print("  4. On a fresh complaint, submit scene_a_after_gamed.jpg as the repair photo")
    print("     -> expect REJECTED (GPS far away AND/OR background landmarks don't match).")
    print("  5. On a fresh complaint, submit scene_a_after_same_spot_ai.jpg")
    print("     -> expect FLAGGED for manual review (AI-authenticity heuristic).")


if __name__ == "__main__":
    main()

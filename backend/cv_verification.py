"""
cv_verification.py — the anti-gaming verification pipeline.

A contractor could try to defeat a naive "does the after-photo show a fixed
road" check by photographing a *different*, already-repaired pothole. This
module defeats that specific attack by requiring the before/after pair to
share:
  1. GPS proximity (haversine distance under a tolerance)
  2. The same physical background — ORB keypoint matching + RANSAC homography
     between the two photos. A different location will simply not share
     enough matched features/landmarks with a valid homography, regardless
     of how "fixed" the road in the second photo looks.
  3. A localized change specifically inside the pothole region once the two
     photos are aligned by that homography (so the surroundings stay
     consistent while the pothole area itself changes) — this rules out
     "photoshopping" a random patch into an unrelated photo.

It also runs a set of classical image-forensics heuristics to flag likely
AI-generated / synthetic photos. These heuristics are NOT a certified
"AI-free" guarantee — no such guarantee exists with public, verifiable
techniques — they are explainable forensic signals (EXIF presence, frequency
spectrum shape, sensor-noise residual statistics, JPEG error-level analysis)
that are combined into a suspicion score and routed to human review above a
threshold, rather than silently auto-rejecting.

Every function returns plain dicts of numbers + short human-readable
explanations, because the whole point of this platform is that citizens can
see *why* something was verified, flagged, or rejected — not just a status
that changes with no visible reasoning.
"""
import io
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageChops

# ---------------------------------------------------------------------------
# Tunable thresholds (kept as module constants + exposed via config.py so a
# municipality can calibrate them for their camera hardware without touching
# the algorithm code).
# ---------------------------------------------------------------------------
GPS_TOLERANCE_M = 100.0          # phone GPS is commonly +/-10-50m in urban canyons
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.25
MAX_SCALE_RATIO = 4.0           # reject wildly different zoom/distance
SSIM_INSIDE_MAX = 0.80          # pothole region must have visibly changed (low similarity)
SSIM_OUTSIDE_MIN = 0.25         # SSIM is only a secondary check; perspective/exposure can lower it
AI_SUSPICION_FLAG_THRESHOLD = 50.0
AI_SUSPICION_HIGH_THRESHOLD = 75.0

RESIZE_MAX_DIM = 1000  # normalize image size before CV ops for speed/consistency

# Camera-only enforcement: a photo whose own EXIF capture timestamp is much
# older than "now" was not just taken with the device camera — it was pulled
# from the gallery/camera roll (or downloaded from somewhere else entirely).
# Both the citizen report photo and the contractor repair photo are checked
# against this. Generous enough to absorb upload lag over a slow connection,
# tight enough that reusing an old photo fails it by a wide margin.
PHOTO_FRESHNESS_MAX_AGE_S = 20 * 60  # 20 minutes

# Duplicate-pothole-report merging. STRICT is close enough that we merge on
# GPS alone (two people standing 8m apart are almost certainly describing the
# same hole). Between STRICT and LOOSE we additionally require the photos to
# share visual background (ORB/RANSAC, same machinery as repair verification)
# before merging, since 20m can easily contain two distinct potholes on the
# same stretch of road.
DUPLICATE_RADIUS_STRICT_M = 8.0
DUPLICATE_RADIUS_LOOSE_M = 20.0

# Repair-photo fallback: when a contractor uses a genuinely different camera
# position/frame, feature homography can fail even though the phone is at the
# reported site. Keep this fallback deliberately tighter than the general GPS
# tolerance and require visible repair evidence in the after photo.
DIRECT_REPAIR_GPS_TOLERANCE_M = 30.0


# ---------------------------------------------------------------------------
# Basic image IO helpers
# ---------------------------------------------------------------------------
def load_image_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    h, w = img.shape[:2]
    scale = RESIZE_MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# 1) GPS extraction (EXIF) + haversine distance
# ---------------------------------------------------------------------------
def _dms_to_deg(dms, ref):
    try:
        deg = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0
    except (TypeError, IndexError, ZeroDivisionError):
        return None
    if ref in ("S", "W"):
        deg = -deg
    return deg


def extract_exif_gps(path):
    """Returns (lat, lon) from EXIF GPS IFD if present, else None."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        gps_ifd = exif.get_ifd(0x8825)  # GPS IFD tag
        if not gps_ifd:
            return None
        lat = gps_ifd.get(2)
        lat_ref = gps_ifd.get(1)
        lon = gps_ifd.get(4)
        lon_ref = gps_ifd.get(3)
        if lat is None or lon is None:
            return None
        lat_deg = _dms_to_deg([float(x) for x in lat], lat_ref)
        lon_deg = _dms_to_deg([float(x) for x in lon], lon_ref)
        if lat_deg is None or lon_deg is None:
            return None
        return (lat_deg, lon_deg)
    except Exception:
        return None


def has_camera_exif(path):
    """Signal for the AI-heuristic: does the file carry ordinary camera EXIF?"""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return False
        # Tag 271 = Make, 272 = Model, 306 = DateTime, 36867 = DateTimeOriginal
        return any(t in exif for t in (271, 272, 306, 36867))
    except Exception:
        return False


def extract_exif_capture_time(path):
    """Returns a unix timestamp from EXIF DateTimeOriginal/DateTime if present, else None."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
        if not raw:
            return None
        # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
        return time.mktime(time.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S"))
    except Exception:
        return None


def check_photo_freshness(path, reference_time=None, max_age_s=PHOTO_FRESHNESS_MAX_AGE_S):
    """
    Camera-only enforcement, server side (the frontend's file picker
    `capture="environment"` hint and lastModified check are easy for a
    determined user to route around; this EXIF check cannot be spoofed
    without also rewriting the file's metadata).

    Returns {"ok": bool|None, "age_s": float|None, "explanation": str}.
    ok=None means "no EXIF capture time available" (common once a photo has
    gone through some apps' share sheets) — treated as inconclusive, not a
    failure, and folded into the AI-authenticity score instead of hard-failing
    the submission on a single missing tag.
    """
    reference_time = reference_time or time.time()
    captured_at = extract_exif_capture_time(path)
    if captured_at is None:
        return {"ok": None, "age_s": None, "explanation": "Photo carries no EXIF capture timestamp to check."}
    age = reference_time - captured_at
    if age < -300:  # allow a little clock skew, not a photo "from the future"
        return {
            "ok": False, "age_s": round(age, 1),
            "explanation": "Photo's capture timestamp is in the future relative to submission — clock mismatch or edited metadata.",
        }
    ok = age <= max_age_s
    return {
        "ok": ok, "age_s": round(age, 1),
        "explanation": (
            f"Photo was captured {age/60:.1f} min before it was submitted"
            + ("." if ok else f" — older than the {max_age_s/60:.0f}-minute freshness window; this does not look like a photo taken live with the camera just now.")
        ),
    }


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def check_gps(complaint_lat, complaint_lon, submitted_lat, submitted_lon, tolerance_m=GPS_TOLERANCE_M):
    dist = haversine_m(complaint_lat, complaint_lon, submitted_lat, submitted_lon)
    passed = dist <= tolerance_m
    return {
        "distance_m": round(dist, 1),
        "tolerance_m": tolerance_m,
        "pass": passed,
        "explanation": (
            f"Repair photo location is {dist:.1f}m from the reported pothole "
            f"(tolerance {tolerance_m:.0f}m) — {'within range' if passed else 'too far away, likely a different site'}."
        ),
    }


# ---------------------------------------------------------------------------
# 2) Pothole region auto-detection (classical CV, citizen can override in UI)
# ---------------------------------------------------------------------------
def detect_pothole_bbox(img_bgr):
    """
    Locates the pothole/damage region in a BGR image as (x, y, w, h).

    Three-tier fallback, each tier a strict upgrade over the next:
      1. Trained YOLOv8 ONNX model (backend/pothole_ml_detector.py) — best
         accuracy, but requires running the ml/ deep-learning pipeline on
         your own machine (needs torch/GPU/the real dataset) — see ml/README.md.
      2. Trained HOG+SVM model (backend/pothole_svm_detector.py) — real,
         genuinely trained scikit-learn model that ships pre-trained in the
         repo (ml/train_classical_detector.py), so this tier works out of
         the box with no training step required.
      3. Classical contour/thresholding heuristic — final fallback, needs
         no model file of any kind.
    """
    try:
        import pothole_ml_detector as ml
        ml_result = ml.detect_pothole_bbox_ml(img_bgr)
        if ml_result is not None:
            x, y, w, h, _confidence = ml_result
            return (x, y, w, h)
    except Exception as e:
        print(f"[cv_verification] YOLO/ONNX pothole detector unavailable ({e}); trying the HOG+SVM model.")

    try:
        import pothole_svm_detector as svm
        svm_result = svm.detect_pothole_bbox_svm(img_bgr)
        if svm_result is not None:
            x, y, w, h, _confidence = svm_result
            return (x, y, w, h)
    except Exception as e:
        print(f"[cv_verification] HOG+SVM pothole detector unavailable ({e}); using classical heuristic.")

    return detect_pothole_bbox_classical(img_bgr)


def detect_pothole_bbox_classical(img_bgr):
    """Classical contour/thresholding fallback (no ML model required)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_y0 = int(h * 0.25)  # road/pothole is usually in the lower part of the frame
    roi = gray[roi_y0:, :]
    thresh = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    min_area = 0.002 * w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        aspect = ww / float(hh + 1e-5)
        if aspect > 4.5 or aspect < 0.18:  # filter thin line-like artifacts (cracks, shadows)
            continue
        if area > best_area:
            best_area = area
            best = (x, y + roi_y0, ww, hh)

    if best is None:
        bw, bh = int(w * 0.25), int(h * 0.2)
        return (w // 2 - bw // 2, int(h * 0.55) - bh // 2, bw, bh)

    x, y, ww, hh = best
    pad = int(0.15 * max(ww, hh))
    x = max(0, x - pad)
    y = max(0, y - pad)
    ww = min(w - x, ww + 2 * pad)
    hh = min(h - y, hh + 2 * pad)
    return (x, y, ww, hh)


# ---------------------------------------------------------------------------
# 3) Landmark / angle matching — ORB + RANSAC homography
# ---------------------------------------------------------------------------
def feature_match_homography(img_before_bgr, img_after_bgr, bbox=None, relaxed=False):
    """Match the stable road/background, not the pothole itself.

    Repair photos naturally contain a large changed region.  The old matcher
    used every pixel/feature, so ORB could spend many of its best matches on
    the changed pothole and then reject an otherwise valid same-location pair.
    We now prefer SIFT when available, mask the reported pothole region, and
    use a slightly more tolerant RANSAC check for repair verification.
    """
    g1 = cv2.cvtColor(img_before_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_after_bgr, cv2.COLOR_BGR2GRAY)

    mask1 = np.full(g1.shape, 255, dtype=np.uint8)
    if bbox is not None:
        x, y, bw, bh = [int(v) for v in bbox]
        pad = max(8, int(0.12 * max(bw, bh)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(g1.shape[1], x + bw + pad), min(g1.shape[0], y + bh + pad)
        mask1[y0:y1, x0:x1] = 0

    # Prefer SIFT because it is much more stable than ORB when exposure,
    # perspective, and repaired surface texture change between photos.
    try:
        detector = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.02)
        k1, d1 = detector.detectAndCompute(g1, mask1)
        k2, d2 = detector.detectAndCompute(g2, None)
        norm = cv2.NORM_L2
        ratio = 0.78 if relaxed else 0.75
    except Exception:
        detector = cv2.ORB_create(nfeatures=3500, scaleFactor=1.15, nlevels=8)
        k1, d1 = detector.detectAndCompute(g1, mask1)
        k2, d2 = detector.detectAndCompute(g2, None)
        norm = cv2.NORM_HAMMING
        ratio = 0.82 if relaxed else 0.75

    result = {
        "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
        "scale_ratio": None, "pass": False, "homography": None,
        "explanation": "",
    }

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        result["explanation"] = "Not enough stable background detail was found to compare the two photos."
        return result

    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    result["good_matches"] = len(good)
    min_good = 8 if relaxed else MIN_GOOD_MATCHES
    min_inliers = 5 if relaxed else MIN_INLIERS
    min_ratio = 0.20 if relaxed else MIN_INLIER_RATIO

    if len(good) < 4:
        result["explanation"] = f"Only {len(good)} stable background matches were found."
        return result

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ransac_threshold = 6.0 if relaxed else 5.0
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)

    if H is None or mask is None:
        result["explanation"] = "Stable features were found, but they did not form a consistent geometric alignment."
        return result

    inliers = int(mask.sum())
    inlier_ratio = inliers / max(1, len(good))
    a, b = H[0, 0], H[0, 1]
    c, d = H[1, 0], H[1, 1]
    scale_ratio = float(math.sqrt(abs(a * d - b * c)) or 0)

    result["inliers"] = inliers
    result["inlier_ratio"] = round(inlier_ratio, 3)
    result["scale_ratio"] = round(scale_ratio, 3)
    result["homography"] = H.tolist()

    scale_ok = 0 < scale_ratio < (MAX_SCALE_RATIO if relaxed else 3.0) and scale_ratio > 1 / (MAX_SCALE_RATIO if relaxed else 3.0)
    passed = len(good) >= min_good and inliers >= min_inliers and inlier_ratio >= min_ratio and scale_ok
    result["pass"] = passed

    if passed:
        result["explanation"] = (
            f"Stable road/background features match: {inliers} geometric inliers "
            f"({inlier_ratio*100:.0f}% of {len(good)} matches)."
        )
    else:
        reasons = []
        if len(good) < min_good:
            reasons.append(f"too few matches ({len(good)}/{min_good})")
        if inliers < min_inliers:
            reasons.append(f"too few inliers ({inliers}/{min_inliers})")
        if inlier_ratio < min_ratio:
            reasons.append(f"low geometric consistency ({inlier_ratio*100:.0f}%)")
        if not scale_ok:
            reasons.append("camera scale changed too much")
        result["explanation"] = "Background alignment is not reliable: " + "; ".join(reasons) + "."
    return result


def photos_share_background(path_a, path_b):
    """
    Thin convenience wrapper around feature_match_homography for two photos
    that are NOT before/after of the same repair (no assumption of a repair
    having happened) — used by the duplicate-complaint merge check to decide
    whether two citizens' photos, taken a few meters apart, are actually the
    same physical pothole seen from a different angle.
    """
    try:
        img_a = load_image_bgr(path_a)
        img_b = load_image_bgr(path_b)
    except Exception:
        return False, None
    result = feature_match_homography(img_a, img_b, relaxed=False)
    return bool(result["pass"]), result


# ---------------------------------------------------------------------------
# 4) Localized repair-region change detection
# ---------------------------------------------------------------------------
def region_change_analysis(img_before_bgr, img_after_bgr, homography, bbox):
    """
    Warps the after-photo into the before-photo's frame using the homography,
    then checks that the pothole bbox region CHANGED (low similarity) while
    the rest of the shared, matched scene stayed CONSISTENT (high similarity).
    This is what rules out someone splicing an unrelated "fixed road" patch
    into a copy of the original photo.
    """
    from skimage.metrics import structural_similarity as ssim

    h, w = img_before_bgr.shape[:2]
    H = np.array(homography, dtype=np.float64)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return {"pass": False, "explanation": "Could not geometrically align the two photos to compare the repair area."}

    warped_after = cv2.warpPerspective(img_after_bgr, H_inv, (w, h))
    valid_mask = cv2.warpPerspective(
        np.ones(img_after_bgr.shape[:2], dtype=np.uint8) * 255, H_inv, (w, h)
    )

    x, y, bw, bh = bbox
    x, y = max(0, x), max(0, y)
    bw, bh = min(w - x, bw), min(h - y, bh)
    if bw <= 4 or bh <= 4:
        return {"pass": False, "explanation": "Pothole region box is invalid."}

    before_gray = cv2.cvtColor(img_before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(warped_after, cv2.COLOR_BGR2GRAY)

    inside_before = before_gray[y:y + bh, x:x + bw]
    inside_after = after_gray[y:y + bh, x:x + bw]
    inside_valid = valid_mask[y:y + bh, x:x + bw]

    if inside_valid.mean() < 200:  # after-photo did not actually cover this region once aligned
        return {
            "pass": False,
            "ssim_inside": None, "ssim_outside": None,
            "explanation": "The repair photo does not clearly cover the reported pothole area once aligned to the original.",
        }

    ssim_inside = float(ssim(inside_before, inside_after))

    # Outside region: whole overlapping area minus the pothole box, sampled on a grid for speed.
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[y:y + bh, x:x + bw] = False
    outside_mask &= valid_mask > 200
    ys, xs = np.where(outside_mask)
    if len(ys) > 4000:
        idx = np.linspace(0, len(ys) - 1, 4000).astype(int)
        ys, xs = ys[idx], xs[idx]

    if len(ys) < 50:
        ssim_outside = None
        outside_pass = True  # not enough shared background to judge; don't penalize
    else:
        # Patch-wise comparison around sampled points for a local-structure SSIM proxy.
        diffs = []
        half = 7
        for py, px in zip(ys[::4], xs[::4]):
            y0, y1 = max(0, py - half), min(h, py + half)
            x0, x1 = max(0, px - half), min(w, px + half)
            b_patch = before_gray[y0:y1, x0:x1]
            a_patch = after_gray[y0:y1, x0:x1]
            if b_patch.size < 25 or a_patch.size < 25 or b_patch.shape != a_patch.shape:
                continue
            try:
                diffs.append(ssim(b_patch, a_patch))
            except Exception:
                continue
        ssim_outside = float(np.mean(diffs)) if diffs else None
        outside_pass = ssim_outside is None or ssim_outside >= SSIM_OUTSIDE_MIN

    inside_pass = ssim_inside <= SSIM_INSIDE_MAX
    passed = inside_pass and outside_pass

    explanation_parts = []
    if inside_pass:
        explanation_parts.append(
            f"the pothole area visibly changed (similarity {ssim_inside*100:.0f}%, expected change detected)"
        )
    else:
        explanation_parts.append(
            f"the pothole area looks unchanged (similarity {ssim_inside*100:.0f}%, no repair evidence)"
        )
    if ssim_outside is not None:
        if outside_pass:
            explanation_parts.append(f"surrounding background stayed consistent ({ssim_outside*100:.0f}% similarity)")
        else:
            explanation_parts.append(
                f"surrounding background looks suspiciously different ({ssim_outside*100:.0f}% similarity) — possible tampering"
            )

    return {
        "pass": passed,
        "ssim_inside": round(ssim_inside, 3),
        "ssim_outside": round(ssim_outside, 3) if ssim_outside is not None else None,
        "explanation": "Region check: " + "; ".join(explanation_parts) + ".",
    }


# ---------------------------------------------------------------------------
# 5) AI-generated / synthetic image suspicion heuristic
# ---------------------------------------------------------------------------
def _fft_high_freq_ratio(gray):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_d = math.sqrt(cy ** 2 + cx ** 2) + 1e-6
    high_mask = dist > 0.5 * max_d
    high_energy = magnitude[high_mask].mean()
    total_energy = magnitude.mean() + 1e-6
    return float(high_energy / total_energy), magnitude


def _periodic_peak_score(magnitude):
    """Detects isolated sharp spectral peaks away from the DC region — a common
    artifact of GAN/diffusion upsampling layers acting like a regular grid filter."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    center_mask = np.zeros_like(magnitude, dtype=bool)
    r = int(0.08 * min(h, w))
    Y, X = np.ogrid[:h, :w]
    center_mask[(Y - cy) ** 2 + (X - cx) ** 2 <= r * r] = True
    outer = magnitude[~center_mask]
    thresh = outer.mean() + 4 * outer.std()
    peaks = np.sum(magnitude[~center_mask] > thresh)
    return float(peaks) / outer.size * 1000  # scaled small number


def _noise_residual_stats(gray):
    denoised = cv2.medianBlur(gray, 5)
    residual = gray.astype(np.int16) - denoised.astype(np.int16)
    return float(residual.std())


def _error_level_analysis(path):
    """Recompress at fixed JPEG quality and measure the mean amplified diff.
    Real camera photos usually show a moderate, texture-correlated ELA response;
    unusually flat (near-zero) responses can indicate an image was produced by
    a pipeline that never went through a normal camera JPEG encode."""
    try:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        resaved = Image.open(buf)
        diff = ImageChops.difference(img, resaved)
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        arr = np.array(diff).astype(np.float64) * (255.0 / max_diff)
        return float(arr.mean())
    except Exception:
        return None


def ai_generation_heuristic(path, reference_time=None):
    img = load_image_bgr(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    exif_ok = has_camera_exif(path)
    hf_ratio, magnitude = _fft_high_freq_ratio(gray)
    peak_score = _periodic_peak_score(magnitude)
    noise_std = _noise_residual_stats(gray)
    ela = _error_level_analysis(path)
    freshness = check_photo_freshness(path, reference_time=reference_time)

    score = 0.0
    signals = {}

    signals["camera_exif_present"] = exif_ok
    if not exif_ok:
        score += 25

    signals["photo_freshness"] = freshness
    if freshness["ok"] is False:
        # A stale/gallery photo is a strong, hard-to-fake signal — weighted
        # more heavily than the softer statistical signals below.
        score += 35

    signals["high_freq_energy_ratio"] = round(hf_ratio, 4)
    if hf_ratio < 0.55:  # unusually low high-frequency energy => overly smooth/synthetic texture
        score += 20

    signals["spectral_peak_score"] = round(peak_score, 4)
    if peak_score > 2.0:  # isolated periodic spectral peaks => possible upsampling grid artifact
        score += 15

    signals["noise_residual_std"] = round(noise_std, 3)
    if noise_std < 1.5:  # real camera sensor noise is rarely this low
        score += 25

    signals["error_level_mean"] = round(ela, 3) if ela is not None else None
    if ela is not None and ela < 1.0:
        score += 15

    score = min(100.0, score)
    flagged = score >= AI_SUSPICION_FLAG_THRESHOLD

    if freshness["ok"] is False:
        summary = "This photo appears to be an older file (from the gallery or elsewhere), not a fresh camera capture."
    elif score >= AI_SUSPICION_HIGH_THRESHOLD:
        summary = "Multiple forensic signals suggest this photo may not be an authentic camera capture."
    elif flagged:
        summary = "Some forensic signals are atypical for an authentic camera photo — recommend manual review."
    else:
        summary = "Forensic signals are consistent with an authentic camera photo."

    return {
        "score": round(score, 1),
        "flagged": flagged,
        "signals": signals,
        "explanation": summary + " (Note: this is a heuristic signal, not a certified proof — it flags for human review rather than auto-rejecting.)",
    }


# ---------------------------------------------------------------------------
# 5b) Different-frame repair evidence fallback
# ---------------------------------------------------------------------------
def _repair_patch_candidates(img_bgr):
    """Find large road-surface regions that look like fresh asphalt/patching.

    This is intentionally a candidate generator, not a pothole classifier.
    It combines darkness, low colourfulness and local texture/edge density so
    it can handle a contractor taking a different photo than the citizen.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Fresh asphalt is often darker than the surrounding road. Thresholds are
    # intentionally broad because cameras/exposure vary substantially.
    dark = ((val < 115) & (sat < 100)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_area = 0.015 * w * h
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.12 * w or bh < 0.08 * h:
            continue
        if bw / float(bh + 1e-6) > 7 or bh / float(bw + 1e-6) > 7:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_v = float(cv2.mean(val, mask=mask)[0])
        mean_s = float(cv2.mean(sat, mask=mask)[0])
        # Prefer substantial, dark, low-saturation road patches.
        score = (area / (w * h)) * 100.0 * max(0.0, 1.0 - mean_v / 180.0)
        candidates.append((score, (x, y, bw, bh), mean_v, mean_s))
    candidates.sort(reverse=True, key=lambda z: z[0])
    return candidates[:8]


def repair_visual_fallback(img_before_bgr, img_after_bgr, bbox):
    """Validate a repair when no reliable before/after homography exists.

    Different photos often have no common keypoints because the phone is
    moved several metres, the repaired area changes completely, or the road
    occupies a different part of the frame. In that case we compare the
    *shape/scale* of the reported damaged area with visible fresh-patch
    candidates in the after image instead of demanding pixel alignment.
    """
    h1, w1 = img_before_bgr.shape[:2]
    h2, w2 = img_after_bgr.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    before_area = max(1, bw * bh)
    before_cx = (x + bw / 2) / w1
    before_cy = (y + bh / 2) / h1
    before_ar = bw / float(bh + 1e-6)
    before_frac = before_area / float(w1 * h1)

    candidates = _repair_patch_candidates(img_after_bgr)
    if not candidates:
        return {
            "pass": False, "method": "different_frame_repair_fallback",
            "repair_candidate_count": 0,
            "explanation": "No sufficiently large fresh-road repair patch was detected in the after photo."
        }

    best = None
    for score, (ax, ay, aw, ah), mean_v, mean_s in candidates:
        acx = (ax + aw / 2) / w2
        acy = (ay + ah / 2) / h2
        aar = aw / float(ah + 1e-6)
        afrac = (aw * ah) / float(w2 * h2)

        # Camera framing can move substantially, so use broad normalized
        # geometry rather than pixel coordinates.
        center_delta = math.hypot(acx - before_cx, acy - before_cy)
        ar_similarity = min(before_ar, aar) / max(before_ar, aar, 1e-6)
        size_similarity = min(before_frac, afrac) / max(before_frac, afrac, 1e-6)
        geom_score = 0.45 * ar_similarity + 0.35 * size_similarity + 0.20 * max(0.0, 1.0 - center_delta)
        total = 0.65 * geom_score + 0.35 * min(1.0, score / 3.0)
        item = (total, (ax, ay, aw, ah), mean_v, mean_s, center_delta, ar_similarity, size_similarity)
        if best is None or item[0] > best[0]:
            best = item

    total, after_bbox, mean_v, mean_s, center_delta, ar_similarity, size_similarity = best
    passed = total >= 0.43 and ar_similarity >= 0.25 and size_similarity >= 0.18
    return {
        "pass": passed,
        "method": "different_frame_repair_fallback",
        "repair_candidate_count": len(candidates),
        "after_repair_bbox": after_bbox,
        "geometry_score": round(total * 100, 1),
        "center_delta_normalized": round(center_delta, 3),
        "bbox_aspect_similarity": round(ar_similarity, 3),
        "bbox_size_similarity": round(size_similarity, 3),
        "after_patch_mean_value": round(mean_v, 1),
        "after_patch_mean_saturation": round(mean_s, 1),
        "explanation": (
            f"Different-frame repair check {'found' if passed else 'did not find'} a plausible fresh road patch "
            f"(geometry {total*100:.0f}%, {len(candidates)} candidates)."
        ),
    }


# ---------------------------------------------------------------------------
# 6) Full pipeline orchestration
# ---------------------------------------------------------------------------
def run_full_verification(before_path, after_path, complaint_gps, submitted_gps, bbox=None, submitted_at=None):
    """
    complaint_gps / submitted_gps: (lat, lon) tuples
    bbox: (x, y, w, h) in the BEFORE image's coordinate space; auto-detected if None.
    submitted_at: unix timestamp the repair photo was submitted, used as the
        reference point for the photo-freshness check (defaults to now).
    Returns a fully explainable dict — this is what gets stored & shown publicly.
    """
    img_before = load_image_bgr(before_path)
    img_after = load_image_bgr(after_path)

    gps_result = check_gps(complaint_gps[0], complaint_gps[1], submitted_gps[0], submitted_gps[1])
    if bbox is None:
        bbox = detect_pothole_bbox(img_before)
    feature_result = feature_match_homography(img_before, img_after, bbox=bbox, relaxed=True)
    ai_result_after = ai_generation_heuristic(after_path, reference_time=submitted_at)

    different_frame_result = None
    if feature_result["pass"] and feature_result.get("homography"):
        region_result = region_change_analysis(img_before, img_after, feature_result["homography"], bbox)

        # SSIM outside the repair area is intentionally only a secondary signal.
        # Real before/after photos often differ in exposure, crop, phone position,
        # grass movement, shadows, and JPEG compression even when they are clearly
        # the same road location. If feature matching already found a strong
        # geometric alignment AND the pothole itself clearly changed, a low global
        # SSIM score must not reject an otherwise valid repair.
        strong_background_match = (
            feature_result.get("good_matches", 0) >= 12
            and feature_result.get("inliers", 0) >= 8
            and feature_result.get("inlier_ratio", 0.0) >= 0.35
        )
        if (
            not region_result.get("pass", False)
            and region_result.get("ssim_inside") is not None
            and region_result.get("ssim_inside") <= SSIM_INSIDE_MAX
            and strong_background_match
        ):
            region_result["pass"] = True
            region_result["explanation"] = (
                region_result.get("explanation", "")
                + " Strong geometric background matching confirms the same scene, so the lower outside-SSIM score was treated as a lighting/perspective difference rather than tampering."
            )
    else:
        # A different camera position can have almost no common keypoints.
        # Do not immediately reject: if GPS is tightly co-located, look for a
        # fresh repair patch with geometry consistent with the reported area.
        if gps_result["pass"] and gps_result["distance_m"] <= DIRECT_REPAIR_GPS_TOLERANCE_M:
            different_frame_result = repair_visual_fallback(img_before, img_after, bbox)
        region_result = {
            "pass": bool(different_frame_result and different_frame_result.get("pass")),
            "ssim_inside": None, "ssim_outside": None,
            "method": "different_frame_repair_fallback" if different_frame_result else "not_aligned",
            "explanation": (
                different_frame_result["explanation"]
                if different_frame_result else
                "Region check skipped — backgrounds did not match closely enough to align the photos."
            ),
        }

    # --- contractor-scam signal ------------------------------------------------
    # The specific pattern this platform was built to catch: the contractor
    # really was standing at the right spot, camera aimed the right way (GPS
    # + background landmarks both check out) — but the pothole itself shows
    # no repair evidence. That combination is not "wrong location", it's
    # "photographed at/near the pothole without actually fixing it" — the
    # clearest possible signature of gaming the verifier rather than an
    # honest mistake, so it's called out as its own explicit signal (and
    # tracked per-contractor — see handlers_admin.contractor_risk_summary)
    # rather than being folded anonymously into the generic "rejected" bucket.
    gaming_attempt = bool(
        gps_result["pass"] and feature_result["pass"]
        and region_result.get("ssim_inside") is not None and not region_result["pass"]
    )

    # --- decide overall status -------------------------------------------------
    location_and_repair_fallback = bool(
        gps_result["pass"]
        and gps_result["distance_m"] <= DIRECT_REPAIR_GPS_TOLERANCE_M
        and different_frame_result
        and different_frame_result.get("pass")
    )

    if not gps_result["pass"]:
        overall_status = "rejected"
        headline = "Rejected: the repair photo is outside the allowed GPS area."
    elif not feature_result["pass"] and not location_and_repair_fallback:
        overall_status = "rejected"
        headline = "Rejected: the repair photo could not be linked to the reported site. Move closer to the reported pothole and include the repaired area in the frame."
    elif gaming_attempt:
        overall_status = "rejected"
        headline = (
            "Rejected: the photo was taken at the right location and angle, but shows no actual repair — "
            "this looks like a photo of the still-damaged pothole, not evidence of a fix."
        )
    elif not region_result["pass"]:
        overall_status = "rejected"
        headline = "Rejected: no visible repair was detected in the reported pothole area."
    elif ai_result_after["score"] >= AI_SUSPICION_HIGH_THRESHOLD:
        overall_status = "flagged"
        headline = "Flagged for manual review: possible synthetic/AI-generated image detected."
    elif ai_result_after["flagged"]:
        overall_status = "flagged"
        headline = "Flagged for manual review: some image-authenticity signals need a human check."
    else:
        overall_status = "verified"
        headline = "Verified: location, background landmarks, and repair evidence all check out."

    # --- weighted confidence score for the dashboard ---------------------------
    gps_score = 100.0 if gps_result["pass"] else max(0.0, 100.0 - (gps_result["distance_m"] - gps_result["tolerance_m"]))
    feature_score = min(100.0, feature_result["inlier_ratio"] * 100 * 1.2) if feature_result["good_matches"] else 0.0
    if location_and_repair_fallback:
        feature_score = max(feature_score, 70.0)
    region_score = 100.0 if region_result["pass"] else 0.0
    ai_score = 100.0 - ai_result_after["score"]
    overall_confidence = round(
        0.25 * gps_score + 0.30 * feature_score + 0.30 * region_score + 0.15 * ai_score, 1
    )

    return {
        "overall_status": overall_status,
        "overall_confidence": overall_confidence,
        "headline": headline,
        "gps": gps_result,
        "landmark_match": feature_result,
        "region_change": region_result,
        "different_frame_fallback": different_frame_result,
        "ai_authenticity": ai_result_after,
        "gaming_attempt": gaming_attempt,
        "pothole_bbox_used": bbox,
    }

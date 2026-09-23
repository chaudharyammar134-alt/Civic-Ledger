import time
import json
import os

from db import get_conn, log_event, rows_to_list
import cv_verification as cvv
from utils import save_upload

REPAIR_PHOTO_DIR = os.path.join(os.path.dirname(__file__), "storage", "repairs")


def contractor_risk_summary(conn, contractor_id):
    """
    Per-contractor track record across every repair submission they've made,
    used by the admin dashboard to surface a contractor who is repeatedly
    triggering the "right location, no actual repair" gaming signal — a
    pattern an admin can't see from any single job in isolation, only from
    the history. This is a behavioural signal for admin attention, not an
    automatic penalty: a contractor is never auto-blocked by it.
    """
    rows = conn.execute(
        """SELECT vr.overall_status, vr.gaming_attempt, vr.photo_freshness_ok
           FROM verification_results vr
           JOIN repair_submissions rs ON rs.id = vr.repair_submission_id
           WHERE rs.contractor_id=?""",
        (contractor_id,),
    ).fetchall()

    total = len(rows)
    rejected = sum(1 for r in rows if r["overall_status"] == "rejected")
    gaming_attempts = sum(1 for r in rows if r["gaming_attempt"])
    stale_photos = sum(1 for r in rows if r["photo_freshness_ok"] == 0)
    rejection_rate = round(rejected / total, 3) if total else 0.0

    # A contractor needs a minimum sample size before a rate is meaningful —
    # one bad job out of one is noise, not a pattern.
    risk_level = "low"
    if total >= 3 and (gaming_attempts >= 2 or rejection_rate >= 0.5):
        risk_level = "high"
    elif total >= 2 and (gaming_attempts >= 1 or rejection_rate >= 0.34):
        risk_level = "elevated"

    return {
        "total_submissions": total,
        "rejected_submissions": rejected,
        "rejection_rate": rejection_rate,
        "gaming_attempts": gaming_attempts,
        "stale_photo_submissions": stale_photos,
        "risk_level": risk_level,
    }


def handle_contractor_jobs(ctx):
    if not ctx.user or ctx.user["role"] != "contractor":
        return ctx.send_json(403, {"error": "contractor login required"})
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM complaints WHERE assigned_contractor_id=? AND status IN ('assigned','disputed') ORDER BY assigned_at ASC",
        (ctx.user["uid"],),
    ).fetchall()
    jobs = []
    for r in rows:
        d = dict(r)
        d["before_photo_url"] = f"/media/complaints/{os.path.basename(d['before_photo_path'])}"
        d.pop("before_photo_path", None)
        if d.get("pothole_bbox"):
            d["pothole_bbox"] = json.loads(d["pothole_bbox"])
        jobs.append(d)
    ctx.send_json(200, {"jobs": jobs})


def handle_repair_submit(ctx, id):
    if not ctx.user or ctx.user["role"] != "contractor":
        return ctx.send_json(403, {"error": "contractor login required"})

    conn = get_conn()
    complaint = conn.execute("SELECT * FROM complaints WHERE id=?", (id,)).fetchone()
    if not complaint:
        return ctx.send_json(404, {"error": "complaint not found"})
    if complaint["assigned_contractor_id"] != ctx.user["uid"]:
        return ctx.send_json(403, {"error": "you are not the contractor assigned to this job"})
    if complaint["status"] not in ("assigned", "disputed"):
        return ctx.send_json(400, {"error": f"this job is not awaiting a repair submission (status: {complaint['status']})"})

    fields, files = ctx.multipart()
    if "after_photo" not in files:
        return ctx.send_json(400, {"error": "an after-repair photo is required"})
    try:
        sub_lat = float(fields.get("lat"))
        sub_lon = float(fields.get("lon"))
    except (TypeError, ValueError):
        return ctx.send_json(400, {"error": "lat/lon (device GPS at time of photo) are required"})

    after_path, _ = save_upload(files["after_photo"], REPAIR_PHOTO_DIR, prefix="after_")
    after_exif_gps = cvv.extract_exif_gps(after_path)

    now = time.time()
    cur = conn.execute(
        """INSERT INTO repair_submissions
           (complaint_id, contractor_id, after_photo_path, after_photo_exif_gps, submitted_lat, submitted_lon, submitted_at)
           VALUES (?,?,?,?,?,?,?)""",
        (id, ctx.user["uid"], after_path, json.dumps(after_exif_gps) if after_exif_gps else None, sub_lat, sub_lon, now),
    )
    conn.commit()
    submission_id = cur.lastrowid
    log_event(id, "repair_submitted", detail="Contractor uploaded after-repair photo.", actor_id=ctx.user["uid"])

    bbox = json.loads(complaint["pothole_bbox"]) if complaint["pothole_bbox"] else None

    try:
        result = cvv.run_full_verification(
            complaint["before_photo_path"], after_path,
            (complaint["lat"], complaint["lon"]), (sub_lat, sub_lon),
            bbox=bbox, submitted_at=now,
        )
    except Exception as e:
        result = {
            "overall_status": "flagged", "overall_confidence": 0.0,
            "headline": f"Automated verification could not run ({e}); routed to manual review.",
            "gps": {}, "landmark_match": {}, "region_change": {}, "ai_authenticity": {},
            "gaming_attempt": False,
            "pothole_bbox_used": bbox,
        }

    freshness_ok = result.get("ai_authenticity", {}).get("signals", {}).get("photo_freshness", {}).get("ok")

    conn.execute(
        """INSERT INTO verification_results
           (repair_submission_id, overall_status, overall_confidence,
            gps_distance_m, gps_pass, feature_good_matches, feature_inliers, feature_inlier_ratio, feature_pass,
            region_ssim_inside, region_ssim_outside, region_pass,
            ai_suspicion_score, ai_pass, explanation_json, created_at, gaming_attempt, photo_freshness_ok)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            submission_id, result["overall_status"], result["overall_confidence"],
            result.get("gps", {}).get("distance_m"), int(result.get("gps", {}).get("pass", False)),
            result.get("landmark_match", {}).get("good_matches"),
            result.get("landmark_match", {}).get("inliers"),
            result.get("landmark_match", {}).get("inlier_ratio"),
            int(result.get("landmark_match", {}).get("pass", False)),
            result.get("region_change", {}).get("ssim_inside"),
            result.get("region_change", {}).get("ssim_outside"),
            int(result.get("region_change", {}).get("pass", False)),
            result.get("ai_authenticity", {}).get("score"),
            int(not result.get("ai_authenticity", {}).get("flagged", False)),
            json.dumps(result), now,
            int(bool(result.get("gaming_attempt", False))),
            None if freshness_ok is None else int(bool(freshness_ok)),
        ),
    )

    new_status = {"verified": "verified", "rejected": "rejected", "flagged": "disputed"}[result["overall_status"]]
    # 'flagged' cases are routed into the same queue admins review as disputes, so
    # nothing sits invisibly — it always shows up somewhere in the public/admin flow.
    conn.execute("UPDATE complaints SET status=?, updated_at=? WHERE id=?", (new_status, now, id))
    conn.commit()

    log_event(
        id, f"verification_{result['overall_status']}",
        detail=result["headline"], actor_id=None,
    )

    ctx.send_json(201, {"submission_id": submission_id, "verification": result})

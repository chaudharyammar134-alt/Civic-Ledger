import time
import json
import os

import cv2

from db import get_conn, log_event, rows_to_list, row_to_dict
import cv_verification as cvv
from utils import save_upload

COMPLAINT_PHOTO_DIR = os.path.join(os.path.dirname(__file__), "storage", "complaints")
VALID_CATEGORIES = {"pothole", "waterlogging", "streetlight", "garbage", "other"}

# Complaints in these statuses are still "open" — a nearby new report of the
# same category is very likely the same physical pothole and gets merged
# into it (see _find_duplicate_target) instead of creating a second ledger
# entry. 'verified'/'resolved' are excluded on purpose: if a pothole that was
# already marked fixed gets reported again, that's a *recurrence*, not a
# duplicate, and deserves its own fresh case.
OPEN_STATUSES_FOR_DUPLICATE_MERGE = ("reported", "assigned", "repair_submitted", "disputed", "rejected")


def _find_duplicate_target(conn, category, lat, lon, new_photo_path):
    """
    Looks for an existing open complaint that is almost certainly the same
    physical pothole as the one just being reported, using location first and
    (when the two locations aren't close enough to be certain on GPS alone)
    shared visual background as a second, more precise signal — the same
    ORB/RANSAC background-matching machinery the repair-verification pipeline
    uses to prove two photos were taken in the same place.

    Returns (complaint_row, distance_m, match_method) or (None, None, None).
    """
    candidates = conn.execute(
        f"SELECT * FROM complaints WHERE category=? AND status IN "
        f"({','.join('?' * len(OPEN_STATUSES_FOR_DUPLICATE_MERGE))}) "
        f"ORDER BY created_at DESC LIMIT 200",
        (category, *OPEN_STATUSES_FOR_DUPLICATE_MERGE),
    ).fetchall()

    scored = []
    for c in candidates:
        dist = cvv.haversine_m(lat, lon, c["lat"], c["lon"])
        if dist <= cvv.DUPLICATE_RADIUS_LOOSE_M:
            scored.append((dist, c))
    scored.sort(key=lambda t: t[0])

    for dist, c in scored:
        if dist <= cvv.DUPLICATE_RADIUS_STRICT_M:
            # Close enough that two independent citizens' GPS readings landing
            # within a few meters of each other is itself strong evidence —
            # no need to also demand a visual match (angle/lighting/zoom can
            # differ enough between two strangers' phones to legitimately
            # fail ORB matching even for the exact same pothole).
            return c, dist, "gps_strict"
        # Between STRICT and LOOSE: require a visual background match before
        # merging, since this radius could plausibly contain two separate
        # potholes on the same stretch of road.
        try:
            same_scene, _ = cvv.photos_share_background(new_photo_path, c["before_photo_path"])
        except Exception:
            same_scene = False
        if same_scene:
            return c, dist, "gps_and_visual"

    return None, None, None


def handle_create_complaint(ctx):
    if not ctx.user or ctx.user["role"] != "citizen":
        return ctx.send_json(403, {"error": "only citizens can file complaints"})

    fields, files = ctx.multipart()
    description = (fields.get("description") or "").strip()
    category = (fields.get("category") or "pothole").strip().lower()
    address_text = (fields.get("address_text") or "").strip()

    try:
        lat = float(fields.get("lat"))
        lon = float(fields.get("lon"))
    except (TypeError, ValueError):
        return ctx.send_json(400, {"error": "lat/lon (from device GPS) are required"})

    if "photo" not in files:
        return ctx.send_json(400, {"error": "a photo of the pothole is required"})
    if not description or len(description) < 5:
        return ctx.send_json(400, {"error": "please add a short description"})
    if category not in VALID_CATEGORIES:
        category = "other"

    photo_path, _ = save_upload(files["photo"], COMPLAINT_PHOTO_DIR, prefix="before_")

    exif_gps = cvv.extract_exif_gps(photo_path)
    freshness = cvv.check_photo_freshness(photo_path)

    conn = get_conn()
    now = time.time()

    # --- duplicate check: is this the same pothole someone already reported? ---
    dup_target, dup_distance, dup_method = _find_duplicate_target(conn, category, lat, lon, photo_path)
    if dup_target is not None:
        conn.execute(
            """INSERT INTO duplicate_reports
               (complaint_id, citizen_id, photo_path, photo_exif_gps, lat, lon, distance_m, match_method, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                dup_target["id"], ctx.user["uid"], photo_path,
                json.dumps(exif_gps) if exif_gps else None, lat, lon,
                round(dup_distance, 1), dup_method, now,
            ),
        )
        conn.execute(
            "UPDATE complaints SET report_count = report_count + 1, updated_at=? WHERE id=?",
            (now, dup_target["id"]),
        )
        conn.commit()
        log_event(
            dup_target["id"], "duplicate_merged",
            detail=(
                f"Additional photo from another citizen merged into this case "
                f"({dup_distance:.1f}m away, {'GPS match' if dup_method == 'gps_strict' else 'GPS + matching background'})."
            ),
            actor_id=ctx.user["uid"],
        )
        return ctx.send_json(200, {
            "id": dup_target["id"], "status": dup_target["status"],
            "merged": True,
            "merged_distance_m": round(dup_distance, 1),
            "message": "This looks like the same pothole someone already reported — your photo has been added as extra evidence on that case instead of opening a duplicate.",
        })

    bbox_field = fields.get("pothole_bbox")
    bbox = None
    if bbox_field:
        try:
            b = json.loads(bbox_field)
            bbox = (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
        except Exception:
            bbox = None
    if bbox is None:
        try:
            img = cvv.load_image_bgr(photo_path)
            bbox = cvv.detect_pothole_bbox(img)
        except Exception:
            bbox = None

    cur = conn.execute(
        """INSERT INTO complaints
           (citizen_id, description, category, status, lat, lon, address_text,
            before_photo_path, before_photo_exif_gps, pothole_bbox, created_at, updated_at)
           VALUES (?,?,?, 'reported', ?,?,?, ?,?,?, ?,?)""",
        (
            ctx.user["uid"], description, category, lat, lon, address_text,
            photo_path, json.dumps(exif_gps) if exif_gps else None,
            json.dumps(bbox) if bbox else None, now, now,
        ),
    )
    conn.commit()
    complaint_id = cur.lastrowid

    detail = "Complaint filed by citizen."
    if freshness["ok"] is False:
        detail += " Note: submitted photo's timestamp suggests it was not freshly taken with the camera — flagged for admin attention."
    log_event(complaint_id, "reported", detail=detail, actor_id=ctx.user["uid"])

    ctx.send_json(201, {"id": complaint_id, "status": "reported", "merged": False})


def _complaint_public_dict(row, conn):
    d = dict(row)
    d["before_photo_url"] = f"/media/complaints/{os.path.basename(d['before_photo_path'])}"
    d.pop("before_photo_path", None)
    if d.get("pothole_bbox"):
        d["pothole_bbox"] = json.loads(d["pothole_bbox"])
    if d.get("before_photo_exif_gps"):
        d["before_photo_exif_gps"] = json.loads(d["before_photo_exif_gps"])

    citizen = conn.execute("SELECT name FROM users WHERE id=?", (d["citizen_id"],)).fetchone()
    d["citizen_name"] = citizen["name"] if citizen else "Unknown"
    if d.get("assigned_contractor_id"):
        contractor = conn.execute("SELECT name FROM users WHERE id=?", (d["assigned_contractor_id"],)).fetchone()
        d["assigned_contractor_name"] = contractor["name"] if contractor else None
    return d


def handle_list_complaints(ctx):
    conn = get_conn()
    status = ctx.query.get("status")
    category = ctx.query.get("category")
    q = "SELECT * FROM complaints"
    clauses, params = [], []
    if status:
        clauses.append("status=?")
        params.append(status)
    if category:
        clauses.append("category=?")
        params.append(category)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY created_at DESC LIMIT 500"
    rows = conn.execute(q, params).fetchall()
    complaints = [_complaint_public_dict(r, conn) for r in rows]
    ctx.send_json(200, {"complaints": complaints})


def handle_get_complaint(ctx, id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM complaints WHERE id=?", (id,)).fetchone()
    if not row:
        return ctx.send_json(404, {"error": "complaint not found"})
    d = _complaint_public_dict(row, conn)

    events = conn.execute(
        "SELECT * FROM status_events WHERE complaint_id=? ORDER BY created_at ASC", (id,)
    ).fetchall()
    d["timeline"] = rows_to_list(events)

    submissions = conn.execute(
        "SELECT * FROM repair_submissions WHERE complaint_id=? ORDER BY submitted_at DESC", (id,)
    ).fetchall()
    subs_out = []
    for s in submissions:
        sd = dict(s)
        sd["after_photo_url"] = f"/media/repairs/{os.path.basename(sd['after_photo_path'])}"
        sd.pop("after_photo_path", None)
        vr = conn.execute(
            "SELECT * FROM verification_results WHERE repair_submission_id=? ORDER BY created_at DESC LIMIT 1",
            (s["id"],),
        ).fetchone()
        if vr:
            vd = dict(vr)
            vd["explanation"] = json.loads(vd["explanation_json"])
            vd.pop("explanation_json", None)
            sd["verification"] = vd
        subs_out.append(sd)
    d["repair_submissions"] = subs_out

    reviews = conn.execute(
        "SELECT * FROM admin_reviews WHERE complaint_id=? ORDER BY created_at DESC", (id,)
    ).fetchall()
    d["admin_reviews"] = rows_to_list(reviews)

    dup_rows = conn.execute(
        "SELECT * FROM duplicate_reports WHERE complaint_id=? ORDER BY created_at ASC", (id,)
    ).fetchall()
    additional_reports = []
    for r in dup_rows:
        rd = dict(r)
        rd["photo_url"] = f"/media/complaints/{os.path.basename(rd['photo_path'])}"
        rd.pop("photo_path", None)
        citizen = conn.execute("SELECT name FROM users WHERE id=?", (rd["citizen_id"],)).fetchone()
        rd["citizen_name"] = citizen["name"] if citizen else "Unknown"
        additional_reports.append(rd)
    d["additional_reports"] = additional_reports

    ctx.send_json(200, d)


def handle_dispute_complaint(ctx, id):
    if not ctx.user:
        return ctx.send_json(401, {"error": "login required"})
    conn = get_conn()
    row = conn.execute("SELECT * FROM complaints WHERE id=?", (id,)).fetchone()
    if not row:
        return ctx.send_json(404, {"error": "complaint not found"})
    if ctx.user["uid"] != row["citizen_id"] and ctx.user["role"] != "admin":
        return ctx.send_json(403, {"error": "only the reporting citizen or an admin can dispute this complaint"})

    body = ctx.json()
    reason = (body.get("reason") or "").strip()
    if not reason:
        return ctx.send_json(400, {"error": "please explain why you are disputing the result"})

    conn.execute("UPDATE complaints SET status='disputed', updated_at=? WHERE id=?", (time.time(), id))
    conn.commit()
    log_event(id, "disputed", detail=reason, actor_id=ctx.user["uid"])
    ctx.send_json(200, {"status": "disputed"})

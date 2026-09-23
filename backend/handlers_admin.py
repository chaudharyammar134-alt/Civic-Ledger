import time
from db import get_conn, log_event, rows_to_list
from handlers_contractor import contractor_risk_summary


def _require_admin(ctx):
    if not ctx.user or ctx.user["role"] != "admin":
        ctx.send_json(403, {"error": "admin login required"})
        return False
    return True


def handle_admin_complaints(ctx):
    if not _require_admin(ctx):
        return
    conn = get_conn()
    rows = conn.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 1000").fetchall()
    ctx.send_json(200, {"complaints": rows_to_list(rows)})


def handle_admin_contractors(ctx):
    if not _require_admin(ctx):
        return
    conn = get_conn()
    rows = conn.execute("SELECT id,name,email,created_at FROM users WHERE role='contractor' ORDER BY name").fetchall()
    contractors = []
    for r in rows:
        d = dict(r)
        d.update(contractor_risk_summary(conn, d["id"]))
        contractors.append(d)
    ctx.send_json(200, {"contractors": contractors})


def handle_admin_assign(ctx, id):
    if not _require_admin(ctx):
        return
    conn = get_conn()
    complaint = conn.execute("SELECT * FROM complaints WHERE id=?", (id,)).fetchone()
    if not complaint:
        return ctx.send_json(404, {"error": "complaint not found"})
    if complaint["status"] not in ("reported",):
        return ctx.send_json(400, {"error": f"cannot assign a complaint with status '{complaint['status']}'"})

    body = ctx.json()
    contractor_id = body.get("contractor_id")
    contractor = conn.execute("SELECT * FROM users WHERE id=? AND role='contractor'", (contractor_id,)).fetchone()
    if not contractor:
        return ctx.send_json(400, {"error": "contractor not found"})

    now = time.time()
    conn.execute(
        "UPDATE complaints SET status='assigned', assigned_contractor_id=?, assigned_at=?, updated_at=? WHERE id=?",
        (contractor_id, now, now, id),
    )
    conn.commit()
    log_event(id, "assigned", detail=f"Assigned to contractor: {contractor['name']}", actor_id=ctx.user["uid"])
    ctx.send_json(200, {"status": "assigned"})


def handle_admin_review(ctx, id):
    """
    Manual decision for a complaint that needs a human call:
      - confirm_verified / confirm_rejected: settles a 'disputed' case (either
        the automated check flagged it, or a citizen disputed the result).
      - reassign: available for BOTH 'disputed' cases and, importantly,
        'rejected' ones — a rejected repair doesn't require the citizen to
        file a dispute first before the job can go back out; the admin can
        send it straight back out to a contractor. An optional contractor_id
        in the body lets the admin move the job to a *different* contractor
        (e.g. if the one who did it has a poor track record — see
        /api/admin/contractors for each contractor's rejection/gaming stats)
        instead of only ever giving the same contractor another try.
    """
    if not _require_admin(ctx):
        return
    conn = get_conn()
    complaint = conn.execute("SELECT * FROM complaints WHERE id=?", (id,)).fetchone()
    if not complaint:
        return ctx.send_json(404, {"error": "complaint not found"})

    body = ctx.json()
    decision = body.get("decision")
    note = (body.get("note") or "").strip()
    if decision not in ("confirm_verified", "confirm_rejected", "reassign"):
        return ctx.send_json(400, {"error": "decision must be one of: confirm_verified, confirm_rejected, reassign"})

    status = complaint["status"]
    now = time.time()

    if decision in ("confirm_verified", "confirm_rejected"):
        if status != "disputed":
            return ctx.send_json(400, {"error": f"'{decision}' only applies to a disputed case (this one is '{status}')"})
        new_status = "verified" if decision == "confirm_verified" else "rejected"
        conn.execute("UPDATE complaints SET status=?, updated_at=? WHERE id=?", (new_status, now, id))
        detail = f"{decision}: {note}" if note else decision

    else:  # reassign
        if status not in ("disputed", "rejected"):
            return ctx.send_json(400, {"error": f"reassign only applies to a disputed or rejected case (this one is '{status}')"})

        requested_contractor_id = body.get("contractor_id")
        if requested_contractor_id:
            contractor = conn.execute(
                "SELECT * FROM users WHERE id=? AND role='contractor'", (requested_contractor_id,)
            ).fetchone()
            if not contractor:
                return ctx.send_json(400, {"error": "contractor not found"})
        else:
            contractor = conn.execute(
                "SELECT * FROM users WHERE id=? AND role='contractor'", (complaint["assigned_contractor_id"],)
            ).fetchone()
            if not contractor:
                return ctx.send_json(400, {"error": "no contractor currently assigned — pass a contractor_id to reassign"})

        new_status = "assigned"
        conn.execute(
            "UPDATE complaints SET status=?, assigned_contractor_id=?, assigned_at=?, updated_at=? WHERE id=?",
            (new_status, contractor["id"], now, now, id),
        )
        same_contractor = contractor["id"] == complaint["assigned_contractor_id"]
        detail = (
            f"reassign: sent back to the same contractor ({contractor['name']}) for another attempt"
            if same_contractor else
            f"reassign: moved from the previous contractor to {contractor['name']}"
        )
        if note:
            detail += f" — {note}"

    conn.execute(
        "INSERT INTO admin_reviews (complaint_id, admin_id, decision, note, created_at) VALUES (?,?,?,?,?)",
        (id, ctx.user["uid"], decision, note, now),
    )
    conn.commit()
    log_event(id, "admin_review", detail=detail, actor_id=ctx.user["uid"])
    ctx.send_json(200, {"status": new_status})

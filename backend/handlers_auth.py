import time
import re
from db import get_conn, row_to_dict
import auth as auth_mod

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def handle_register(ctx):
    body = ctx.json()
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    role = body.get("role") or "citizen"

    if role not in ("citizen", "contractor"):
        return ctx.send_json(400, {"error": "role must be 'citizen' or 'contractor' (admin accounts are created by seeding, not self-registration)"})
    if not name or len(name) < 2:
        return ctx.send_json(400, {"error": "name is required"})
    if not EMAIL_RE.match(email):
        return ctx.send_json(400, {"error": "a valid email is required"})
    if len(password) < 6:
        return ctx.send_json(400, {"error": "password must be at least 6 characters"})

    conn = get_conn()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        return ctx.send_json(409, {"error": "an account with this email already exists"})

    pw_hash, salt = auth_mod.hash_password(password)
    cur = conn.execute(
        "INSERT INTO users (role,name,email,password_hash,salt,created_at) VALUES (?,?,?,?,?,?)",
        (role, name, email, pw_hash, salt, time.time()),
    )
    conn.commit()
    user_id = cur.lastrowid
    token = auth_mod.create_token(user_id, role)
    ctx.send_json(201, {
        "token": token,
        "user": {"id": user_id, "name": name, "email": email, "role": role},
    })


def handle_login(ctx):
    body = ctx.json()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not auth_mod.verify_password(password, row["password_hash"], row["salt"]):
        return ctx.send_json(401, {"error": "invalid email or password"})

    token = auth_mod.create_token(row["id"], row["role"])
    ctx.send_json(200, {
        "token": token,
        "user": {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]},
    })


def handle_me(ctx):
    if not ctx.user:
        return ctx.send_json(401, {"error": "not authenticated"})
    conn = get_conn()
    row = conn.execute("SELECT id,name,email,role FROM users WHERE id=?", (ctx.user["uid"],)).fetchone()
    ctx.send_json(200, {"user": row_to_dict(row)})

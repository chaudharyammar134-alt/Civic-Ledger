"""
app.py — the whole backend's entry point.

Deliberately built on Python's stdlib `http.server` (ThreadingHTTPServer)
instead of a web framework: this project has zero third-party runtime
dependencies beyond OpenCV/NumPy/Pillow/scikit-image/scikit-learn (which do
the actual computer-vision work). That means `pip install -r requirements.txt`
never has to reach the network for anything auth/routing/serving-related, and
there is exactly one moving part to run: `python app.py`.

Routing, auth-token extraction, multipart parsing, static file serving and
uploaded-photo serving all live here. The actual business logic lives in
handlers_auth.py / handlers_complaints.py / handlers_contractor.py /
handlers_admin.py, each of which receives a `RequestContext` (`ctx`) instead
of the raw socket handler, so they stay easy to read and to unit test.
"""
import json
import os
import re
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import db
import auth as auth_mod
from utils import parse_multipart

import handlers_auth as h_auth
import handlers_complaints as h_complaints
import handlers_contractor as h_contractor
import handlers_admin as h_admin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
COMPLAINT_MEDIA_DIR = os.path.join(BASE_DIR, "storage", "complaints")
REPAIR_MEDIA_DIR = os.path.join(BASE_DIR, "storage", "repairs")
PORT = int(os.environ.get("PORT", "8000"))

for d in (COMPLAINT_MEDIA_DIR, REPAIR_MEDIA_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Request context passed to every handler
# ---------------------------------------------------------------------------
class RequestContext:
    def __init__(self, handler, method, parsed_url, body):
        self._handler = handler
        self.method = method
        self.path = parsed_url.path
        self.query = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}
        self.body = body
        self.headers = handler.headers
        self.user = self._authenticate()

    def _authenticate(self):
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None
        token = header[len("Bearer "):].strip()
        return auth_mod.verify_token(token)

    def json(self):
        if not self.body:
            return {}
        try:
            return json.loads(self.body.decode("utf-8"))
        except Exception:
            return {}

    def multipart(self):
        ctype = self.headers.get("Content-Type", "")
        return parse_multipart(self.body, ctype)

    def send_json(self, status, obj):
        payload = json.dumps(obj, default=str).encode("utf-8")
        self._handler.send_response(status)
        self._handler.send_header("Content-Type", "application/json")
        self._handler.send_header("Content-Length", str(len(payload)))
        self._add_cors(self._handler)
        self._handler.end_headers()
        self._handler.wfile.write(payload)

    @staticmethod
    def _add_cors(handler):
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")


# ---------------------------------------------------------------------------
# Routing table: (method, compiled path regex, handler, param names)
# ---------------------------------------------------------------------------
def _route(pattern):
    """Turns '/api/complaints/<id>/dispute' into a compiled regex with named groups."""
    regex = re.sub(r"<(\w+)>", r"(?P<\1>[^/]+)", pattern)
    return re.compile("^" + regex + "$")


def handle_activity_feed(ctx):
    """Global, public, chronological feed of every status change across every
    complaint — this is the direct answer to 'complaints vanish with no
    visible follow-through': nothing here is hidden or admin-only."""
    limit = int(ctx.query.get("limit", 100))
    conn = db.get_conn()
    rows = conn.execute(
        """SELECT e.*, c.description, c.category, c.status AS complaint_status
           FROM status_events e JOIN complaints c ON c.id = e.complaint_id
           ORDER BY e.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    ctx.send_json(200, {"events": db.rows_to_list(rows)})


def handle_platform_stats(ctx):
    """Aggregate counts for the public transparency dashboard header."""
    conn = db.get_conn()
    rows = conn.execute("SELECT status, COUNT(*) as n FROM complaints GROUP BY status").fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    total = sum(counts.values())
    ctx.send_json(200, {"total": total, "by_status": counts})


ROUTES = [
    ("POST", _route("/api/auth/register"), lambda ctx, **kw: h_auth.handle_register(ctx)),
    ("POST", _route("/api/auth/login"), lambda ctx, **kw: h_auth.handle_login(ctx)),
    ("GET", _route("/api/auth/me"), lambda ctx, **kw: h_auth.handle_me(ctx)),

    ("POST", _route("/api/complaints"), lambda ctx, **kw: h_complaints.handle_create_complaint(ctx)),
    ("GET", _route("/api/complaints"), lambda ctx, **kw: h_complaints.handle_list_complaints(ctx)),
    ("GET", _route("/api/complaints/<id>"), lambda ctx, id: h_complaints.handle_get_complaint(ctx, int(id))),
    ("POST", _route("/api/complaints/<id>/dispute"), lambda ctx, id: h_complaints.handle_dispute_complaint(ctx, int(id))),
    ("POST", _route("/api/complaints/<id>/repair"), lambda ctx, id: h_contractor.handle_repair_submit(ctx, int(id))),

    ("GET", _route("/api/contractor/jobs"), lambda ctx, **kw: h_contractor.handle_contractor_jobs(ctx)),

    ("GET", _route("/api/admin/complaints"), lambda ctx, **kw: h_admin.handle_admin_complaints(ctx)),
    ("GET", _route("/api/admin/contractors"), lambda ctx, **kw: h_admin.handle_admin_contractors(ctx)),
    ("POST", _route("/api/admin/complaints/<id>/assign"), lambda ctx, id: h_admin.handle_admin_assign(ctx, int(id))),
    ("POST", _route("/api/admin/complaints/<id>/review"), lambda ctx, id: h_admin.handle_admin_review(ctx, int(id))),

    ("GET", _route("/api/activity"), lambda ctx, **kw: handle_activity_feed(ctx)),
    ("GET", _route("/api/stats"), lambda ctx, **kw: handle_platform_stats(ctx)),
]


# ---------------------------------------------------------------------------
# Static file serving (frontend + uploaded media)
# ---------------------------------------------------------------------------
def _safe_join(base, *parts):
    path = os.path.abspath(os.path.join(base, *parts))
    if not path.startswith(os.path.abspath(base)):
        return None
    return path


def _serve_file(handler, path, extra_headers=None):
    if not os.path.isfile(path):
        handler.send_response(404)
        handler.end_headers()
        return
    ctype, _ = mimetypes.guess_type(path)
    ctype = ctype or "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    RequestContext._add_cors(handler)
    if extra_headers:
        for k, v in extra_headers.items():
            handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "PotholeTracker/1.0"

    def log_message(self, fmt, *args):
        pass  # keep console clean; flip this on for debugging

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = parsed.path

        # --- uploaded photo media -------------------------------------------------
        if path.startswith("/media/complaints/"):
            fp = _safe_join(COMPLAINT_MEDIA_DIR, path[len("/media/complaints/"):])
            return _serve_file(self, fp) if fp else self._send_404()
        if path.startswith("/media/repairs/"):
            fp = _safe_join(REPAIR_MEDIA_DIR, path[len("/media/repairs/"):])
            return _serve_file(self, fp) if fp else self._send_404()

        # --- API routes --------------------------------------------------------
        if path.startswith("/api/"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            ctx = RequestContext(self, method, parsed, body)
            for route_method, regex, fn in ROUTES:
                if route_method != method:
                    continue
                m = regex.match(path)
                if m:
                    try:
                        fn(ctx, **m.groupdict())
                    except Exception as e:  # never leak a stack trace to the client
                        ctx.send_json(500, {"error": f"internal error: {e}"})
                    return
            return ctx.send_json(404, {"error": "no such API route"})

        # --- static frontend -----------------------------------------------------
        if method != "GET":
            return self._send_404()
        rel = path.lstrip("/") or "index.html"
        fp = _safe_join(FRONTEND_DIR, rel)
        if fp is None:
            return self._send_404()
        if os.path.isdir(fp):
            fp = os.path.join(fp, "index.html")
        if not os.path.isfile(fp):
            fp = os.path.join(FRONTEND_DIR, "index.html")  # SPA-style fallback
        _serve_file(self, fp)

    def _send_404(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_OPTIONS(self):
        self.send_response(204)
        RequestContext._add_cors(self)
        self.end_headers()


# ---------------------------------------------------------------------------
# First-run seeding: one admin account + a couple of demo contractors, so the
# platform is usable immediately without a separate seeding step.
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_EMAIL = "admin@municipal-tracker.local"
DEFAULT_ADMIN_PASSWORD = "admin12345"


def seed_defaults():
    conn = db.get_conn()
    existing_admin = conn.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    if not existing_admin:
        pw_hash, salt = auth_mod.hash_password(DEFAULT_ADMIN_PASSWORD)
        conn.execute(
            "INSERT INTO users (role,name,email,password_hash,salt,created_at) VALUES (?,?,?,?,?,?)",
            ("admin", "Municipal Admin", DEFAULT_ADMIN_EMAIL, pw_hash, salt, time.time()),
        )
        conn.commit()
        print(f"[seed] created default admin account: {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD}")

    existing_contractors = conn.execute("SELECT COUNT(*) AS n FROM users WHERE role='contractor'").fetchone()["n"]
    if existing_contractors == 0:
        demo_contractors = [
            ("BuildRight Civil Works", "contractor1@municipal-tracker.local"),
            ("Metro Road Solutions", "contractor2@municipal-tracker.local"),
        ]
        for name, email in demo_contractors:
            pw_hash, salt = auth_mod.hash_password("contractor123")
            conn.execute(
                "INSERT INTO users (role,name,email,password_hash,salt,created_at) VALUES (?,?,?,?,?,?)",
                ("contractor", name, email, pw_hash, salt, time.time()),
            )
        conn.commit()
        print("[seed] created 2 demo contractor accounts (password: contractor123)")


def main():
    db.init_db()
    seed_defaults()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Pothole Tracker running at http://localhost:{PORT}")
    print(f"  Admin dashboard login : {DEFAULT_ADMIN_EMAIL} / {DEFAULT_ADMIN_PASSWORD}")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

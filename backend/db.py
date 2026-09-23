"""
db.py — SQLite persistence layer. Zero external dependencies (stdlib sqlite3 only).

Schema is intentionally normalized so the public transparency dashboard can
reconstruct a full audit trail (status_events) for every complaint — this is
the direct fix for "complaints vanish with no visible follow-through".
"""
import sqlite3
import os
import threading
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "pothole_tracker.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK(role IN ('citizen','contractor','admin')),
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citizen_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'reported'
        CHECK(status IN ('reported','assigned','repair_submitted','verified','rejected','disputed','resolved')),
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    address_text TEXT,
    before_photo_path TEXT NOT NULL,
    before_photo_exif_gps TEXT,
    pothole_bbox TEXT,
    assigned_contractor_id INTEGER,
    assigned_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY(citizen_id) REFERENCES users(id),
    FOREIGN KEY(assigned_contractor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS repair_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id INTEGER NOT NULL,
    contractor_id INTEGER NOT NULL,
    after_photo_path TEXT NOT NULL,
    after_photo_exif_gps TEXT,
    submitted_lat REAL NOT NULL,
    submitted_lon REAL NOT NULL,
    submitted_at REAL NOT NULL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(id),
    FOREIGN KEY(contractor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS verification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repair_submission_id INTEGER NOT NULL,
    overall_status TEXT NOT NULL CHECK(overall_status IN ('verified','flagged','rejected')),
    overall_confidence REAL NOT NULL,
    gps_distance_m REAL,
    gps_pass INTEGER,
    feature_good_matches INTEGER,
    feature_inliers INTEGER,
    feature_inlier_ratio REAL,
    feature_pass INTEGER,
    region_ssim_inside REAL,
    region_ssim_outside REAL,
    region_pass INTEGER,
    ai_suspicion_score REAL,
    ai_pass INTEGER,
    explanation_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(repair_submission_id) REFERENCES repair_submissions(id)
);

CREATE TABLE IF NOT EXISTS status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    actor_id INTEGER,
    created_at REAL NOT NULL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(id)
);

CREATE TABLE IF NOT EXISTS admin_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id INTEGER NOT NULL,
    admin_id INTEGER NOT NULL,
    decision TEXT NOT NULL,
    note TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(id)
);

-- Extra photos/reports of a pothole that already has an open complaint.
-- Instead of creating a second row in `complaints` (which would fork the
-- public timeline and let the same physical pothole show up twice on the
-- ledger), a new report that GPS + visual matching identifies as "the same
-- hole, different angle" is folded in here as corroborating evidence on the
-- original complaint.
CREATE TABLE IF NOT EXISTS duplicate_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id INTEGER NOT NULL,
    citizen_id INTEGER NOT NULL,
    photo_path TEXT NOT NULL,
    photo_exif_gps TEXT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    distance_m REAL,
    match_method TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(id),
    FOREIGN KEY(citizen_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_complaints_lat_lon ON complaints(lat, lon);
CREATE INDEX IF NOT EXISTS idx_events_complaint ON status_events(complaint_id);
CREATE INDEX IF NOT EXISTS idx_duplicate_reports_complaint ON duplicate_reports(complaint_id);
"""

# Columns added after the initial release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so this runs each ALTER inside its own try/except — safe to call
# on both a brand-new database (schema already has none of these) and an
# existing one that predates them.
MIGRATIONS = [
    "ALTER TABLE verification_results ADD COLUMN gaming_attempt INTEGER",
    "ALTER TABLE verification_results ADD COLUMN photo_freshness_ok INTEGER",
    "ALTER TABLE complaints ADD COLUMN report_count INTEGER NOT NULL DEFAULT 1",
]


def get_conn():
    """Thread-local SQLite connection (the stdlib server is threaded)."""
    if not hasattr(_local, "conn"):
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _local.conn = conn
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    for stmt in MIGRATIONS:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def log_event(complaint_id, event_type, detail=None, actor_id=None, conn=None):
    conn = conn or get_conn()
    conn.execute(
        "INSERT INTO status_events (complaint_id, event_type, detail, actor_id, created_at) "
        "VALUES (?,?,?,?,?)",
        (complaint_id, event_type, detail, actor_id, time.time()),
    )
    conn.commit()


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows):
    return [dict(r) for r in rows]

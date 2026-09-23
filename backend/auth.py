"""
auth.py — password hashing (PBKDF2-HMAC-SHA256, stdlib hashlib) and signed,
stateless session tokens (HMAC-SHA256, stdlib hmac). No third-party auth
libraries needed.
"""
import hashlib
import hmac
import secrets
import base64
import json
import time
import os

SECRET_PATH = os.path.join(os.path.dirname(__file__), "data", "secret.key")
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
PBKDF2_ITERATIONS = 260_000


def _get_secret():
    os.makedirs(os.path.dirname(SECRET_PATH), exist_ok=True)
    if not os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, "wb") as f:
            f.write(secrets.token_bytes(32))
    with open(SECRET_PATH, "rb") as f:
        return f.read()


def hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return digest.hex(), salt.hex()


def verify_password(password: str, stored_hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    digest_hex, _ = hash_password(password, salt)
    return hmac.compare_digest(digest_hex, stored_hash_hex)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def create_token(user_id: int, role: str) -> str:
    payload = {"uid": user_id, "role": role, "exp": time.time() + TOKEN_TTL_SECONDS}
    payload_bytes = json.dumps(payload).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    sig = hmac.new(_get_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{payload_b64}.{sig_b64}"


def verify_token(token: str):
    """Returns the payload dict if valid & unexpired, else None."""
    try:
        payload_b64, sig_b64 = token.split(".")
    except (ValueError, AttributeError):
        return None
    expected_sig = hmac.new(_get_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    try:
        given_sig = _b64url_decode(sig_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, given_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload

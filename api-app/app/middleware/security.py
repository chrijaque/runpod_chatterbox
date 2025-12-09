import hmac
import hashlib
import time
from typing import Optional

from fastapi import Request, HTTPException

from ..config import settings


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def compute_signature(secret: str, method: str, path: str, timestamp: str, body: bytes) -> str:
    message = f"{method}\n{path}\n{timestamp}\n".encode("utf-8") + (body or b"")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


async def verify_hmac(request: Request) -> None:
    """Dependency to verify HMAC signature for internal endpoints."""
    if not settings.SECURITY_ENABLE_HMAC:
        return

    secret = settings.MINSTRALY_API_SHARED_SECRET
    if not secret:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing shared secret")

    ts = request.headers.get("X-Minstraly-Timestamp")
    sig = request.headers.get("X-Minstraly-Signature")
    idem = request.headers.get("X-Idempotency-Key")
    if not ts or not sig or not idem:
        raise HTTPException(status_code=401, detail="Missing auth headers")

    try:
        ts_int = int(ts)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    now_ms = int(time.time() * 1000)
    if abs(now_ms - ts_int) > settings.HMAC_MAX_SKEW_SECONDS * 1000:
        raise HTTPException(status_code=401, detail="Stale request")

    body = await request.body()
    expected = compute_signature(secret, request.method.upper(), request.url.path, ts, body)
    if not _constant_time_compare(expected, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Idempotency key guard disabled - Redis not configured
    # Proceed without idempotency enforcement


async def verify_firebase_auth(request: Request) -> Optional[str]:
    """Stub for Firebase Auth verification. Returns uid if enabled and valid."""
    if not settings.SECURITY_ENABLE_FIREBASE_AUTH:
        return None
    # TODO: Implement with firebase_admin auth.verify_id_token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1]
    # Placeholder until implemented
    return None


async def verify_app_check(request: Request) -> None:
    """Stub for Firebase App Check verification."""
    if not settings.SECURITY_ENABLE_APP_CHECK:
        return
    token = request.headers.get("X-Firebase-AppCheck")
    if not token:
        raise HTTPException(status_code=401, detail="Missing App Check token")
    # TODO: Implement App Check verification with Firebase Admin



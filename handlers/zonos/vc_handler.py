import base64
import glob
import hashlib
import hmac
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import runpod

# Configure logging (default WARNING; opt-in verbose via VERBOSE_LOGS=true)
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""
Zonos Voice Cloning handler for RunPod runtime.

Contract (mirrors chatterbox VC handler):
- Input: user_id, voice_id, name, language, is_kids_voice, audio_path (R2) or audio_data (base64)
- Output artifacts in R2:
  - private/users/{user_id}/voices/{language}/{kids/}profiles/{voice_id}.npy   (speaker embedding)
  - private/users/{user_id}/voices/{language}/{kids/}samples/{voice_id}.mp3    (preview sample)
- Calls back to the app using the existing HMAC callback scheme.
"""


# ---------------------------------------------------------------------------------
# Disk/cache management (kept intentionally minimal)
# ---------------------------------------------------------------------------------
def _ensure_cache_env_dirs() -> None:
    try:
        models_root = Path(os.getenv("MODELS_ROOT", "/models"))
        hf_root = Path(os.getenv("HF_ROOT", str(models_root / "hf")))
        torch_root = Path(os.getenv("TORCH_ROOT", str(models_root / "torch")))
        for p in [models_root, hf_root, hf_root / "hub", torch_root]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        os.environ.setdefault("HF_HOME", str(hf_root))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root))
        os.environ.setdefault("TORCH_HOME", str(torch_root))
        os.environ.setdefault("XDG_CACHE_HOME", str(models_root / "xdg"))
    except Exception:
        pass


def _safe_remove(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def cleanup_runtime_storage(*, temp_age_seconds: int = 60 * 30) -> None:
    """Best-effort cleanup to avoid filling ephemeral disk."""
    try:
        if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() != "true":
            return
        now = time.time()
        for d in [Path("/temp_voice"), Path("/voice_samples"), Path("/voice_profiles")]:
            if not d.exists():
                continue
            for entry in d.iterdir():
                try:
                    mtime = entry.stat().st_mtime if entry.exists() else now
                    if (now - mtime) > temp_age_seconds:
                        _safe_remove(entry)
                except Exception:
                    pass
    except Exception:
        pass


# Initialize cache env as early as possible
_ensure_cache_env_dirs()


# ---------------------------------------------------------------------------------
# R2 storage helpers (copied conceptually from chatterbox.storage.r2_storage)
# ---------------------------------------------------------------------------------
def _encode_metadata_value(value: str) -> str:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        return f"base64:{encoded}"


def upload_to_r2(
    data: bytes,
    destination_key: str,
    *,
    content_type: str = "application/octet-stream",
    metadata: Optional[dict] = None,
    bucket_name: Optional[str] = None,
) -> Optional[str]:
    try:
        import boto3

        r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_endpoint = os.getenv("R2_ENDPOINT")
        r2_bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "minstraly-storage")
        r2_public_url = os.getenv("NEXT_PUBLIC_R2_PUBLIC_URL") or os.getenv("R2_PUBLIC_URL")

        if not all([r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ENDPOINT)")
            return None

        s3_client = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name="auto",
        )

        extra_args: Dict[str, Any] = {"ContentType": content_type}
        if metadata:
            encoded_metadata = {str(k): _encode_metadata_value(str(v)) for k, v in metadata.items()}
            extra_args["Metadata"] = encoded_metadata

        s3_client.put_object(Bucket=r2_bucket_name, Key=destination_key, Body=data, **extra_args)
        if r2_public_url:
            return f"{r2_public_url.rstrip('/')}/{destination_key}"
        return destination_key
    except Exception as e:
        logger.error(f"❌ R2 upload failed: {e}")
        return None


def download_from_r2(source_key: str, *, bucket_name: Optional[str] = None) -> Optional[bytes]:
    try:
        import boto3

        r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_endpoint = os.getenv("R2_ENDPOINT")
        r2_bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "minstraly-storage")

        if not all([r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured (R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ENDPOINT)")
            return None

        s3_client = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name="auto",
        )
        resp = s3_client.get_object(Bucket=r2_bucket_name, Key=source_key)
        return resp["Body"].read()
    except Exception as e:
        logger.error(f"❌ R2 download failed: {e}")
        return None


# ---------------------------------------------------------------------------------
# Callback signing (same scheme as chatterbox handlers)
# ---------------------------------------------------------------------------------
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()


def _normalize_callback_url(url: str) -> str:
    try:
        p = urlparse(url)
        if APP_BASE_URL and (p.hostname in ("localhost", "127.0.0.1")):
            base = urlparse(APP_BASE_URL)
            return urlunparse((base.scheme or "https", base.netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        pass
    return url


def _canonicalize_callback_url(url: str) -> str:
    try:
        p = urlparse(url)
        scheme = p.scheme or "https"
        netloc = p.netloc
        if netloc == "minstraly.com":
            netloc = "www.minstraly.com"
        if not netloc and p.path:
            return f"https://www.minstraly.com{p.path}"
        return urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return url


def _post_signed_callback(callback_url: str, payload: Dict[str, Any], *, retries: int = 3, base_delay: float = 2.0) -> None:
    import requests

    secret = os.getenv("MINSTRALY_API_SHARED_SECRET")
    if not secret:
        raise RuntimeError("MINSTRALY_API_SHARED_SECRET not set; cannot sign callback")

    url = _canonicalize_callback_url(_normalize_callback_url(callback_url))
    parsed = urlparse(url)
    path_for_signing = parsed.path or "/api/voice-clone/callback"

    body_bytes = json.dumps(payload).encode("utf-8")
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            ts = str(int(time.time() * 1000))
            prefix = f"POST\n{path_for_signing}\n{ts}\n".encode("utf-8")
            sig = hmac.new(secret.encode("utf-8"), prefix + body_bytes, hashlib.sha256).hexdigest()
            headers = {
                "Content-Type": "application/json",
                "X-Minstraly-Timestamp": ts,
                "X-Minstraly-Signature": sig,
            }
            resp = requests.post(url, data=body_bytes, headers=headers, timeout=20)
            resp.raise_for_status()
            return
        except Exception as e:
            last_exc = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
    if last_exc:
        raise last_exc


def send_error_callback(callback_url: Optional[str], payload: Dict[str, Any]) -> None:
    if not callback_url:
        return
    try:
        _post_signed_callback(callback_url, payload, retries=3)
    except Exception as e:
        logger.error(f"❌ Failed to send error callback: {e}")


def send_success_callback(callback_url: Optional[str], payload: Dict[str, Any]) -> None:
    if not callback_url:
        return
    try:
        _post_signed_callback(callback_url, payload, retries=3)
    except Exception as e:
        logger.error(f"❌ Failed to send success callback: {e}")


# ---------------------------------------------------------------------------------
# Model init (Zonos)
# ---------------------------------------------------------------------------------
ZONOS_MODEL_ID = os.getenv("ZONOS_MODEL_ID", "Zyphra/Zonos-v0.1-transformer").strip() or "Zyphra/Zonos-v0.1-transformer"


def _select_device() -> str:
    forced = (os.getenv("VC_DEVICE") or os.getenv("MINSTRALY_DEVICE") or os.getenv("DEVICE") or "").lower().strip()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


zonos_model = None
_device = _select_device()
logger.info(f"🔧 Initializing Zonos model on device={_device} repo={ZONOS_MODEL_ID}")
try:
    from zonos.model import Zonos

    zonos_model = Zonos.from_pretrained(ZONOS_MODEL_ID, device=_device)
    logger.info("✅ Zonos model ready")
except Exception as e:
    logger.error(f"❌ Failed to initialize Zonos model: {e}")
    zonos_model = None


# ---------------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------------
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")
TEMP_VOICE_DIR = Path("/temp_voice")
for _d in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TEMP_VOICE_DIR]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
os.environ.setdefault("TMPDIR", str(TEMP_VOICE_DIR))


def _extract_metadata(input_data: Dict[str, Any]) -> Dict[str, Any]:
    meta = input_data.get("metadata", {}) if isinstance(input_data.get("metadata"), dict) else {}
    top = input_data
    return {
        "callback_url": top.get("callback_url") or meta.get("callback_url"),
        "user_id": top.get("user_id") or meta.get("user_id"),
        "voice_id": top.get("voice_id") or meta.get("voice_id"),
        "voice_name": top.get("voice_name") or meta.get("voice_name") or top.get("name") or meta.get("name"),
        "profile_filename": top.get("profile_filename") or meta.get("profile_filename"),
        "sample_filename": top.get("sample_filename") or meta.get("sample_filename"),
    }


def _infer_ext_from_r2_key(key: str) -> str:
    lower = str(key).lower()
    for ext in (".wav", ".mp3", ".ogg", ".m4a", ".webm"):
        if lower.endswith(ext):
            return ext
    return ".wav"


def _prepare_audio_file(input_data: Dict[str, Any], voice_id: str) -> Path:
    audio_path = input_data.get("audio_path")
    audio_data = input_data.get("audio_data")
    audio_format = (input_data.get("audio_format") or "wav").lower()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if audio_path:
        ext = _infer_ext_from_r2_key(audio_path)
        tmp = TEMP_VOICE_DIR / f"{voice_id}_{ts}{ext}"
        raw = download_from_r2(audio_path)
        if raw is None:
            raise RuntimeError(f"Failed to download audio from R2: {audio_path}")
        tmp.write_bytes(raw)
        return tmp

    if not audio_data:
        raise RuntimeError("Missing audio_data and audio_path")

    tmp = TEMP_VOICE_DIR / f"{voice_id}_{ts}.{audio_format}"
    tmp.write_bytes(base64.b64decode(audio_data))
    return tmp


def _transcode_to_mp3(src_path: Path, dest_path: Path) -> None:
    """
    Convert arbitrary audio into a mono 44.1kHz mp3 preview.
    Uses ffmpeg for robustness.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-b:a",
        os.getenv("VOICE_SAMPLE_BITRATE", "192k"),
        str(dest_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr[-400:]}")


def _build_profile_key(user_id: str, language: str, voice_id: str, is_kids_voice: bool) -> str:
    kids_prefix = "kids/" if is_kids_voice else ""
    return f"private/users/{user_id}/voices/{language}/{kids_prefix}profiles/{voice_id}.npy"


def _build_sample_key(user_id: str, language: str, voice_id: str, is_kids_voice: bool) -> str:
    kids_prefix = "kids/" if is_kids_voice else ""
    return f"private/users/{user_id}/voices/{language}/{kids_prefix}samples/{voice_id}.mp3"


# ---------------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------------
def handler(event, responseFormat="base64"):
    cleanup_runtime_storage()

    try:
        if zonos_model is None:
            raise RuntimeError("Zonos model not available")

        input_data = event.get("input") or {}
        if not isinstance(input_data, dict):
            raise RuntimeError("Invalid event payload: missing input dict")

        meta = _extract_metadata(input_data)
        callback_url = meta.get("callback_url") or (event.get("metadata", {}) or {}).get("callback_url")

        user_id = (meta.get("user_id") or input_data.get("user_id") or "").strip()
        name = (input_data.get("name") or meta.get("voice_name") or "").strip()
        voice_id = (meta.get("voice_id") or input_data.get("voice_id") or "").strip()
        language = (input_data.get("language") or "en").strip()
        is_kids_voice = bool(input_data.get("is_kids_voice") or False)

        if not user_id or not voice_id or not name:
            raise RuntimeError("Missing required fields: user_id, voice_id, name")

        logger.info(f"🧬 Zonos VC request voice_id={voice_id} user_id={user_id} lang={language} kids={is_kids_voice}")

        # Prepare audio
        tmp_audio = _prepare_audio_file(input_data, voice_id)

        # Compute speaker embedding
        import torch
        import torchaudio
        import numpy as np
        from io import BytesIO

        wav, sr = torchaudio.load(str(tmp_audio))
        speaker = zonos_model.make_speaker_embedding(wav, sr)  # [1, d] bfloat16 on model device
        speaker_np = speaker.detach().to("cpu", dtype=torch.float32).numpy()

        buf = BytesIO()
        np.save(buf, speaker_np, allow_pickle=False)
        profile_bytes = buf.getvalue()

        # Create preview sample mp3 from the provided recording (no TTS needed)
        tmp_mp3 = TEMP_VOICE_DIR / f"{voice_id}_{datetime.utcnow():%Y%m%d_%H%M%S}.mp3"
        _transcode_to_mp3(tmp_audio, tmp_mp3)
        sample_bytes = tmp_mp3.read_bytes()

        # Upload to R2 using standardized keys
        profile_key = _build_profile_key(user_id, language, voice_id, is_kids_voice)
        sample_key = _build_sample_key(user_id, language, voice_id, is_kids_voice)

        storage_metadata = {
            "user_id": user_id,
            "voice_id": voice_id,
            "voice_name": name,
            "language": language,
            "is_kids_voice": str(is_kids_voice).lower(),
            "model_type": "zonos",
        }

        profile_url = upload_to_r2(profile_bytes, profile_key, content_type="application/octet-stream", metadata=storage_metadata)
        if not profile_url:
            raise RuntimeError("Failed to upload profile to R2")
        sample_url = upload_to_r2(sample_bytes, sample_key, content_type="audio/mpeg", metadata=storage_metadata)
        if not sample_url:
            raise RuntimeError("Failed to upload sample to R2")

        result = {
            "status": "success",
            "user_id": user_id,
            "voice_id": voice_id,
            "voice_name": name,
            "language": language,
            "is_kids_voice": bool(is_kids_voice),
            "model_type": "zonos",
            # App callback prefers r2_* fields
            "r2_profile_path": profile_key,
            "r2_sample_path": sample_key,
            # Legacy fields (kept for backward compatibility)
            "profile_path": profile_key,
            "sample_path": sample_key,
            "recorded_path": input_data.get("audio_path") or "",
        }

        # Cleanup temp files
        try:
            tmp_audio.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            tmp_mp3.unlink(missing_ok=True)
        except Exception:
            pass

        # Callback
        if callback_url:
            send_success_callback(callback_url, result)

        cleanup_runtime_storage()
        return result

    except Exception as e:
        logger.error(f"❌ Zonos VC failed: {type(e).__name__}: {e}")
        input_data = event.get("input") or {}
        callback_url = None
        try:
            if isinstance(input_data, dict):
                meta = _extract_metadata(input_data)
                callback_url = meta.get("callback_url") or (event.get("metadata", {}) or {}).get("callback_url")
        except Exception:
            callback_url = None

        payload = {
            "status": "error",
            "error": str(e),
            "message": str(e),
            "user_id": (input_data.get("user_id") if isinstance(input_data, dict) else None),
            "voice_id": (input_data.get("voice_id") if isinstance(input_data, dict) else None),
            "voice_name": (input_data.get("name") if isinstance(input_data, dict) else None),
            "language": (input_data.get("language") if isinstance(input_data, dict) else None),
            "model_type": "zonos",
        }
        send_error_callback(callback_url, payload)
        cleanup_runtime_storage()
        return {"status": "error", "error": str(e), "model_type": "zonos"}


runpod.serverless.start({"handler": handler})


import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import runpod

# Configure logging (default WARNING; opt-in verbose via VERBOSE_LOGS=true)
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""
Zonos TTS handler for RunPod runtime.

Mirrors the Chatterbox contract:
- Input: user_id, story_id, voice_id, text, profile_path or profile_base64, language, story_type, model_type
- Chunk long text, generate per-chunk audio, stitch to a single mp3
- Upload final mp3 to R2 under:
  private/users/{user_id}/stories/audio/{language}/{story_id}/{version_id}.mp3
- POST signed callback to /api/tts/callback (and /api/tts/error-callback on failures)
"""


# ---------------------------------------------------------------------------------
# Cache + temp dirs
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
        for d in [Path("/temp_voice"), Path("/tts_generated")]:
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


_ensure_cache_env_dirs()

TEMP_VOICE_DIR = Path("/temp_voice")
TTS_GENERATED_DIR = Path("/tts_generated")
for _d in [TEMP_VOICE_DIR, TTS_GENERATED_DIR]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
os.environ.setdefault("TMPDIR", str(TEMP_VOICE_DIR))


# ---------------------------------------------------------------------------------
# R2 helpers (same approach as chatterbox.storage.r2_storage)
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
            extra_args["Metadata"] = {str(k): _encode_metadata_value(str(v)) for k, v in metadata.items()}

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


def _post_signed_callback(callback_url: str, payload: Dict[str, Any], *, retries: int = 4, base_delay: float = 5.0) -> None:
    import requests

    secret = os.getenv("MINSTRALY_API_SHARED_SECRET")
    if not secret:
        raise RuntimeError("MINSTRALY_API_SHARED_SECRET not set; cannot sign callback")

    url = _canonicalize_callback_url(_normalize_callback_url(callback_url))
    parsed = urlparse(url)
    path_for_signing = parsed.path or "/api/tts/callback"

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
            resp = requests.post(url, data=body_bytes, headers=headers, timeout=25)
            resp.raise_for_status()
            return
        except Exception as e:
            last_exc = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
    if last_exc:
        raise last_exc


def _derive_error_callback(callback_url: Optional[str]) -> Optional[str]:
    if not callback_url:
        return None
    try:
        if callback_url.endswith("/callback"):
            return callback_url.rsplit("/", 1)[0] + "/error-callback"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------------
# Zonos model init
# ---------------------------------------------------------------------------------
ZONOS_MODEL_ID = os.getenv("ZONOS_MODEL_ID", "Zyphra/Zonos-v0.1-transformer").strip() or "Zyphra/Zonos-v0.1-transformer"


def _select_device() -> str:
    forced = (os.getenv("TTS_DEVICE") or os.getenv("MINSTRALY_DEVICE") or os.getenv("DEVICE") or "").lower().strip()
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
# Language mapping (app language codes -> Zonos supported_language_codes)
# ---------------------------------------------------------------------------------
def map_language(lang: str) -> str:
    l = (lang or "en").strip().lower()
    if l == "en":
        return "en-us"
    if l == "fr":
        return "fr-fr"
    if l == "de":
        return "de"
    if l == "ja":
        return "ja"
    if l == "zh":
        # Best-effort: use yue/zh variants only if you explicitly pass them; default to zh
        return "zh"
    return l


# ---------------------------------------------------------------------------------
# Chunking + stitching helpers
# ---------------------------------------------------------------------------------
def chunk_text(text: str, *, target_chars: int = 420, max_chars: int = 600) -> List[Tuple[str, int]]:
    """
    Punctuation-aware chunking.
    Returns list of (chunk, pause_ms_after).
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # Split on paragraph boundaries first
    paras = [p.strip() for p in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n\n") if p.strip()]
    chunks: List[Tuple[str, int]] = []

    def _emit_para(p: str, is_last_para: bool) -> None:
        i = 0
        p = " ".join(p.split())
        while i < len(p):
            remaining = len(p) - i
            if remaining <= max_chars:
                chunks.append((p[i:].strip(), 600 if not is_last_para else 0))
                break
            window_end = min(len(p), i + max_chars)
            # Search breakpoints in the latter half of the window
            search_start = i + max(target_chars // 2, 1)
            search_start = min(search_start, window_end - 1)
            best = -1
            for j in range(window_end - 1, search_start - 1, -1):
                if p[j] in ".!?":
                    best = j + 1
                    break
            if best == -1:
                for j in range(window_end - 1, search_start - 1, -1):
                    if p[j] in ";:":
                        best = j + 1
                        break
            if best == -1:
                # fallback whitespace
                for j in range(window_end - 1, search_start - 1, -1):
                    if p[j].isspace():
                        best = j + 1
                        break
            if best == -1:
                best = window_end
            chunk = p[i:best].strip()
            if chunk:
                chunks.append((chunk, 250))
            i = best

        # Replace last pause for paragraph boundary
        if chunks and chunks[-1][1] == 250:
            last_text, _ = chunks[-1]
            chunks[-1] = (last_text, 600 if not is_last_para else 0)

    for idx, para in enumerate(paras):
        _emit_para(para, is_last_para=(idx == len(paras) - 1))

    # Ensure final pause is 0
    if chunks:
        last_text, _ = chunks[-1]
        chunks[-1] = (last_text, 0)
    return chunks


def _stitch_mp3(wav_paths: List[Path], pauses_ms: List[int], out_mp3: Path) -> None:
    """
    Stitch wav chunks with short silences using ffmpeg concat demuxer.
    This avoids loading full audio into Python.
    """
    if len(wav_paths) != len(pauses_ms):
        raise RuntimeError("wav_paths and pauses_ms mismatch")

    # Create silence wav files for pauses (reused by pause length)
    silence_cache: Dict[int, Path] = {}

    def _silence_wav(ms: int) -> Path:
        if ms <= 0:
            return None  # type: ignore
        if ms in silence_cache:
            return silence_cache[ms]
        p = TEMP_VOICE_DIR / f"silence_{ms}ms.wav"
        # Generate silence: 44100 Hz mono
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=mono",
            "-t",
            f"{ms/1000.0:.3f}",
            str(p),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg silence generation failed: {proc.stderr[-300:]}")
        silence_cache[ms] = p
        return p

    # Build concat list
    concat_list = TEMP_VOICE_DIR / f"concat_{out_mp3.stem}_{int(time.time())}.txt"
    lines: List[str] = []
    for wav, pause in zip(wav_paths, pauses_ms):
        lines.append(f"file '{wav.as_posix()}'")
        if pause and pause > 0:
            s = _silence_wav(pause)
            lines.append(f"file '{s.as_posix()}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:a",
        "libmp3lame",
        "-b:a",
        os.getenv("TTS_MP3_BITRATE", "192k"),
        str(out_mp3),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        concat_list.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr[-400:]}")


def _load_speaker_from_profile_bytes(profile_bytes: bytes):
    import numpy as np
    import torch

    arr = np.load(BytesIO(profile_bytes), allow_pickle=False)
    t = torch.tensor(arr, dtype=torch.float32)
    # Ensure shape [1, D]
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _build_story_audio_key(user_id: str, language: str, story_id: str, version_id: str) -> str:
    return f"private/users/{user_id}/stories/audio/{language}/{story_id}/{version_id}.mp3"


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

        api_metadata = input_data.get("metadata", {}) if isinstance(input_data.get("metadata"), dict) else {}
        callback_url = api_metadata.get("callback_url") or (event.get("metadata", {}) or {}).get("callback_url")
        error_callback_url = (
            input_data.get("error_callback_url")
            or api_metadata.get("error_callback_url")
            or _derive_error_callback(callback_url)
        )

        text = (input_data.get("text") or "").strip()
        voice_id = (input_data.get("voice_id") or api_metadata.get("voice_id") or "").strip()
        user_id = (input_data.get("user_id") or api_metadata.get("user_id") or "").strip()
        story_id = (input_data.get("story_id") or api_metadata.get("story_id") or "").strip()
        language = (input_data.get("language") or api_metadata.get("language") or "en").strip()
        story_type = (input_data.get("story_type") or api_metadata.get("story_type") or "user").strip()
        voice_name = (input_data.get("voice_name") or api_metadata.get("voice_name") or "").strip()
        output_basename = (input_data.get("output_basename") or api_metadata.get("output_basename") or "").strip()
        model_type = (input_data.get("model_type") or api_metadata.get("model_type") or "zonos").strip().lower()

        profile_base64 = input_data.get("profile_base64") or ""
        profile_path = input_data.get("profile_path") or api_metadata.get("profile_path") or ""

        if not user_id or not story_id or not voice_id or not text:
            raise RuntimeError("Missing required fields: user_id, story_id, voice_id, text")

        if model_type not in ("zonos", "zyphra"):
            # This endpoint is zonos-only; be explicit to avoid surprises.
            raise RuntimeError(f"Invalid model_type for Zonos endpoint: {model_type}")

        logger.info(f"🗣️ Zonos TTS request story_id={story_id} voice_id={voice_id} user_id={user_id} text_len={len(text)}")

        # Load speaker embedding from profile bytes
        if profile_base64:
            profile_bytes = base64.b64decode(profile_base64)
        elif profile_path:
            profile_bytes = download_from_r2(profile_path)
            if profile_bytes is None:
                raise RuntimeError(f"Failed to download profile from R2: {profile_path}")
        else:
            raise RuntimeError("Either profile_base64 or profile_path is required")

        speaker = _load_speaker_from_profile_bytes(profile_bytes)

        # Move speaker to model device/dtype
        import torch

        speaker = speaker.to(zonos_model.device, dtype=torch.bfloat16)

        # Chunk the text
        chunk_items = chunk_text(text)
        if not chunk_items:
            raise RuntimeError("Text chunking produced no chunks")

        # Generate each chunk to wav
        from zonos.conditioning import make_cond_dict
        import torchaudio

        zonos_lang = map_language(language)
        wav_paths: List[Path] = []
        pauses: List[int] = []

        for idx, (chunk, pause_ms) in enumerate(chunk_items):
            cond = make_cond_dict(text=chunk, speaker=speaker, language=zonos_lang)
            conditioning = zonos_model.prepare_conditioning(cond)
            codes = zonos_model.generate(conditioning)
            wavs = zonos_model.autoencoder.decode(codes).cpu()

            out_wav = TTS_GENERATED_DIR / f"{output_basename or story_id}_{idx:04d}.wav"
            torchaudio.save(str(out_wav), wavs[0], zonos_model.autoencoder.sampling_rate)
            wav_paths.append(out_wav)
            pauses.append(int(pause_ms))

        # Stitch to mp3
        version_id = output_basename or f"{voice_id}_{int(time.time()*1000)}"
        out_mp3 = TTS_GENERATED_DIR / f"{version_id}.mp3"
        _stitch_mp3(wav_paths, pauses, out_mp3)

        audio_bytes = out_mp3.read_bytes()
        r2_key = _build_story_audio_key(user_id, language, story_id, version_id)
        storage_meta = {
            "user_id": user_id,
            "story_id": story_id,
            "voice_id": voice_id,
            "voice_name": voice_name or voice_id,
            "language": language,
            "story_type": story_type,
            "model_type": "zonos",
        }

        audio_url = upload_to_r2(audio_bytes, r2_key, content_type="audio/mpeg", metadata=storage_meta)
        if not audio_url:
            raise RuntimeError("Failed to upload narration to R2")

        payload = {
            "story_id": story_id,
            "user_id": user_id,
            "voice_id": voice_id,
            "voice_name": voice_name or voice_id,
            "language": language,
            "audio_url": audio_url,
            "storage_path": r2_key,
            "r2_path": r2_key,
            "metadata": {
                **(api_metadata if isinstance(api_metadata, dict) else {}),
                "model_type": "zonos",
                "storagePath": r2_key,
                "r2Path": r2_key,
            },
        }

        if callback_url:
            _post_signed_callback(callback_url, payload, retries=4)

        # Cleanup generated files
        for p in wav_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            out_mp3.unlink(missing_ok=True)
        except Exception:
            pass

        cleanup_runtime_storage()
        return {"status": "success", "audio_url": audio_url, "audio_path": r2_key, "metadata": payload.get("metadata", {})}

    except Exception as e:
        logger.error(f"❌ Zonos TTS failed: {type(e).__name__}: {e}")
        input_data = event.get("input") or {}
        api_metadata = input_data.get("metadata", {}) if isinstance(input_data, dict) and isinstance(input_data.get("metadata"), dict) else {}
        callback_url = api_metadata.get("callback_url") or (event.get("metadata", {}) or {}).get("callback_url")
        error_callback_url = (
            (input_data.get("error_callback_url") if isinstance(input_data, dict) else None)
            or api_metadata.get("error_callback_url")
            or _derive_error_callback(callback_url)
        )

        err_payload = {
            "story_id": (input_data.get("story_id") if isinstance(input_data, dict) else None),
            "user_id": (input_data.get("user_id") if isinstance(input_data, dict) else None),
            "voice_id": (input_data.get("voice_id") if isinstance(input_data, dict) else None),
            "job_id": event.get("id") or event.get("requestId"),
            "error": str(e),
            "error_details": None,
            "metadata": {
                **(api_metadata if isinstance(api_metadata, dict) else {}),
                "model_type": "zonos",
            },
        }

        try:
            if error_callback_url:
                _post_signed_callback(error_callback_url, err_payload, retries=4)
        except Exception as cb_e:
            logger.error(f"❌ Error callback failed: {cb_e}")

        cleanup_runtime_storage()
        return {"status": "error", "error": str(e), "model_type": "zonos"}


runpod.serverless.start({"handler": handler})


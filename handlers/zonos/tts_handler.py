import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
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
_RE_WORD = re.compile(r"\b[\w’']+\b", re.UNICODE)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    try:
        v = int(str(raw).strip()) if raw is not None else default
    except Exception:
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _env_float(name: str, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = os.getenv(name)
    try:
        v = float(str(raw).strip()) if raw is not None else default
    except Exception:
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _normalize_text(s: str) -> str:
    return " ".join((s or "").replace("\r\n", "\n").replace("\r", "\n").split())


def _split_paragraphs(text: str) -> List[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    return paras


def _split_sentences(paragraph: str) -> List[str]:
    """
    Best-effort sentence segmentation that keeps punctuation.
    Falls back to returning the whole paragraph if no boundaries are found.
    """
    p = (paragraph or "").strip()
    if not p:
        return []

    # Normalize whitespace, but keep punctuation and quotes.
    p = re.sub(r"\s+", " ", p)

    # Split on ., !, ? (optionally followed by closing quotes/parens) when followed by whitespace.
    # Note: This is heuristic and language-agnostic; it’s good enough for narration chunking.
    parts: List[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?:[\"”’')\]]+)?\s+", p):
        end = m.end()
        sent = p[start:end].strip()
        if sent:
            parts.append(sent)
        start = end
    tail = p[start:].strip()
    if tail:
        parts.append(tail)

    if not parts:
        return [p]
    return parts


def _count_words(text: str) -> int:
    return len(_RE_WORD.findall(text or ""))


def _estimate_seconds(text: str, *, words_per_sec: float) -> float:
    wc = _count_words(text)
    if wc <= 0:
        return 0.0
    return wc / max(words_per_sec, 0.1)


def _split_long_sentence(sentence: str, *, max_words: int) -> List[str]:
    """
    Split an overlong sentence into smaller clauses without exceeding max_words.
    Prefers separators ;,: then whitespace.
    """
    s = (sentence or "").strip()
    if not s:
        return []
    words = _RE_WORD.findall(s)
    if len(words) <= max_words:
        return [s]

    # First try clause separators.
    for sep in ("; ", ": ", ", "):
        if sep in s:
            clauses = [c.strip() for c in s.split(sep) if c.strip()]
            if len(clauses) > 1:
                out: List[str] = []
                buf: List[str] = []
                buf_words = 0
                for clause in clauses:
                    cw = _count_words(clause)
                    if buf and (buf_words + cw) > max_words:
                        out.append(_normalize_text(" ".join(buf)))
                        buf = [clause]
                        buf_words = cw
                    else:
                        buf.append(clause)
                        buf_words += cw
                if buf:
                    out.append(_normalize_text(" ".join(buf)))
                if all(_count_words(x) <= max_words for x in out):
                    return out

    # Fallback: split purely by words.
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        out.append(" ".join(tokens[i : i + max_words]).strip())
        i += max_words
    return out


def chunk_text_v2_duration_aware(text: str) -> List[Tuple[str, int]]:
    """
    Duration-aware sentence packing for long-form narration.
    Returns list of (chunk_text, pause_ms_after).

    Key goals:
    - Avoid mid-sentence truncation by keeping chunks well under the model’s ~30s cap.
    - Prefer breaking on sentence boundaries, then clause boundaries, and only then whitespace.
    - Use small explicit pauses only at paragraph boundaries (punctuation handles most pauses).
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # Tunables (quality-first defaults)
    target_sec = _env_float("ZONOS_TTS_TARGET_CHUNK_SEC", 16.0, min_value=6.0, max_value=26.0)
    hard_cap_sec = _env_float("ZONOS_TTS_HARD_CAP_CHUNK_SEC", 23.5, min_value=8.0, max_value=28.5)
    words_per_sec = _env_float("ZONOS_TTS_WORDS_PER_SEC", 2.4, min_value=1.4, max_value=4.0)
    paragraph_pause_ms = _env_int("ZONOS_TTS_PARAGRAPH_PAUSE_MS", 160, min_value=0, max_value=600)

    # Convert the cap into a word cap (as a backstop for splitting a single sentence).
    hard_cap_words = max(12, int(math.floor(hard_cap_sec * words_per_sec)))

    chunks: List[Tuple[str, int]] = []
    paras = _split_paragraphs(raw)
    for p_idx, para in enumerate(paras):
        sentences = _split_sentences(para)
        if not sentences:
            continue

        buf: List[str] = []
        buf_sec = 0.0

        def flush(*, pause_ms_after: int) -> None:
            nonlocal buf, buf_sec
            if not buf:
                return
            chunk = _normalize_text(" ".join(buf))
            chunks.append((chunk, int(pause_ms_after)))
            buf = []
            buf_sec = 0.0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # If a sentence is too long, split it into smaller pieces.
            sent_parts = _split_long_sentence(sent, max_words=hard_cap_words)
            for part in sent_parts:
                part = part.strip()
                if not part:
                    continue

                part_sec = _estimate_seconds(part, words_per_sec=words_per_sec)

                # If adding this part would exceed the hard cap, flush first.
                if buf and (buf_sec + part_sec) > hard_cap_sec:
                    # Mid-paragraph chunk boundary: do NOT add paragraph pause.
                    flush(pause_ms_after=0)

                # Start a new chunk if we’re past target duration already.
                if buf and buf_sec >= target_sec:
                    # Mid-paragraph chunk boundary: do NOT add paragraph pause.
                    flush(pause_ms_after=0)

                buf.append(part)
                buf_sec += part_sec

        # End of paragraph: add a small pause (unless it’s the final paragraph).
        is_last_para = p_idx == (len(paras) - 1)
        flush(pause_ms_after=0 if is_last_para else paragraph_pause_ms)

    # Ensure final pause is 0
    if chunks:
        last_text, _ = chunks[-1]
        chunks[-1] = (last_text, 0)
    return chunks


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


def _ffmpeg_process_wav(
    in_wav: Path,
    out_wav: Path,
    *,
    trim_silence: bool,
    target_lufs: Optional[float],
    limiter_db: Optional[float],
) -> None:
    """
    Apply lightweight post-processing to improve stitching quality:
    - trim leading/trailing silence (reduces dead air and visible gaps)
    - loudness normalize (reduces chunk-to-chunk level jumps)
    - limiter (avoids clipped joins after normalization)
    """
    af_parts: List[str] = []

    if trim_silence:
        # IMPORTANT:
        # ffmpeg's silenceremove can remove *interior* “silent” regions depending on parameters and thresholds.
        # For long-form narration we only want to trim leading and trailing silence.
        # We do this by trimming the start, reversing, trimming the (new) start, and reversing back.
        thr = os.getenv("ZONOS_TTS_SILENCE_THRESHOLD_DB", "-55").strip() or "-55"
        start_dur = _env_float("ZONOS_TTS_SILENCE_START_DUR_SEC", 0.08, min_value=0.0, max_value=1.0)
        end_dur = _env_float("ZONOS_TTS_SILENCE_END_DUR_SEC", 0.12, min_value=0.0, max_value=1.5)
        # Trim leading
        lead = f"silenceremove=start_periods=1:start_duration={start_dur:.3f}:start_threshold={thr}dB"
        # Trim trailing (via reverse)
        trail = f"silenceremove=start_periods=1:start_duration={end_dur:.3f}:start_threshold={thr}dB"
        af_parts.append(f"{lead},areverse,{trail},areverse")

    if target_lufs is not None:
        tp = _env_float("ZONOS_TTS_TARGET_TRUE_PEAK_DBTP", -1.5, min_value=-10.0, max_value=-0.5)
        lra = _env_float("ZONOS_TTS_TARGET_LRA", 11.0, min_value=1.0, max_value=20.0)
        af_parts.append(f"loudnorm=I={target_lufs:.1f}:TP={tp:.1f}:LRA={lra:.1f}")

    if limiter_db is not None:
        af_parts.append(f"alimiter=limit={limiter_db:.2f}dB")

    af = ",".join(af_parts) if af_parts else "anull"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-af",
        af,
        str(out_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg postprocess failed: {proc.stderr[-500:]}")


def _linear_fade(x, fade_in: int = 0, fade_out: int = 0):
    import torch

    y = x
    n = y.shape[-1]
    if fade_in > 0 and n > 0:
        fi = min(fade_in, n)
        ramp = torch.linspace(0.0, 1.0, steps=fi, device=y.device, dtype=y.dtype)
        y[..., :fi] = y[..., :fi] * ramp
    if fade_out > 0 and n > 0:
        fo = min(fade_out, n)
        ramp = torch.linspace(1.0, 0.0, steps=fo, device=y.device, dtype=y.dtype)
        y[..., -fo:] = y[..., -fo:] * ramp
    return y


def _crossfade_tail_head(prev, nxt, crossfade_samples: int):
    import torch

    if crossfade_samples <= 0:
        return torch.cat([prev, nxt], dim=-1)
    cf = min(crossfade_samples, prev.shape[-1], nxt.shape[-1])
    if cf <= 0:
        return torch.cat([prev, nxt], dim=-1)

    a = prev[..., :-cf]
    b1 = prev[..., -cf:]
    b2 = nxt[..., :cf]
    c = nxt[..., cf:]

    ramp = torch.linspace(0.0, 1.0, steps=cf, device=prev.device, dtype=prev.dtype)
    mixed = b1 * (1.0 - ramp) + b2 * ramp
    return torch.cat([a, mixed, c], dim=-1)


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


def _build_story_audio_debug_prefix(user_id: str, language: str, story_id: str, version_id: str) -> str:
    return f"private/users/{user_id}/stories/audio/{language}/{story_id}/debug/{version_id}"


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

        # Rollout: keep old pipeline available for fast rollback
        use_v2 = _env_bool("ZONOS_LONG_TTS_V2", True)

        # Common
        version_id = output_basename or f"{voice_id}_{int(time.time()*1000)}"
        out_mp3 = TTS_GENERATED_DIR / f"{version_id}.mp3"

        if not use_v2:
            # ---------------------------
            # v1 (legacy) pipeline
            # ---------------------------
            chunk_items = chunk_text(text)
            if not chunk_items:
                raise RuntimeError("Text chunking produced no chunks")

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

            _stitch_mp3(wav_paths, pauses, out_mp3)

            # Cleanup generated files
            for p in wav_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            # ---------------------------
            # v2 (quality-first) pipeline
            # ---------------------------
            from zonos.conditioning import make_cond_dict
            import torchaudio
            import torch

            zonos_lang = map_language(language)

            # Chunk by estimated duration to avoid 30s truncation mid-sentence.
            chunk_items = chunk_text_v2_duration_aware(text)
            if not chunk_items:
                raise RuntimeError("Text chunking produced no chunks")

            # Generation controls (stable across chunks)
            speaking_rate = _env_float("ZONOS_TTS_SPEAKING_RATE", 15.0, min_value=4.0, max_value=30.0)
            pitch_std = _env_float("ZONOS_TTS_PITCH_STD", 20.0, min_value=0.0, max_value=120.0)
            fmax = _env_float("ZONOS_TTS_FMAX", 22050.0, min_value=8000.0, max_value=24000.0)
            cfg_scale = _env_float("ZONOS_TTS_CFG_SCALE", 2.0, min_value=1.05, max_value=5.0)
            min_p = _env_float("ZONOS_TTS_MIN_P", 0.10, min_value=0.0, max_value=0.5)
            # Keep emotion explicit for stability (defaults from Zonos make_cond_dict).
            emotion = [
                0.3077,
                0.0256,
                0.0256,
                0.0256,
                0.0256,
                0.0256,
                0.2564,
                0.3077,
            ]

            tokens_per_sec = _env_float("ZONOS_TTS_TOKENS_PER_SEC", 86.0, min_value=40.0, max_value=140.0)
            prefix_sec = _env_float("ZONOS_TTS_PREFIX_SEC", 1.2, min_value=0.0, max_value=3.0)
            prefix_tokens = int(round(prefix_sec * tokens_per_sec))

            # Postprocess controls
            trim_silence = _env_bool("ZONOS_TTS_TRIM_SILENCE", True)
            enable_loudnorm = _env_bool("ZONOS_TTS_LOUDNORM", True)
            target_lufs_val: Optional[float] = None
            if enable_loudnorm:
                target_lufs = os.getenv("ZONOS_TTS_TARGET_LUFS")
                if target_lufs is not None and target_lufs.strip():
                    tl = target_lufs.strip().lower()
                    if tl not in ("none", "off", "false", "disable", "disabled"):
                        try:
                            target_lufs_val = float(target_lufs)
                        except Exception:
                            target_lufs_val = -16.0
                    else:
                        target_lufs_val = None
                else:
                    target_lufs_val = -16.0
            limiter_db = _env_float("ZONOS_TTS_LIMITER_DB", -1.2, min_value=-12.0, max_value=-0.1)
            chunk_limiter = _env_bool("ZONOS_TTS_CHUNK_LIMITER", False)

            crossfade_ms = _env_int("ZONOS_TTS_CROSSFADE_MS", 80, min_value=0, max_value=250)
            sr = int(zonos_model.autoencoder.sampling_rate)
            crossfade_samples = int(round(sr * (crossfade_ms / 1000.0)))

            micro_fade_ms = _env_int("ZONOS_TTS_MICRO_FADE_MS", 10, min_value=0, max_value=50)
            micro_fade_samples = int(round(sr * (micro_fade_ms / 1000.0)))

            debug = _env_bool("ZONOS_TTS_DEBUG", False)
            debug_upload_chunks = _env_bool("ZONOS_TTS_DEBUG_UPLOAD_CHUNKS", False)
            debug_prefix = _build_story_audio_debug_prefix(user_id, language, story_id, version_id)

            deterministic = _env_bool("ZONOS_TTS_DETERMINISTIC", True)
            if deterministic:
                seed_base = int(hashlib.sha256(f"{user_id}:{story_id}:{voice_id}".encode("utf-8")).hexdigest()[:8], 16)
                torch.manual_seed(seed_base)

            # Optional text overlap (disabled by default). Use cautiously: it can introduce duplicates if misconfigured.
            words_per_sec = _env_float("ZONOS_TTS_WORDS_PER_SEC", 2.4, min_value=1.4, max_value=4.0)
            text_overlap_words = _env_int("ZONOS_TTS_TEXT_OVERLAP_WORDS", 0, min_value=0, max_value=40)
            text_overlap_with_prefix = _env_bool("ZONOS_TTS_TEXT_OVERLAP_WITH_PREFIX", False)
            prev_chunk_words: List[str] = []

            # Stitch in memory as float32 mono.
            stitched = None
            prev_codes_tail = None
            prev_pause_ms = 0

            manifest: Dict[str, Any] = {
                "version": "zonos_long_tts_v2",
                "story_id": story_id,
                "user_id": user_id,
                "voice_id": voice_id,
                "language": language,
                "sampling_rate": sr,
                "cfg_scale": cfg_scale,
                "min_p": min_p,
                "speaking_rate": speaking_rate,
                "pitch_std": pitch_std,
                "fmax": fmax,
                "prefix_sec": prefix_sec,
                "prefix_tokens": prefix_tokens,
                "crossfade_ms": crossfade_ms,
                "chunks": [],
            }

            for idx, (chunk, pause_ms_after) in enumerate(chunk_items):
                chunk_norm = _normalize_text(chunk)

                # Optional overlap: prepend last N words from previous chunk to reduce prosody resets.
                # Default is OFF to avoid accidental duplication.
                effective_overlap_words = 0
                if idx > 0 and text_overlap_words > 0 and (text_overlap_with_prefix or prefix_tokens <= 0):
                    overlap = prev_chunk_words[-text_overlap_words:] if prev_chunk_words else []
                    if overlap:
                        effective_overlap_words = len(overlap)
                        chunk_norm = _normalize_text(" ".join(overlap) + " " + chunk_norm)

                wc = _count_words(chunk_norm)
                est_sec = _estimate_seconds(chunk_norm, words_per_sec=words_per_sec)

                # Avoid truncation: size max_new_tokens based on estimate with headroom.
                # Hard cap stays at 30s worth of tokens (model default).
                est_tokens = int(math.ceil(est_sec * tokens_per_sec))
                max_new_tokens = int(min(max(est_tokens + int(tokens_per_sec * 6), int(tokens_per_sec * 10)), int(tokens_per_sec * 30)))

                if deterministic:
                    # Keep deterministic but vary by chunk index to avoid pathological repetition.
                    torch.manual_seed(seed_base + idx * 1337)

                uncond = None
                cond = make_cond_dict(
                    text=chunk_norm,
                    speaker=speaker,
                    language=zonos_lang,
                    emotion=emotion,
                    speaking_rate=speaking_rate,
                    pitch_std=pitch_std,
                    fmax=fmax,
                    unconditional_keys={"vqscore_8", "dnsmos_ovrl"},
                    device=zonos_model.device,
                )
                conditioning = zonos_model.prepare_conditioning(cond, uncond_dict=uncond)

                audio_prefix_codes = None
                used_prefix_tokens = 0
                # Only continue across chunks when we did NOT intentionally insert a paragraph pause.
                if prev_pause_ms == 0 and prev_codes_tail is not None and prefix_tokens > 0:
                    audio_prefix_codes = prev_codes_tail
                    used_prefix_tokens = int(audio_prefix_codes.shape[-1])

                t0 = time.time()
                codes = zonos_model.generate(
                    conditioning,
                    audio_prefix_codes=audio_prefix_codes,
                    max_new_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    sampling_params={"min_p": float(min_p)},
                    progress_bar=False,
                )
                gen_s = time.time() - t0

                # Keep tail codes for next chunk continuity.
                if prefix_tokens > 0:
                    prev_codes_tail = codes[..., -min(prefix_tokens, codes.shape[-1]) :].detach()
                else:
                    prev_codes_tail = None

                # IMPORTANT:
                # When we pass `audio_prefix_codes`, Zonos returns `codes` that *include* the prefix codes
                # at the start. Trimming in sample-space by decoding the prefix separately can misalign
                # (decoder context / convolution receptive field), causing joins to happen “mid chunk”.
                # Instead, drop the prefix codes in code-space *before* decoding.
                codes_for_decode = codes
                prefix_trim_samples = 0
                prefix_code_len = 0
                if audio_prefix_codes is not None and used_prefix_tokens > 0:
                    prefix_code_len = int(used_prefix_tokens)
                    if codes_for_decode.shape[-1] > prefix_code_len:
                        codes_for_decode = codes_for_decode[..., prefix_code_len:]
                    else:
                        # Extremely unlikely, but avoid decoding empty/negative-length codes.
                        codes_for_decode = codes_for_decode[..., 0:0]

                wavs = zonos_model.autoencoder.decode(codes_for_decode).cpu()
                raw_audio = wavs[0].to(torch.float32)  # [1, T] or [T] depending on decode
                if raw_audio.ndim == 1:
                    raw_audio = raw_audio.unsqueeze(0)
                raw_len = int(raw_audio.shape[-1])
                # prefix_trim_samples is now always 0 (we trim by code length instead).

                # Save raw chunk wav for postprocess/debug.
                raw_wav = TTS_GENERATED_DIR / f"{version_id}_raw_{idx:04d}.wav"
                proc_wav = TTS_GENERATED_DIR / f"{version_id}_proc_{idx:04d}.wav"
                torchaudio.save(str(raw_wav), raw_audio, sr)

                _ffmpeg_process_wav(
                    raw_wav,
                    proc_wav,
                    trim_silence=trim_silence,
                    # Do NOT loudnorm per chunk (it causes chunk-to-chunk loudness/tonal shifts).
                    target_lufs=None,
                    limiter_db=(limiter_db if chunk_limiter else None),
                )

                proc_audio, proc_sr = torchaudio.load(str(proc_wav))
                if int(proc_sr) != sr:
                    proc_audio = torchaudio.functional.resample(proc_audio, int(proc_sr), sr)
                proc_audio = proc_audio[:1, :].to(torch.float32)  # mono
                proc_len = int(proc_audio.shape[-1])

                # With precise prefix trimming, we don’t need long overlap crossfades.
                overlap_samples = 0

                # Stitch
                if stitched is None:
                    stitched = proc_audio
                else:
                    # Metrics on join candidates (before we mutate proc_audio)
                    join_metrics: Dict[str, Any] = {}
                    try:
                        win = int(round(sr * 0.10))  # 100ms windows
                        if win > 0 and stitched.shape[-1] >= win and proc_audio.shape[-1] >= win:
                            tail = stitched[..., -win:]
                            head = proc_audio[..., :win]
                            tail_rms = float(torch.sqrt(torch.mean(tail * tail)).item())
                            head_rms = float(torch.sqrt(torch.mean(head * head)).item())
                            join_metrics = {
                                "join_tail_rms_100ms": round(tail_rms, 6),
                                "join_head_rms_100ms": round(head_rms, 6),
                                "join_rms_delta_100ms": round(abs(tail_rms - head_rms), 6),
                            }
                    except Exception:
                        join_metrics = {}

                    # Insert pause after previous chunk if requested (paragraph boundary).
                    if prev_pause_ms and prev_pause_ms > 0:
                        pause_samples = int(round(sr * (prev_pause_ms / 1000.0)))
                        if pause_samples > 0:
                            silence = torch.zeros((1, pause_samples), dtype=stitched.dtype)
                            # small fade to avoid clicks around inserted silence
                            stitched = _linear_fade(stitched, fade_out=micro_fade_samples)
                            stitched = torch.cat([stitched, silence], dim=-1)
                            proc_audio = _linear_fade(proc_audio, fade_in=micro_fade_samples)

                        # When we intentionally inserted silence, avoid overlap mixing; just drop duplicated prefix.
                        if overlap_samples > 0 and proc_audio.shape[-1] > overlap_samples:
                            proc_audio = proc_audio[..., overlap_samples:]
                        stitched = torch.cat([stitched, proc_audio], dim=-1)
                    else:
                        # Always use a short fixed crossfade for smooth joins.
                        stitched = _crossfade_tail_head(stitched, proc_audio, crossfade_samples)

                prev_pause_ms = int(pause_ms_after or 0)

                # Debug/metrics
                chunk_info = {
                    "index": idx,
                    "word_count": wc,
                    "estimated_sec": round(est_sec, 3),
                    "pause_ms_after": int(pause_ms_after or 0),
                    "max_new_tokens": int(max_new_tokens),
                    "gen_seconds": round(gen_s, 3),
                    "codes_len": int(codes.shape[-1]),
                    "used_prefix_tokens": int(used_prefix_tokens),
                    "prefix_code_len": int(prefix_code_len),
                    "prefix_trim_samples": int(prefix_trim_samples),
                    "overlap_samples": int(overlap_samples),
                    "text_overlap_words": int(effective_overlap_words),
                    "raw_samples": int(raw_len),
                    "processed_samples": int(proc_len),
                    "raw_seconds": round(raw_len / float(sr), 3) if sr else None,
                    "processed_seconds": round(proc_len / float(sr), 3) if sr else None,
                }
                if idx > 0 and join_metrics:
                    chunk_info.update(join_metrics)
                manifest["chunks"].append(chunk_info)
                if _VERBOSE_LOGS or debug:
                    logger.info(f"🧩 chunk={idx} words={wc} est={est_sec:.1f}s codes={codes.shape[-1]} prefix={used_prefix_tokens} overlap={overlap_samples} gen={gen_s:.1f}s")

                # Debug uploads
                if debug and debug_upload_chunks:
                    try:
                        upload_to_r2(raw_wav.read_bytes(), f"{debug_prefix}/chunks/raw_{idx:04d}.wav", content_type="audio/wav")
                        upload_to_r2(proc_wav.read_bytes(), f"{debug_prefix}/chunks/proc_{idx:04d}.wav", content_type="audio/wav")
                    except Exception as up_e:
                        logger.error(f"⚠️ Debug chunk upload failed: {up_e}")

                # Cleanup chunk files (keep if debug)
                if not debug:
                    try:
                        raw_wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        proc_wav.unlink(missing_ok=True)
                    except Exception:
                        pass

                # Track words for potential overlap on next chunk (based on *original* chunk, not overlapped text).
                prev_chunk_words = _RE_WORD.findall(_normalize_text(chunk))

            if stitched is None:
                raise RuntimeError("No audio produced")

            # Final micro-fade out to avoid clicks at end.
            stitched = _linear_fade(stitched, fade_out=micro_fade_samples)

            final_wav = TTS_GENERATED_DIR / f"{version_id}.wav"
            torchaudio.save(str(final_wav), stitched.cpu(), sr)

            # Encode MP3 once at the end.
            final_af_parts: List[str] = []
            if enable_loudnorm and target_lufs_val is not None:
                tp = _env_float("ZONOS_TTS_TARGET_TRUE_PEAK_DBTP", -1.5, min_value=-10.0, max_value=-0.5)
                lra = _env_float("ZONOS_TTS_TARGET_LRA", 11.0, min_value=1.0, max_value=20.0)
                final_af_parts.append(f"loudnorm=I={target_lufs_val:.1f}:TP={tp:.1f}:LRA={lra:.1f}")
            # Always keep a final limiter when loudnorm is enabled, to avoid overs after normalization.
            if enable_loudnorm and limiter_db is not None:
                final_af_parts.append(f"alimiter=limit={limiter_db:.2f}dB")
            final_af = ",".join(final_af_parts) if final_af_parts else None

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(final_wav),
            ]
            if final_af:
                cmd += ["-af", final_af]
            cmd += [
                "-c:a",
                "libmp3lame",
                "-b:a",
                os.getenv("TTS_MP3_BITRATE", "192k"),
                str(out_mp3),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg mp3 encode failed: {proc.stderr[-500:]}")

            if debug:
                try:
                    upload_to_r2(json.dumps(manifest, ensure_ascii=True, indent=2).encode("utf-8"), f"{debug_prefix}/manifest.json", content_type="application/json")
                except Exception as m_e:
                    logger.error(f"⚠️ Debug manifest upload failed: {m_e}")

            # Cleanup final wav unless debugging
            if not debug:
                try:
                    final_wav.unlink(missing_ok=True)
                except Exception:
                    pass

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


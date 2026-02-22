import runpod
import time  
import os
import base64
import logging
import sys
import glob
import pathlib
import shutil
import requests
import hmac
import hashlib
import inspect
import random
from urllib.parse import urlparse, urlunparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging (default WARNING; opt-in verbose via VERBOSE_LOGS=true)
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""Minimal, production-focused TTS handler for RunPod runtime."""

EXPERIMENT_ENV_KEYS = (
    "VERBOSE_LOGS",
    "CHATTERBOX_EXPERIMENT_MODE",
    "CHATTERBOX_EXPERIMENT_NAME",
    "CHATTERBOX_EXPERIMENT_ISSUE_ONLY_MODE",
    "CHATTERBOX_EXPERIMENT_ENABLE_TOKEN_GUARDS",
    "CHATTERBOX_EXPERIMENT_ENABLE_SILENCE_GATE",
    "CHATTERBOX_EXPERIMENT_ENABLE_QA_REGEN",
    "CHATTERBOX_EXPERIMENT_ENABLE_RETRY_PARAM_DRIFT",
    "CHATTERBOX_EXPERIMENT_ENABLE_ADAPTIVE_VOICE_PARAMS",
    "CHATTERBOX_EXPERIMENT_FORCE_ADAPTIVE_BLEND",
    "CHATTERBOX_EXPERIMENT_VERBOSE_CHUNK_LOGS",
    "CHATTERBOX_QA_REGEN_MODE",
    "CHATTERBOX_ENABLE_QUALITY_ANALYSIS",
    "CHATTERBOX_FAIL_ON_BAD_CHUNK",
    "CHATTERBOX_CHUNK_REGEN_ATTEMPTS",
)


def _experiment_env_snapshot() -> dict:
    """Collect experiment-related env vars from THIS worker process."""
    return {k: os.getenv(k, "<unset>") for k in EXPERIMENT_ENV_KEYS}


def _log_worker_experiment_context(context: str, *, api_metadata: Optional[dict] = None, input_payload: Optional[dict] = None) -> None:
    """
    High-signal diagnostics for experiment mode.
    Uses warning level so logs are visible even when VERBOSE_LOGS=false.
    """
    try:
        env_snapshot = _experiment_env_snapshot()
        logger.warning("🧪 [%s] Worker experiment env snapshot: %s", context, env_snapshot)
        logger.warning("🧪 [%s] Worker logger level=%s (VERBOSE_LOGS=%s)", context, logging.getLevelName(logger.level), os.getenv("VERBOSE_LOGS", "<unset>"))
        if isinstance(input_payload, dict):
            logger.warning(
                "🧪 [%s] Input experiment hints: mode=%s name=%s issue_only=%s qa_mode=%s",
                context,
                input_payload.get("CHATTERBOX_EXPERIMENT_MODE"),
                input_payload.get("CHATTERBOX_EXPERIMENT_NAME"),
                input_payload.get("CHATTERBOX_EXPERIMENT_ISSUE_ONLY_MODE"),
                input_payload.get("CHATTERBOX_QA_REGEN_MODE"),
            )
        if isinstance(api_metadata, dict):
            logger.warning(
                "🧪 [%s] Metadata experiment hints: mode=%s name=%s issue_only=%s qa_mode=%s",
                context,
                api_metadata.get("CHATTERBOX_EXPERIMENT_MODE"),
                api_metadata.get("CHATTERBOX_EXPERIMENT_NAME"),
                api_metadata.get("CHATTERBOX_EXPERIMENT_ISSUE_ONLY_MODE"),
                api_metadata.get("CHATTERBOX_QA_REGEN_MODE"),
            )
    except Exception as e:
        logger.warning("🧪 [%s] Failed to log worker experiment context: %s", context, e)


def _env_true(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _patch_t3_inference_diagnostics(model) -> None:
    """
    Runtime safety net: ensure experiment logs and progress suppression even if
    the packaged chatterbox.tts version is older than expected.
    """
    try:
        if not model or not hasattr(model, "t3") or not hasattr(model.t3, "inference"):
            return
        if getattr(model.t3.inference, "_minstraly_exp_wrapped", False):
            return

        original_inference = model.t3.inference

        def _wrapped_inference(*args, **kwargs):
            experiment_on = _env_true("CHATTERBOX_EXPERIMENT_MODE", False)
            if experiment_on:
                show_progress = _env_true("CHATTERBOX_EXPERIMENT_SHOW_SAMPLING_PROGRESS", False)
                kwargs["show_progress"] = show_progress
                logger.warning(
                    "🧪 T3 inference call | forced_show_progress=%s max_new_tokens=%s",
                    show_progress,
                    kwargs.get("max_new_tokens"),
                )

            result = original_inference(*args, **kwargs)

            if experiment_on:
                try:
                    shape = getattr(result, "shape", None)
                    seq_len = int(shape[-1]) if shape is not None and len(shape) >= 1 else -1
                    logger.warning(
                        "🧪 T3 raw token output | shape=%s seq_len=%s",
                        tuple(shape) if shape is not None else None,
                        seq_len,
                    )
                except Exception as diag_e:
                    logger.warning("🧪 T3 raw token diagnostics failed: %s", diag_e)
            return result

        _wrapped_inference._minstraly_exp_wrapped = True
        model.t3.inference = _wrapped_inference
        logger.warning("🧪 Installed T3 inference diagnostics wrapper")
    except Exception as e:
        logger.warning("🧪 Failed to install T3 inference diagnostics wrapper: %s", e)


def _patch_chunk_context_tracking(model) -> None:
    """
    Track current chunk text/id around _generate_with_prepared_conditionals so
    downstream wrappers (e.g., vocoder diagnostics) can attribute failures to
    specific chunk text spans.
    """
    try:
        if not model or not hasattr(model, "_generate_with_prepared_conditionals"):
            return
        original = model._generate_with_prepared_conditionals
        if getattr(original, "_minstraly_chunk_ctx_wrapped", False):
            return

        def _wrapped_generate_with_ctx(*args, **kwargs):
            text = kwargs.get("text")
            if text is None and len(args) >= 1:
                text = args[0]
            diagnostics_chunk_id = kwargs.get("diagnostics_chunk_id")

            chunk_ctx = {
                "chunk_id": diagnostics_chunk_id,
                "text_len": len(text) if isinstance(text, str) else -1,
                "text_preview": (text[:220] if isinstance(text, str) else ""),
                "text_full": (text if isinstance(text, str) else ""),
            }
            setattr(model, "_minstraly_chunk_ctx", chunk_ctx)

            # Optional deterministic mode for reproducible debugging.
            fixed_seed = os.getenv("CHATTERBOX_EXPERIMENT_FIXED_SEED")
            if fixed_seed not in (None, ""):
                base_seed = _env_int("CHATTERBOX_EXPERIMENT_FIXED_SEED", 1337)
                chunk_offset = diagnostics_chunk_id if isinstance(diagnostics_chunk_id, int) else 0
                seed = base_seed + int(chunk_offset)
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
                try:
                    random.seed(seed)
                except Exception:
                    pass
                try:
                    import numpy as _np
                    _np.random.seed(seed)
                except Exception:
                    pass
                logger.warning(
                    "🧪 Fixed-seed applied | base=%s chunk_offset=%s seed=%s chunk_id=%s",
                    base_seed,
                    chunk_offset,
                    seed,
                    diagnostics_chunk_id,
                )

            try:
                return original(*args, **kwargs)
            finally:
                setattr(model, "_minstraly_chunk_ctx", None)

        _wrapped_generate_with_ctx._minstraly_chunk_ctx_wrapped = True
        model._generate_with_prepared_conditionals = _wrapped_generate_with_ctx
        logger.warning("🧪 Installed chunk-context tracking wrapper")
    except Exception as e:
        logger.warning("🧪 Failed to install chunk-context tracking wrapper: %s", e)


def _patch_s3_inference_diagnostics(model) -> None:
    """
    Runtime wrapper for vocoder output diagnostics.
    Logs peak/RMS and can hard-fail on effectively silent output to prove/deny
    the vocoder-silence theory with hard evidence.
    """
    try:
        if not model or not hasattr(model, "s3gen") or not hasattr(model.s3gen, "inference"):
            return
        if getattr(model.s3gen.inference, "_minstraly_s3_exp_wrapped", False):
            return

        original_inference = model.s3gen.inference

        def _wrapped_s3_inference(*args, **kwargs):
            result = original_inference(*args, **kwargs)

            experiment_on = _env_true("CHATTERBOX_EXPERIMENT_MODE", False)
            if not experiment_on:
                return result

            wav = result[0] if isinstance(result, tuple) and len(result) > 0 else result
            token_count = -1
            chunk_id = None
            text_len = -1
            text_preview = ""
            text_full = ""
            try:
                speech_tokens = kwargs.get("speech_tokens")
                if speech_tokens is not None and hasattr(speech_tokens, "numel"):
                    token_count = int(speech_tokens.numel())
            except Exception:
                token_count = -1
            try:
                ctx = getattr(model, "_minstraly_chunk_ctx", None) or {}
                chunk_id = ctx.get("chunk_id")
                text_len = int(ctx.get("text_len", -1))
                text_preview = str(ctx.get("text_preview", ""))
                text_full = str(ctx.get("text_full", ""))
            except Exception:
                pass

            try:
                import torch

                if wav is None or not hasattr(wav, "numel") or int(wav.numel()) == 0:
                    samples = 0
                    peak = 0.0
                    rms = 0.0
                    silence_ratio = 1.0
                    longest_silence_sec = 0.0
                    longest_internal_silence_sec = 0.0
                else:
                    wav_detached = wav.detach()
                    samples = int(wav_detached.numel())
                    peak = float(torch.max(torch.abs(wav_detached)).item()) if samples else 0.0
                    rms = float(torch.sqrt(torch.mean(wav_detached.float() ** 2)).item()) if samples else 0.0

                    # Intra-chunk silence analysis catches "mid-chunk silent spans"
                    # that global peak/RMS cannot detect.
                    wav_flat = wav_detached.float().reshape(-1)
                    sr = int(getattr(model, "sr", 24000))
                    win = max(1, int(sr * 0.025))  # 25ms
                    hop = max(1, int(sr * 0.010))  # 10ms
                    if int(wav_flat.numel()) >= win:
                        frames = wav_flat.unfold(0, win, hop)
                        frame_rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
                    else:
                        frame_rms = torch.sqrt(torch.mean(wav_flat * wav_flat) + 1e-12).reshape(1)

                    frame_silence_threshold = float(
                        os.getenv("CHATTERBOX_EXPERIMENT_FRAME_SILENCE_RMS_THRESHOLD", "5e-4")
                    )
                    silence_mask = frame_rms < frame_silence_threshold
                    silence_ratio = float(silence_mask.float().mean().item()) if int(silence_mask.numel()) else 0.0

                    mask_list = silence_mask.tolist() if hasattr(silence_mask, "tolist") else []

                    def _longest_run(seq):
                        best = 0
                        cur = 0
                        for v in seq:
                            if v:
                                cur += 1
                                if cur > best:
                                    best = cur
                            else:
                                cur = 0
                        return best

                    longest_run_frames = _longest_run(mask_list)
                    frame_hop_seconds = hop / float(sr)
                    longest_silence_sec = float(longest_run_frames * frame_hop_seconds)

                    edge_guard_seconds = float(
                        os.getenv("CHATTERBOX_EXPERIMENT_INTERNAL_SILENCE_EDGE_GUARD_SEC", "0.25")
                    )
                    guard_frames = int(edge_guard_seconds / frame_hop_seconds)
                    internal_mask = list(mask_list)
                    if guard_frames > 0 and len(internal_mask) > (2 * guard_frames):
                        for i in range(guard_frames):
                            internal_mask[i] = False
                            internal_mask[-(i + 1)] = False
                    longest_internal_frames = _longest_run(internal_mask)
                    longest_internal_silence_sec = float(longest_internal_frames * frame_hop_seconds)

                    # Extract first internal silence segment for attribution.
                    def _first_run_bounds(seq):
                        in_run = False
                        start = 0
                        for idx, v in enumerate(seq):
                            if v and not in_run:
                                start = idx
                                in_run = True
                            elif not v and in_run:
                                return start, idx
                        if in_run:
                            return start, len(seq)
                        return None

                    first_internal = _first_run_bounds(internal_mask)
                    first_internal_start_sec = -1.0
                    first_internal_end_sec = -1.0
                    approx_char_idx = -1
                    approx_text_window = ""
                    if first_internal:
                        first_internal_start_sec = float(first_internal[0] * frame_hop_seconds)
                        first_internal_end_sec = float(first_internal[1] * frame_hop_seconds)
                        if text_len > 0 and samples > 0:
                            duration_sec = samples / float(sr)
                            if duration_sec > 0:
                                approx_char_idx = int((first_internal_start_sec / duration_sec) * text_len)
                                approx_char_idx = max(0, min(text_len - 1, approx_char_idx))
                                if text_full:
                                    lo = max(0, approx_char_idx - 120)
                                    hi = min(len(text_full), approx_char_idx + 120)
                                    approx_text_window = text_full[lo:hi]
                    else:
                        first_internal_start_sec = -1.0
                        first_internal_end_sec = -1.0
                        approx_char_idx = -1
                        approx_text_window = ""

                silence_peak_threshold = float(os.getenv("CHATTERBOX_EXPERIMENT_SILENCE_PEAK_THRESHOLD", "1e-6"))
                silence_rms_threshold = float(os.getenv("CHATTERBOX_EXPERIMENT_SILENCE_RMS_THRESHOLD", "1e-7"))
                is_silent = (samples == 0) or (peak < silence_peak_threshold and rms < silence_rms_threshold)
                max_internal_silence_sec = float(
                    os.getenv("CHATTERBOX_EXPERIMENT_MAX_INTERNAL_SILENCE_SEC", "1.0")
                )
                has_internal_silence = longest_internal_silence_sec >= max_internal_silence_sec
                fail_on_internal_silence = _env_true(
                    "CHATTERBOX_EXPERIMENT_FAIL_ON_INTERNAL_SILENCE", False
                )

                logger.warning(
                    "🧪 Vocoder diagnostics | chunk_id=%s token_count=%s samples=%s peak=%.3e rms=%.3e silent=%s "
                    "silence_ratio=%.3f longest_silence=%.2fs longest_internal_silence=%.2fs "
                    "thresholds=(peak<%.1e,rms<%.1e,frame_rms<%.1e,internal>=%.2fs)",
                    chunk_id,
                    token_count,
                    samples,
                    peak,
                    rms,
                    is_silent,
                    silence_ratio,
                    longest_silence_sec,
                    longest_internal_silence_sec,
                    silence_peak_threshold,
                    silence_rms_threshold,
                    frame_silence_threshold,
                    max_internal_silence_sec,
                )
                if text_preview:
                    logger.warning("🧪 Chunk text preview | chunk_id=%s text_len=%s preview=%s", chunk_id, text_len, text_preview)
                if first_internal_start_sec >= 0:
                    logger.warning(
                        "🧪 Internal silence attribution | chunk_id=%s first_internal_start=%.2fs first_internal_end=%.2fs approx_char_idx=%s approx_text_window=%s",
                        chunk_id,
                        first_internal_start_sec,
                        first_internal_end_sec,
                        approx_char_idx,
                        approx_text_window,
                    )

                if has_internal_silence:
                    logger.warning(
                        "🧪 Vocoder internal silence detected | longest_internal_silence=%.2fs threshold=%.2fs action=%s",
                        longest_internal_silence_sec,
                        max_internal_silence_sec,
                        "fail" if fail_on_internal_silence else "log_only",
                    )

                if _env_true("CHATTERBOX_EXPERIMENT_ENABLE_SILENCE_GATE", False):
                    if is_silent:
                        raise RuntimeError(
                            f"Vocoder produced silent output (samples={samples}, peak={peak:.3e}, rms={rms:.3e})"
                        )
                    if has_internal_silence and fail_on_internal_silence:
                        raise RuntimeError(
                            "Vocoder produced long internal silence "
                            f"(longest_internal_silence={longest_internal_silence_sec:.2f}s >= {max_internal_silence_sec:.2f}s)"
                        )
            except Exception as diag_e:
                if isinstance(diag_e, RuntimeError):
                    raise
                logger.warning("🧪 Vocoder diagnostics failed: %s", diag_e)

            return result

        _wrapped_s3_inference._minstraly_s3_exp_wrapped = True
        model.s3gen.inference = _wrapped_s3_inference
        logger.warning("🧪 Installed S3/vocoder diagnostics wrapper")
    except Exception as e:
        logger.warning("🧪 Failed to install S3/vocoder diagnostics wrapper: %s", e)

# ---------------------------------------------------------------------------------
# Disk/cache management: centralize caches and provide cleanup + headroom checks
# ---------------------------------------------------------------------------------
def _ensure_cache_env_dirs():
    """Set cache-related environment variables and ensure directories exist."""
    try:
        # Align model caches to persistent /models path for cold-start avoidance
        models_root = Path(os.getenv("MODELS_ROOT", "/models"))
        hf_root = Path(os.getenv("HF_ROOT", str(models_root / "hf")))
        torch_root = Path(os.getenv("TORCH_ROOT", str(models_root / "torch")))

        for p in [models_root, hf_root, hf_root / "hub", torch_root]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Set HuggingFace cache directories to use MODELS_ROOT (should be network volume)
        # This ensures models download to persistent storage, not local disk
        os.environ.setdefault("HF_HOME", str(hf_root))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root))
        os.environ.setdefault("TORCH_HOME", str(torch_root))
        # Optional caches
        os.environ.setdefault("PIP_CACHE_DIR", str(models_root / "pip"))
        os.environ.setdefault("XDG_CACHE_HOME", str(models_root / "xdg"))
        os.environ.setdefault("NLTK_DATA", str(models_root / "nltk"))

        # If a broken/cyclic symlink exists at ~/.cache/huggingface, remove it
        hf_default = Path.home() / ".cache" / "huggingface"
        try:
            if hf_default.is_symlink():
                try:
                    _ = hf_default.resolve()
                except Exception:
                    hf_default.unlink(missing_ok=True)
        except Exception:
            pass

        # Ensure default HF hub dir exists to avoid token write errors
        try:
            (hf_default / "hub").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    except Exception:
        # Non-fatal: proceed without caches
        pass

def _bytes_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _disk_free_bytes(path: str = "/") -> int:
    try:
        total, used, free = shutil.disk_usage(path)
        return int(free)
    except Exception:
        return 0

def _directory_size_bytes(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        total_size = 0
        for root, dirs, files in os.walk(path, topdown=True):
            for f in files:
                try:
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
                except Exception:
                    pass
        return total_size
    except Exception:
        return 0

def log_disk_usage_summary(context: str = "") -> None:
    try:
        logger.info(f"🧭 Disk usage summary {('(' + context + ')') if context else ''}:")
        points = [
            ("/", Path("/")),
            ("/cache", Path("/cache")),
            ("HF_HOME", Path(os.environ.get("HF_HOME", ""))),
            ("HF_HUB_CACHE", Path(os.environ.get("HF_HUB_CACHE", ""))),
            ("TRANSFORMERS_CACHE", Path(os.environ.get("TRANSFORMERS_CACHE", ""))),
            ("~/.cache/huggingface", Path.home() / ".cache" / "huggingface"),
            ("TORCH_HOME", Path(os.environ.get("TORCH_HOME", ""))),
            ("~/.cache/torch", Path.home() / ".cache" / "torch"),
            ("~/.nv/ComputeCache", Path.home() / ".nv" / "ComputeCache"),
            ("/tmp", Path("/tmp")),
            ("/voice_profiles", Path("/voice_profiles")),
            ("/voice_samples", Path("/voice_samples")),
            ("/temp_voice", Path("/temp_voice")),
            ("/tts_generated", Path("/tts_generated")),
            ("/workspace/chatterbox_embed/.git", Path("/workspace/chatterbox_embed/.git")),
        ]
        rows = []
        for label, p in points:
            try:
                size_b = _directory_size_bytes(p)
                rows.append((label, size_b, str(p)))
            except Exception:
                rows.append((label, 0, str(p)))
        rows.sort(key=lambda r: r[1], reverse=True)
        for label, size_b, pstr in rows:
            logger.info(f"  {label:28} { _bytes_human(size_b):>10 }  -> {pstr}")
    except Exception:
        pass

def _safe_remove(path: Path):
    try:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def cleanup_runtime_storage(force: bool = False, *, temp_age_seconds: int = 60 * 30) -> None:
    """
    Prune temporary working dirs and, when necessary, heavy caches.
    - Always clear temp voice/story dirs aggressively (older-than policy).
    - If force=True or disk is low, also prune model/tool caches.
    """
    try:
        # Disabled by default; enable via ENABLE_STORAGE_MAINTENANCE=true
        if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() != "true":
            return
        # Temp/work dirs
        temp_dirs = [
            Path("/temp_voice"),
            Path("/voice_samples"),
            Path("/voice_profiles"),
            Path("/tts_generated"),
        ]
        now = time.time()
        for d in temp_dirs:
            if not d.exists():
                continue
            for entry in d.iterdir():
                try:
                    mtime = entry.stat().st_mtime if entry.exists() else now
                    if force or (now - mtime) > temp_age_seconds:
                        _safe_remove(entry)
                except Exception:
                    pass

        # Decide whether to prune caches
        min_free_gb = float(os.getenv("MIN_FREE_GB", "10"))
        free_bytes = _disk_free_bytes("/")
        low_space = free_bytes < int(min_free_gb * (1024 ** 3))

        if force or low_space:
            # Log what's using space before cleanup
            log_disk_usage_summary("before_cleanup")
            # Known heavy caches (both centralized and default locations)
            cache_candidates = [
                # Keep HF/Torch caches intact for simplicity and speed
                Path.home() / ".nv" / "ComputeCache",
                Path("/tmp"),
            ]
            for c in cache_candidates:
                try:
                    if c and c.exists():
                        for child in c.iterdir():
                            _safe_remove(child)
                except Exception:
                    pass

        # Log free space after cleanup
        free_after = _disk_free_bytes("/")
        logger.info(
            f"🧹 Cleanup done. Free space: { _bytes_human(free_after) }"
        )
        # Log summary again after cleanup
        log_disk_usage_summary("after_cleanup")
    except Exception:
        # Never fail handler due to cleanup
        pass

def ensure_disk_headroom(min_free_gb: float = None) -> None:
    """Ensure minimum free disk space; trigger cleanup when below threshold."""
    try:
        # Disabled by default; enable via ENABLE_STORAGE_MAINTENANCE=true
        if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() != "true":
            return
        if min_free_gb is None:
            min_free_gb = float(os.getenv("MIN_FREE_GB", "2"))
        free_before = _disk_free_bytes("/")
        logger.info(f"💽 Free disk before check: { _bytes_human(free_before) }")
        if free_before < int(min_free_gb * (1024 ** 3)):
            logger.warning(
                f"⚠️ Low disk space detected (<{min_free_gb} GB). Running cleanup..."
            )
            cleanup_runtime_storage(force=True)
    except Exception:
        pass

# Initialize cache env as early as possible
_ensure_cache_env_dirs()

# Early, pre-import disk headroom preflight (runs before any model downloads)
try:
    if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() == "true":
        _pre_free = _disk_free_bytes("/")
        logger.info(f"💽 Free disk early preflight: { _bytes_human(_pre_free) }")
        _min_gb = float(os.getenv("MIN_FREE_GB", "10"))
        if _pre_free < int(_min_gb * (1024 ** 3)):
            logger.warning(f"⚠️ Low disk space detected in preflight (<{_min_gb} GB). Running cleanup...")
            log_disk_usage_summary("preflight_before_cleanup")
            cleanup_runtime_storage(force=True)
            log_disk_usage_summary("preflight_after_cleanup")
except Exception:
    pass

APP_BASE_URL = os.getenv('APP_BASE_URL', '').strip()

def _normalize_callback_url(url: str) -> str:
    """Replace localhost callback host with APP_BASE_URL if provided; then canonicalize."""
    try:
        p = urlparse(url)
        # Replace localhost hosts if an explicit base URL is provided
        if APP_BASE_URL and (p.hostname in ('localhost', '127.0.0.1')):
            base = urlparse(APP_BASE_URL)
            return urlunparse((base.scheme or 'https', base.netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        pass
    return url

def _persist_failed_callback_to_storage(payload: dict, story_id: str) -> Optional[str]:
    """Persist failed callback payload to R2 for later replay under failed_callbacks/YYYY-MM-DD."""
    try:
        key = f"failed_callbacks/{datetime.utcnow():%Y-%m-%d}/{(story_id or 'unknown')}-{int(time.time())}.json"
        payload_json = json.dumps(payload).encode('utf-8')
        result = upload_to_r2(payload_json, key, content_type='application/json')
        return key if result else None
    except Exception:
        return None

def _write_metadata_json(storage_path: str, metadata: dict) -> Optional[str]:
    """Write a sibling .json metadata file next to the audio object in R2."""
    try:
        if not storage_path:
            return None
        json_path = storage_path.rsplit('.', 1)[0] + '.json'
        metadata_json = json.dumps(metadata).encode('utf-8')
        result = upload_to_r2(metadata_json, json_path, content_type='application/json')
        return json_path if result else None
    except Exception:
        return None

def notify_error_callback(error_callback_url: str, story_id: str, error_message: str, **kwargs):
    """
    Send error callback to the main app when TTS generation fails.
    
    :param error_callback_url: URL of the error callback endpoint
    :param story_id: The ID of the story that failed audio generation
    :param error_message: Human-readable error message
    :param kwargs: Additional parameters (user_id, voice_id, error_details, job_id, metadata)
    """
    payload = {
        "story_id": story_id,
        "error": error_message,
        "error_details": kwargs.get("error_details"),
        "user_id": kwargs.get("user_id"),
        "voice_id": kwargs.get("voice_id"),
        "job_id": kwargs.get("job_id"),
        "metadata": kwargs.get("metadata", {})
    }
    
    try:
        logger.info(f"📤 Sending error callback to: {error_callback_url}")
        logger.info(f"📤 Error callback payload: {payload}")
        _post_signed_callback_with_retry(error_callback_url, payload)
        logger.info(f"✅ Error callback sent successfully for story {story_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send error callback: {e}")
        logger.error(f"❌ Error callback exception type: {type(e)}")
        return False

def _post_signed_callback(callback_url: str, payload: dict):
    """POST JSON payload to callback_url with HMAC headers compatible with app callback."""
    logger.info(f"🔍 _post_signed_callback called with URL: {callback_url}")
    logger.info(f"🔍 _post_signed_callback payload keys: {list(payload.keys())}")
    
    # Create a clean version of payload for logging (without raw data)
    clean_payload = {k: v for k, v in payload.items() if k not in ['audio_data']}
    if 'audio_data' in payload:
        clean_payload['audio_data'] = f"[BASE64 DATA] Length: {len(payload['audio_data'])} chars"
    logger.info(f"🔍 _post_signed_callback clean payload: {clean_payload}")
    
    secret = os.getenv('MINSTRALY_API_SHARED_SECRET')
    if not secret:
        logger.error("❌ MINSTRALY_API_SHARED_SECRET not set; cannot sign callback")
        raise RuntimeError('MINSTRALY_API_SHARED_SECRET not set; cannot sign callback')
    
    logger.info(f"🔍 MINSTRALY_API_SHARED_SECRET exists: {bool(secret)}")
    logger.info(f"🔍 MINSTRALY_API_SHARED_SECRET length: {len(secret) if secret else 0}")

    # Canonicalize callback URL to avoid 307 redirects (prefer www.minstraly.com)
    def _canonicalize_callback_url(url: str) -> str:
        try:
            p = urlparse(url)
            scheme = p.scheme or 'https'
            netloc = p.netloc
            if netloc == 'minstraly.com':
                netloc = 'www.minstraly.com'
            if not netloc and p.path:
                return f'https://www.minstraly.com{p.path}'
            return urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
        except Exception:
            return url

    # Normalize against APP_BASE_URL first, then canonicalize
    canonical_url = _canonicalize_callback_url(_normalize_callback_url(callback_url))

    parsed = urlparse(canonical_url)
    path_for_signing = parsed.path or '/api/tts/callback'
    ts = str(int(time.time() * 1000))
    
    logger.info(f"🔍 Parsed URL: {parsed}")
    logger.info(f"🔍 Path for signing: {path_for_signing}")
    logger.info(f"🔍 Timestamp: {ts}")

    body_bytes = json.dumps(payload).encode('utf-8')
    prefix = f"POST\n{path_for_signing}\n{ts}\n".encode('utf-8')
    message = prefix + body_bytes
    sig = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
    
    logger.info(f"🔍 Body size: {len(body_bytes)} bytes")
    logger.info(f"🔍 Message size: {len(message)} bytes")
    logger.info(f"🔍 Signature: {sig[:20]}...")

    headers = {
        'Content-Type': 'application/json',
        'X-Minstraly-Timestamp': ts,
        'X-Minstraly-Signature': sig,
    }
    
    logger.info(f"🔍 Headers: {headers}")
    logger.info(f"🔍 Making POST request to: {canonical_url}")
    
    resp = requests.post(canonical_url, data=body_bytes, headers=headers, timeout=15)
    
    logger.info(f"🔍 Response status: {resp.status_code}")
    logger.info(f"🔍 Response headers: {dict(resp.headers)}")
    logger.info(f"🔍 Response text: {resp.text[:200]}...")
    
    resp.raise_for_status()
    logger.info(f"✅ Callback POST successful: {resp.status_code}")

def _post_signed_callback_with_retry(callback_url: str, payload: dict, *, retries: int = 4, base_delay: float = 5.0):
    """Retry wrapper around _post_signed_callback with exponential backoff and durable persistence."""
    last_exc: Optional[Exception] = None
    url = _normalize_callback_url(callback_url)
    for attempt in range(1, retries + 1):
        try:
            _post_signed_callback(url, payload)
            return
        except Exception as e:
            last_exc = e
            logger.warning(f"Callback attempt {attempt}/{retries} failed: {e}")
            time.sleep(base_delay * (2 ** (attempt - 1)))
    try:
        _persist_failed_callback_to_storage(payload, payload.get('story_id') or 'unknown')
    except Exception:
        pass
    if last_exc:
        raise last_exc

def clear_python_cache():
    """Clear Python caches and loaded chatterbox modules to ensure fresh load."""
    try:
        for pyc_file in glob.glob("/workspace/**/*.pyc", recursive=True):
            try:
                os.remove(pyc_file)
            except Exception:
                pass
        for pycache_dir in pathlib.Path("/workspace").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
            except Exception:
                pass
        to_clear = [name for name in list(sys.modules.keys()) if 'chatterbox' in name]
        for name in to_clear:
            del sys.modules[name]
    except Exception:
        pass

# Clear cache BEFORE importing any chatterbox modules
clear_python_cache()

# Import storage utilities (always available)
try:
    from chatterbox.storage import resolve_bucket_name, is_r2_bucket, upload_to_r2, download_from_r2, _encode_metadata_value
    logger.info("✅ Successfully imported storage utilities from chatterbox")
except ImportError as e:
    logger.warning(f"⚠️ Could not import storage utilities: {e}")
    # Fallback: define functions locally if import fails
    def resolve_bucket_name(bucket_name: Optional[str] = None, country_code: Optional[str] = None) -> str:
        return os.getenv('R2_BUCKET_NAME', 'minstraly-storage')
    def is_r2_bucket(bucket_name: str) -> bool:
        return bucket_name == 'minstraly-storage' or bucket_name.startswith('r2://')
    def upload_to_r2(data: bytes, destination_key: str, content_type: str = "application/octet-stream", metadata: dict = None, bucket_name: Optional[str] = None) -> Optional[str]:
        logger.error("Storage utilities not available - upload_to_r2 not implemented")
        return None
    def download_from_r2(source_key: str) -> Optional[bytes]:
        logger.error("Storage utilities not available - download_from_r2 not implemented")
        return None

# Import text sanitizer for early validation (lightweight, no model dependencies)
try:
    from chatterbox.chunking import AdvancedTextSanitizer
    TEXT_SANITIZER_AVAILABLE = True
    text_sanitizer = AdvancedTextSanitizer()
    logger.info("✅ Successfully imported AdvancedTextSanitizer for text validation")
except ImportError as e:
    TEXT_SANITIZER_AVAILABLE = False
    text_sanitizer = None
    logger.warning(f"⚠️ Could not import AdvancedTextSanitizer: {e}")

# Import the models from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    from chatterbox.tts import ChatterboxTTS
    FORKED_HANDLER_AVAILABLE = True
    logger.info("✅ Successfully imported ChatterboxVC and ChatterboxTTS from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Could not import models from forked repository: {e}")

# Initialize models once at startup
vc_model = None
tts_model = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")  # For voice clone samples
TTS_GENERATED_DIR = Path("/tts_generated")  # For TTS story generation
TEMP_VOICE_DIR = Path("/temp_voice")

# Create directories if they don't exist
for _d in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TTS_GENERATED_DIR, TEMP_VOICE_DIR]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

os.environ.setdefault("TMPDIR", str(TEMP_VOICE_DIR))

logger.info(f"Using directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# R2 storage is used directly via boto3 - no client initialization needed

# Initialize models
ensure_disk_headroom()

# Check disk space before model initialization
models_root = Path(os.getenv("MODELS_ROOT", "/models"))
free_space_gb = _disk_free_bytes(str(models_root)) / (1024 ** 3)
logger.info(f"💾 Disk space at {models_root}: {free_space_gb:.2f} GB free")

if free_space_gb < 5:
    logger.warning(f"⚠️ Low disk space ({free_space_gb:.2f} GB). Models require ~3-4 GB.")
    logger.warning("💡 Tip: Mount a network volume and set MODELS_ROOT to the volume path (e.g., /runpod-volume/models)")
    if models_root == Path("/models"):
        logger.warning("💡 Current MODELS_ROOT is /models (local disk). Consider using a network volume.")

logger.info("🔧 Initializing models...")
tts_model = None
vc_model = None

if FORKED_HANDLER_AVAILABLE:
    # Use from_pretrained() which will use pre-downloaded models from HuggingFace cache
    # Models are pre-downloaded during Docker build to /models/hf
    # Initialize TTS model first (needed for s3gen)
    try:
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS ready")
        try:
            import chatterbox.tts as _tts_mod
            mod_file = getattr(_tts_mod, "__file__", "<unknown>")
            logger.warning("🧪 Loaded chatterbox.tts from: %s", mod_file)
            runtime_marker = (
                getattr(_tts_mod, "CHATTERBOX_RUNTIME_VERSION", None)
                or getattr(_tts_mod, "__version__", None)
                or getattr(getattr(_tts_mod, "ChatterboxTTS", object), "RUNTIME_VERSION", None)
            )
            if runtime_marker:
                logger.warning("🧪 chatterbox.tts runtime marker: %s", runtime_marker)
            else:
                logger.warning("🧪 chatterbox.tts runtime marker: <missing>")
        except Exception:
            pass
        _patch_chunk_context_tracking(tts_model)
        _patch_t3_inference_diagnostics(tts_model)
        _patch_s3_inference_diagnostics(tts_model)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to initialize TTS model: {error_msg}")
        tts_model = None
    
    # Initialize VC model separately (allow TTS to work even if VC fails)
    try:
        vc_model = ChatterboxVC.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxVC ready")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to initialize VC model: {error_msg}")
        logger.warning("⚠️ VC model unavailable, but TTS will continue to work")
        vc_model = None
else:
    logger.error("❌ Forked repository models not available")
    tts_model = None
    vc_model = None

# -------------------------------------------------------------------
# 🐞  R2 credential debug helper (Firebase/HF are legacy)
# -------------------------------------------------------------------
def _debug_r2_creds():
    """Log R2 credential presence (never raises)."""
    logger.info("🔍 ===== TTS R2 CREDENTIAL VALIDATION =====")
    try:
        r2_account_id = os.getenv("R2_ACCOUNT_ID")
        r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_endpoint = os.getenv("R2_ENDPOINT")
        r2_bucket_name = os.getenv("R2_BUCKET_NAME") or "minstraly-storage"
        r2_public_url = os.getenv("NEXT_PUBLIC_R2_PUBLIC_URL") or os.getenv("R2_PUBLIC_URL")

        logger.info(f"🪣 R2 bucket: {r2_bucket_name}")
        logger.info(f"🔑 R2_ACCOUNT_ID present: {bool(r2_account_id)}")
        logger.info(f"🔑 R2_ACCESS_KEY_ID present: {bool(r2_access_key_id)}")
        logger.info(f"🔑 R2_SECRET_ACCESS_KEY present: {bool(r2_secret_access_key)}")
        logger.info(f"🌐 R2_ENDPOINT present: {bool(r2_endpoint)}")
        logger.info(f"🌍 R2 public URL present: {bool(r2_public_url)}")
    except Exception as e:
        logger.error(f"❌ R2 credential validation failed: {e}")
    logger.info("🔍 ===== END TTS R2 CREDENTIAL VALIDATION =====")

def validate_storage_path(path: str) -> str:
    """
    Validate and sanitize a storage path to prevent directory traversal attacks.
    
    :param path: The storage path to validate
    :return: The sanitized path
    :raises ValueError: If the path contains invalid sequences (.. or leading /)
    """
    if not path:
        raise ValueError("Storage path cannot be empty")
    
    # Check for directory traversal sequences
    if '..' in path:
        raise ValueError(f"Storage path contains invalid '..' sequence: {path}")
    
    # Check for absolute paths (leading slash)
    if path.startswith('/'):
        raise ValueError(f"Storage path cannot start with '/': {path}")
    
    # Normalize the path by removing any trailing slashes (already handled by rstrip in usage)
    return path.strip()

# Removed: initialize_firebase() - R2 is no longer used, only R2

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """
    Upload data directly to R2 only (R2 removed).
    
    :param data: Binary data to upload
    :param destination_blob_name: Destination path in R2
    :param content_type: MIME type of the file
    :param metadata: Optional metadata to store with the file
    :return: Public URL or None if failed
    """
    try:
        # Resolve bucket from metadata hints (non-R2 bucket names are ignored)
        bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
        country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
        resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)
        
        logger.info(f"🔍 Resolved bucket: {resolved_bucket} (from bucket_hint={bucket_hint}, country_hint={country_hint})")
        
        # resolve_bucket_name() always returns an R2 bucket (ignores non-R2 bucket names)
        logger.info(f"✅ Using R2 upload for bucket: {resolved_bucket}")
        return upload_to_r2(data, destination_blob_name, content_type, metadata, bucket_name=resolved_bucket)
        
    except Exception as e:
        logger.error(f"❌ R2 upload failed: {e}")
        import traceback
        logger.error(f"❌ R2 upload traceback: {traceback.format_exc()}")
        return None

def rename_in_firebase(src_path: str, dest_path: str, *, metadata: Optional[dict] = None, content_type: Optional[str] = None) -> Optional[str]:
    """
    Rename (copy + delete) an object in R2 only (R2 removed).
    Returns new public URL or None.
    """
    try:
        # Resolve bucket from metadata hints (non-R2 bucket names are ignored)
        bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
        country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
        
        # resolve_bucket_name() always returns an R2 bucket (ignores non-R2 bucket names)
        resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)
        
        logger.info(f"✅ Rename: Using R2 copy/delete for bucket: {resolved_bucket}")
        
        import boto3
        
        # Get R2 credentials
        r2_account_id = os.getenv('R2_ACCOUNT_ID')
        r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.getenv('R2_ENDPOINT')
        r2_bucket_name = resolved_bucket
        r2_public_url = os.getenv('NEXT_PUBLIC_R2_PUBLIC_URL') or os.getenv('R2_PUBLIC_URL')
        
        if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("❌ R2 credentials not configured")
            return None
        
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name='auto'
        )
        
        # Check if source exists
        try:
            s3_client.head_object(Bucket=r2_bucket_name, Key=src_path)
        except Exception:
            logger.warning(f"⚠️ Source object does not exist in R2: {src_path}")
            return None
        
        # Copy object
        copy_source = {'Bucket': r2_bucket_name, 'Key': src_path}
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        if metadata:
            # R2 metadata must be ASCII-compatible - encode non-ASCII values
            try:
                from chatterbox.storage import _encode_metadata_value
            except ImportError:
                # Fallback: define encoding function locally if import fails
                import base64
                def _encode_metadata_value(value: str) -> str:
                    try:
                        value.encode('ascii')
                        return value
                    except UnicodeEncodeError:
                        encoded = base64.b64encode(value.encode('utf-8')).decode('ascii')
                        return f"base64:{encoded}"
            
            encoded_metadata = {}
            for k, v in metadata.items():
                key_str = str(k)
                value_str = str(v)
                encoded_metadata[key_str] = _encode_metadata_value(value_str)
            extra_args['Metadata'] = encoded_metadata
        
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=r2_bucket_name,
            Key=dest_path,
            **extra_args
        )
        
        # Delete original
        try:
            s3_client.delete_object(Bucket=r2_bucket_name, Key=src_path)
        except Exception as del_e:
            logger.warning(f"⚠️ Could not delete original R2 object {src_path}: {del_e}")
        
        logger.info(f"✅ Renamed in R2: {src_path} → {dest_path}")
        
        # Return public URL if available
        if r2_public_url:
            return f"{r2_public_url.rstrip('/')}/{dest_path}"
        return dest_path
        
    except Exception as e:
        logger.error(f"❌ R2 rename failed {src_path} → {dest_path}: {e}")
        import traceback
        logger.error(f"❌ R2 rename traceback: {traceback.format_exc()}")
        return None

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TTS_GENERATED_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def call_tts_model_generate_tts_story(text, voice_id, profile_base64, language, story_type, is_kids_voice, api_metadata, voice_name=None, profile_path=None, user_id=None, story_id=None):
    """
    Implement TTS generation using available model methods.
    
    Uses the TTS model's generate method for text-to-speech generation.
    """
    global tts_model
    _patch_chunk_context_tracking(tts_model)
    _patch_t3_inference_diagnostics(tts_model)
    _patch_s3_inference_diagnostics(tts_model)
    
    # Extract user_id and story_id from metadata if not provided explicitly
    if not user_id:
        user_id = api_metadata.get("user_id") if isinstance(api_metadata, dict) else ""
    if not story_id:
        story_id = api_metadata.get("story_id") if isinstance(api_metadata, dict) else ""

    _log_worker_experiment_context(
        "call_tts_model_generate_tts_story",
        api_metadata=api_metadata if isinstance(api_metadata, dict) else None,
    )
    if hasattr(tts_model, "experiment_config"):
        try:
            logger.warning("🧪 Runtime tts_model.experiment_config (pre-call): %s", getattr(tts_model, "experiment_config", {}))
        except Exception:
            pass
    
    # Extract genre from metadata to determine TTS parameters
    # Check multiple possible keys and locations
    genre = None
    if isinstance(api_metadata, dict):
        genre = (api_metadata.get("genre") or 
                 api_metadata.get("story_genre") or
                 api_metadata.get("storyGenre"))
    
    # Log genre extraction for debugging
    logger.info(f"🔍 Genre extraction in call_tts_model_generate_tts_story:")
    logger.info(f"   api_metadata type: {type(api_metadata)}")
    if isinstance(api_metadata, dict):
        logger.info(f"   api_metadata keys: {list(api_metadata.keys())}")
        logger.info(f"   genre from api_metadata: {api_metadata.get('genre')}")
        logger.info(f"   story_genre from api_metadata: {api_metadata.get('story_genre')}")
    logger.info(f"   Final extracted genre: {genre}")
    
    # Normalize genre for comparison
    genre_normalized = None
    if genre:
        genre_normalized = str(genre).lower().strip()
        logger.info(f"   Normalized genre: {genre_normalized}")
    
    # Determine if this is an erotic story
    erotic_genres = ['erotic', 'advanced-erotic', 'hardcore erotic', 'hardcore-erotic', 'advanced erotic']
    is_erotic = genre_normalized in erotic_genres if genre_normalized else False
    logger.info(f"   Is erotic story: {is_erotic}")
    if is_erotic:
        logger.info(f"   Matched erotic genre: {genre_normalized}")
    
    # Set TTS parameters based on story type
    if is_erotic:
        # Erotic stories: slower narration, more deliberate pacing
        temperature = 0.65  # Lower temperature for more consistent, less varied delivery
        exaggeration = None  # Use default (0.5) for all genres
        cfg_weight = 0.4  # Lower CFG for slower, more deliberate pacing (default is 0.5)
        pause_scale = 1.4  # Slower narration (default is 1.15)
        # IMPORTANT: Chatterbox has adaptive per-chunk params that can override base temp/cfg/exag.
        # For erotic stories we want the passed params to *actually* take effect, so disable adaptive voice-param overrides.
        adaptive_voice_param_blend = 0.0
        logger.info(f"🎭 Erotic story detected - applying specialized TTS parameters")
        logger.info(
            f"   temperature={temperature}, exaggeration=0.5 (default), cfg_weight={cfg_weight}, "
            f"pause_scale={pause_scale}, adaptive_voice_param_blend={adaptive_voice_param_blend}"
        )
    else:
        # Default parameters for other story types
        temperature = None  # Use model default (0.8)
        exaggeration = None  # Use model default (0.5)
        cfg_weight = 0.5  # Explicit default CFG weight for non-erotic stories
        pause_scale = 1.15  # Default pause scale
        adaptive_voice_param_blend = 1.0

    logger.warning(
        "🧪 Effective TTS params before model call | voice_id=%s story_id=%s genre=%s erotic=%s temp=%s exag=%s cfg=%s pause_scale=%s adaptive_blend=%s",
        voice_id,
        story_id,
        genre,
        is_erotic,
        temperature if temperature is not None else "default",
        exaggeration if exaggeration is not None else "default",
        cfg_weight,
        pause_scale,
        adaptive_voice_param_blend,
    )
    
    logger.info(f"🎯 ===== CALLING TTS GENERATION =====")
    logger.info(f"🔍 Parameters:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  voice_name: {voice_name}")
    logger.info(f"  language: {language}")
    logger.info(f"  story_type: {story_type}")
    logger.info(f"  genre: {genre}")
    logger.info(f"  is_erotic: {is_erotic}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  text_length: {len(text)} characters")
    logger.info(f"  profile_path: {profile_path}")
    logger.info(f"  user_id: {user_id}")
    logger.info(f"  story_id: {story_id}")
    logger.info(f"🔍 TTS Parameters to be used:")
    logger.info(f"  temperature: {temperature if temperature is not None else 'default (0.8)'}")
    logger.info(f"  exaggeration: {exaggeration if exaggeration is not None else 'default (0.5)'}")
    logger.info(f"  cfg_weight: {cfg_weight if cfg_weight is not None else 'default (0.5)'}")
    logger.info(f"  pause_scale: {pause_scale}")
    logger.info(f"  adaptive_voice_param_blend: {adaptive_voice_param_blend}")
    
    start_time = time.time()
    
    try:
        # Check if TTS model is available
        if tts_model is None:
            error_msg = "TTS model not available"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "generation_time": time.time() - start_time
            }
        
        # Try to use the TTS model's generate_tts_story method
        if hasattr(tts_model, 'generate_tts_story'):
            logger.info("🔄 Using TTS model's generate_tts_story method...")
            try:
                result = tts_model.generate_tts_story(
                    text=text,
                    voice_id=voice_id,
                    profile_base64=profile_base64,
                    language=language,
                    story_type=story_type,
                    is_kids_voice=is_kids_voice,
                    metadata=api_metadata,
                    voice_name=voice_name,
                    profile_path=profile_path,
                    user_id=user_id,
                    story_id=story_id,
                    pause_scale=pause_scale,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    adaptive_voice_param_blend=adaptive_voice_param_blend,
                )
                generation_time = time.time() - start_time
                if isinstance(result, dict):
                    logger.warning(
                        "🧪 TTS model result summary | status=%s generation_time=%.2fs has_metadata=%s output_path=%s",
                        result.get("status"),
                        generation_time,
                        "metadata" in result,
                        (result.get("output_path") or result.get("storage_path") or result.get("r2_path") or ""),
                    )
                logger.info(f"✅ TTS generation completed in {generation_time:.2f}s")
                return result
            except Exception as method_error:
                error_msg = f"generate_tts_story method failed: {method_error}"
                logger.error(f"❌ {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "generation_time": time.time() - start_time
                }
        else:
            error_msg = "TTS model doesn't have generate_tts_story method. Please update the RunPod deployment with the latest forked repository version."
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "generation_time": time.time() - start_time
            }
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"❌ TTS generation failed after {generation_time:.2f}s: {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "generation_time": generation_time
        }

def handler(event, responseFormat="base64"):
    """Pure API orchestration: Handle TTS generation requests"""
    global tts_model
    
    ensure_disk_headroom()

    def _return_with_cleanup(obj):
        try:
            cleanup_runtime_storage(force=False)
        except Exception:
            pass
        return obj

    # ===== COMPREHENSIVE INPUT PARAMETER LOGGING =====
    logger.info("🔍 ===== TTS HANDLER INPUT PARAMETERS =====")
    logger.info(f"📥 Raw event keys: {list(event.keys())}")
    logger.info(f"📥 Event type: {type(event)}")
    
    # Log event structure
    if "input" in event:
        logger.info(f"📥 Input keys: {list(event['input'].keys())}")
        for key, value in event["input"].items():
            if key == 'profile_base64' and value:
                logger.info(f"📥 {key}: [BASE64 DATA] Length: {len(value)} chars")
            elif key == 'audio_data' and value:
                logger.info(f"📥 {key}: [BASE64 DATA] Length: {len(value)} chars")
            elif isinstance(value, dict):
                logger.info(f"📥 {key}: {type(value)} with keys: {list(value.keys())}")
            else:
                logger.info(f"📥 {key}: {value}")
    
    if "metadata" in event:
        logger.info(f"📥 Top-level metadata keys: {list(event['metadata'].keys()) if isinstance(event['metadata'], dict) else 'Not a dict'}")
        logger.info(f"📥 Top-level metadata: {event['metadata']}")
    
    # R2 storage is used directly - no initialization needed
    
    # Check if TTS model is available
    if tts_model is None:
        logger.error("❌ TTS model not available")
        return _return_with_cleanup({"status": "error", "error": "TTS model not available"})
    
    logger.info("✅ Using pre-initialized TTS model")
    
    # Handle TTS generation request according to API contract
    text = event["input"].get("text")
    profile_base64 = event["input"].get("profile_base64")
    language = event["input"].get("language", "en")
    story_type = event["input"].get("story_type", "user")
    is_kids_voice = event["input"].get("is_kids_voice", False)
    api_metadata = event["input"].get("metadata", {})
    _log_worker_experiment_context(
        "handler",
        api_metadata=api_metadata if isinstance(api_metadata, dict) else None,
        input_payload=event["input"] if isinstance(event.get("input"), dict) else None,
    )
    profile_path = event["input"].get("profile_path") or (api_metadata.get("profile_path") if isinstance(api_metadata, dict) else None)
    callback_url = api_metadata.get("callback_url") or (event["metadata"].get("callback_url") if isinstance(event.get("metadata"), dict) else None)
    
    # Early validation: Check for disallowed characters BEFORE model initialization
    # This prevents wasting resources on invalid text
    if TEXT_SANITIZER_AVAILABLE and text_sanitizer and text:
        is_valid, error_message, disallowed_chars = text_sanitizer.validate_text_for_language(text, language)
        if not is_valid:
            logger.warning(f"❌ Text validation failed for language '{language}': {error_message}")
            logger.warning(f"❌ Disallowed characters: {disallowed_chars}")
            
            # Send error callback if callback_url is available
            try:
                if callback_url:
                    # Extract story_id and user_id for error callback
                    story_id = api_metadata.get("story_id") or event["input"].get("story_id")
                    user_id = api_metadata.get("user_id") or event["input"].get("user_id")
                    voice_id = event["input"].get("voice_id") or api_metadata.get("voice_id")
                    
                    # Construct error callback URL
                    if "/api/tts/callback" in callback_url:
                        error_callback_url = callback_url.replace("/api/tts/callback", "/api/tts/error-callback")
                    elif "/api/tts/" in callback_url:
                        error_callback_url = callback_url.rsplit("/", 1)[0] + "/error-callback"
                    else:
                        base_url = callback_url.rstrip("/")
                        error_callback_url = f"{base_url}/error-callback"
                    
                    # Send error callback
                    notify_error_callback(
                        error_callback_url=error_callback_url,
                        story_id=story_id or "unknown",
                        error_message=error_message,
                        error_details=f"Text contains characters not supported for language '{language}'. Disallowed characters: {', '.join(repr(c) for c in disallowed_chars[:10])}",
                        user_id=user_id,
                        voice_id=voice_id,
                        job_id=event.get("id"),
                        metadata={
                            "language": language,
                            "story_type": story_type,
                            "text_length": len(text) if text else 0,
                            "disallowed_chars": disallowed_chars[:10],  # Limit to first 10
                            "error_type": "text_validation_error"
                        }
                    )
            except Exception as callback_error:
                logger.error(f"❌ Failed to send error callback: {callback_error}")
            
            return _return_with_cleanup({
                "status": "error",
                "error": error_message,
                "error_type": "text_validation_error",
                "disallowed_characters": disallowed_chars[:20]  # Return first 20 for debugging
            })
        else:
            logger.info(f"✅ Text validation passed for language '{language}'")
    
    # Debug: Log callback_url immediately after extraction
    logger.info(f"🔍 EXTRACTED callback_url: {callback_url}")
    logger.info(f"🔍 EXTRACTED callback_url type: {type(callback_url)}")
    logger.info(f"🔍 EXTRACTED callback_url from api_metadata: {api_metadata.get('callback_url')}")
    logger.info(f"🔍 EXTRACTED callback_url from event metadata: {event.get('metadata', {}).get('callback_url') if isinstance(event.get('metadata'), dict) else 'N/A'}")
    
    # Extract voice_id with fallback to metadata (like VC handler)
    voice_id = event["input"].get("voice_id") or api_metadata.get("voice_id")
    
    # Extract additional metadata variables needed throughout the function
    user_id = api_metadata.get("user_id") or event["input"].get("user_id")
    story_id = api_metadata.get("story_id") or event["input"].get("story_id")
    story_name = api_metadata.get("story_name") or event["input"].get("story_name")
    voice_name = api_metadata.get("voice_name") or event["input"].get("voice_name")
    
    # Extract genre from multiple locations (similar to callback_url extraction)
    genre = (
        event["input"].get("genre")
        or event["input"].get("story_genre")
        or event["input"].get("storyGenre")
        or (api_metadata.get("genre") if isinstance(api_metadata, dict) else None)
        or (api_metadata.get("story_genre") if isinstance(api_metadata, dict) else None)
        or (api_metadata.get("storyGenre") if isinstance(api_metadata, dict) else None)
        or (event.get("metadata", {}).get("genre") if isinstance(event.get("metadata"), dict) else None)
        or (event.get("metadata", {}).get("story_genre") if isinstance(event.get("metadata"), dict) else None)
        or (event.get("metadata", {}).get("storyGenre") if isinstance(event.get("metadata"), dict) else None)
    )
    
    # Ensure genre is in api_metadata for call_tts_model_generate_tts_story to use
    if genre and isinstance(api_metadata, dict):
        api_metadata["genre"] = genre
        api_metadata.setdefault("story_genre", genre)
    
    # Log genre extraction for debugging
    logger.info(f"🔍 Genre extraction:")
    logger.info(f"   From event['input']: {event['input'].get('genre')}")
    logger.info(f"   From api_metadata: {api_metadata.get('genre') if isinstance(api_metadata, dict) else None}")
    logger.info(f"   From event['metadata']: {event.get('metadata', {}).get('genre') if isinstance(event.get('metadata'), dict) else None}")
    logger.info(f"   Final genre: {genre}")
    
    # ===== METADATA BREAKDOWN LOGGING =====
    logger.info("🔍 ===== TTS METADATA BREAKDOWN =====")
    logger.info(f"📋 API metadata: {api_metadata}")
    logger.info(f"📋 API metadata type: {type(api_metadata)}")
    logger.info(f"📋 API metadata keys: {list(api_metadata.keys()) if isinstance(api_metadata, dict) else 'Not a dict'}")
    
    # Log each metadata field with type information
    if isinstance(api_metadata, dict):
        for key, value in api_metadata.items():
            logger.info(f"📋   API metadata {key}: {value} (type: {type(value)})")
    
    # Log top-level metadata
    top_metadata = event.get("metadata", {})
    logger.info(f"📋 Top-level metadata: {top_metadata}")
    logger.info(f"📋 Top-level metadata type: {type(top_metadata)}")
    if isinstance(top_metadata, dict):
        logger.info(f"📋 Top-level metadata keys: {list(top_metadata.keys())}")
        for key, value in top_metadata.items():
            logger.info(f"📋   Top-level metadata {key}: {value} (type: {type(value)})")
    
    logger.info("🔍 ===== END TTS METADATA BREAKDOWN =====")
    
    # ===== R2 CREDENTIAL VALIDATION =====
    _debug_r2_creds()
    
    if not text or not voice_id:
        return _return_with_cleanup({"status": "error", "error": "Both text and voice_id are required"})

    logger.info(f"🎵 TTS request. Voice ID: {voice_id}")
    logger.info(f"🎵 Voice Name: {voice_name}")
    logger.info(f"📝 Text length: {len(text)} characters")
    logger.info(f"🌍 Language: {language}, Story type: {story_type}")
    logger.info(f"👶 Kids voice: {is_kids_voice}")
    
    try:
        # Call the TTS model's generate_tts_story method - it handles everything!
        logger.info("🔄 Calling TTS model's generate_tts_story method...")
        
        result = call_tts_model_generate_tts_story(
            text=text,
            voice_id=voice_id,
            profile_base64=profile_base64,
            language=language,
            story_type=story_type,
            is_kids_voice=is_kids_voice,
            api_metadata=api_metadata,
            voice_name=voice_name,
            profile_path=profile_path,
            user_id=user_id,
            story_id=story_id
        )
        
        # Store callback_url in result for reliable access
        if isinstance(result, dict):
            result["callback_url"] = callback_url
            logger.info(f"🔍 STORED callback_url in result: {callback_url}")
            logger.info(f"🔍 STORED callback_url type: {type(callback_url)}")
            logger.info(f"🔍 STORED callback_url in result keys: {list(result.keys())}")
        else:
            logger.warning(f"⚠️ Result is not a dict, cannot store callback_url. Result type: {type(result)}")
        
        # Debug: Log the result structure for successful TTS generation (without raw data)
        logger.info(f"🔍 TTS generation result type: {type(result)}")
        if isinstance(result, dict):
            # Create a clean version of result without raw data
            clean_result = {k: v for k, v in result.items() if k not in ['audio_data']}
            if 'audio_data' in result:
                clean_result['audio_data'] = f"[BASE64 DATA] Length: {len(result['audio_data'])} chars"
            
            logger.info(f"🔍 TTS generation result keys: {list(result.keys())}")
            logger.info(f"🔍 TTS generation result status: {result.get('status')}")
            logger.info(f"🔍 TTS generation result has firebase_url: {'firebase_url' in result}")
            logger.info(f"🔍 TTS generation result has audio_url: {'audio_url' in result}")
            logger.info(f"🔍 TTS generation result has audio_path: {'audio_path' in result}")
            logger.info(f"🔍 TTS generation result has firebase_path: {'firebase_path' in result}")
            logger.info(f"🔍 TTS generation clean result: {clean_result}")
        else:
            logger.info(f"🔍 TTS generation result: {result}")
        
        # Check if TTS generation failed
        if isinstance(result, dict) and result.get("status") == "error":
            logger.error(f"❌ TTS generation failed: {result.get('message', 'Unknown error')}")
            
            # Send error callback if callback_url is available
            try:
                if callback_url:
                    # Construct error callback URL from success callback URL
                    # Handle different possible callback URL formats
                    if "/api/tts/callback" in callback_url:
                        error_callback_url = callback_url.replace("/api/tts/callback", "/api/tts/error-callback")
                    elif "/api/tts/" in callback_url:
                        # If it's a different TTS endpoint, replace the last part
                        error_callback_url = callback_url.rsplit("/", 1)[0] + "/error-callback"
                    else:
                        # Fallback: append error-callback to the base URL
                        base_url = callback_url.rstrip("/")
                        error_callback_url = f"{base_url}/error-callback"
                    
                    # Use metadata variables already extracted at the top of the function
                    
                    # Send error callback
                    notify_error_callback(
                        error_callback_url=error_callback_url,
                        story_id=story_id or "unknown",
                        error_message=result.get("message", "TTS generation failed"),
                        error_details=f"TTS model returned error status: {result.get('message', 'Unknown error')}",
                        user_id=user_id,
                        voice_id=voice_id,
                        job_id=event.get("id"),  # RunPod job ID
                        metadata={
                            "language": language,
                            "story_type": story_type,
                            "story_name": story_name,
                            "voice_name": voice_name,
                            "text_length": len(text) if text else 0,
                            "generation_time": result.get("generation_time"),
                            "error_type": "tts_model_error"
                        }
                    )
            except Exception as callback_error:
                logger.error(f"❌ Failed to send error callback: {callback_error}")
            
            return _return_with_cleanup(result)
        
        # Return the result from the TTS model
        logger.info(f"📤 TTS generation completed successfully")
        
        # Post-process: rename output file to requested naming if model saved with default name.
        try:
            if isinstance(result, dict) and result.get("status") == "success":
                # Extract additional hints for post-processing
                import re
                output_basename = api_metadata.get("output_basename") or event["input"].get("output_basename")
                output_filename = api_metadata.get("output_filename") or event["input"].get("output_filename")
                if not story_name and output_basename:
                    story_name = output_basename.split("_")[0]
                safe_story = re.sub(r'[^a-z0-9]+', '_', (story_name or 'story').lower()).strip('_')
                base = output_basename or f"{safe_story}_{voice_id}_{story_type}"
                # Determine existing firebase path
                firebase_path = result.get("firebase_path")
                audio_url = result.get("firebase_url") or result.get("audio_url")
                # If we only have URL, derive path from it
                if not firebase_path and audio_url:
                    from urllib.parse import urlparse
                    p = urlparse(audio_url).path
                    firebase_path = p[1:] if p.startswith('/') else p
                # Build target path
                if firebase_path:
                    ext = firebase_path.split('.')[-1].lower() if '.' in firebase_path else 'mp3'
                    # Prefer explicit output_filename (with optional extension), else fall back to output_basename
                    final_filename = None
                    try:
                        if output_filename:
                            # Use only the basename if a path was provided
                            provided_name = os.path.basename(output_filename)
                            if '.' in provided_name:
                                final_filename = provided_name
                                ext = provided_name.rsplit('.', 1)[-1].lower()
                            else:
                                final_filename = f"{provided_name}.{ext}"
                        else:
                            final_filename = f"{base}.{ext}"
                    except Exception:
                        final_filename = f"{base}.{ext}"
                    # Check if storage_path is provided in metadata (for admin generations)
                    storage_path_hint = api_metadata.get("storage_path") if isinstance(api_metadata, dict) else None
                    if storage_path_hint:
                        # Validate storage_path to prevent directory traversal attacks
                        try:
                            validated_path = validate_storage_path(storage_path_hint)
                            # Use provided storage_path and append filename
                            target_path = f"{validated_path.rstrip('/')}/{final_filename}"
                            logger.info(f"🔍 Using storage_path from metadata: {target_path}")
                        except ValueError as e:
                            logger.error(f"❌ Invalid storage_path provided: {e}")
                            # Fall back to default path structure
                            target_path = f"audio/stories/{language}/user/{(user_id or 'user')}/{final_filename}"
                            logger.warning(f"⚠️ Falling back to default path: {target_path}")
                    else:
                        # Store under audio/stories/{language}/user/{user_id}/{file}
                        target_path = f"audio/stories/{language}/user/{(user_id or 'user')}/{final_filename}"
                    # Removed: R2 rename logic - files are uploaded directly to correct R2 path
                    # TTS model uploads directly to R2 with correct path, no rename needed
                    if target_path != firebase_path:
                        logger.info(f"📝 Path mismatch detected but rename skipped (R2-only): {firebase_path} -> {target_path}")
                        logger.info(f"   Using original R2 path from model: {firebase_path}")
                        # Keep the original path from the model (already in R2)
                        result['storage_path'] = firebase_path
                        result['r2_path'] = firebase_path
                        if result.get('firebase_url') or result.get('audio_url'):
                            result['storage_url'] = result.get('firebase_url') or result.get('audio_url')
                            result['r2_url'] = result.get('firebase_url') or result.get('audio_url')
                    else:
                        # Paths match - update result with R2 naming
                        result['storage_path'] = target_path
                        result['r2_path'] = target_path
                        if result.get('firebase_url') or result.get('audio_url'):
                            result['storage_url'] = result.get('firebase_url') or result.get('audio_url')
                            result['r2_url'] = result.get('firebase_url') or result.get('audio_url')
        except Exception as post_e:
            logger.warning(f"⚠️ TTS post-process rename failed: {post_e}")
        # ===== TTS SUCCESS CALLBACK LOGGING =====
        logger.info("🔍 ===== TTS SUCCESS CALLBACK PAYLOAD =====")
        
        # Get callback_url from result (more reliable than variable scope)
        result_callback_url = result.get("callback_url") if isinstance(result, dict) else None
        logger.info(f"🔍 EXTRACTED callback_url from result: {result_callback_url}")
        logger.info(f"🔍 EXTRACTED callback_url from result type: {type(result_callback_url)}")
        logger.info(f"🔍 EXTRACTED callback_url from result exists: {bool(result_callback_url)}")
        if isinstance(result, dict):
            logger.info(f"🔍 EXTRACTED callback_url from result keys: {list(result.keys())}")
            logger.info(f"🔍 EXTRACTED callback_url from result has callback_url key: {'callback_url' in result}")
        
        # Debug: Log callback_url and result for troubleshooting (without raw data)
        logger.info(f"🔍 callback_url from result: {result_callback_url}")
        logger.info(f"🔍 callback_url type: {type(result_callback_url)}")
        logger.info(f"🔍 result type: {type(result)}")
        if isinstance(result, dict):
            # Create a clean version of result without raw data
            clean_result = {k: v for k, v in result.items() if k not in ['audio_data']}
            if 'audio_data' in result:
                clean_result['audio_data'] = f"[BASE64 DATA] Length: {len(result['audio_data'])} chars"
            
            logger.info(f"🔍 result keys: {list(result.keys())}")
            logger.info(f"🔍 result status: {result.get('status')}")
            logger.info(f"🔍 clean result: {clean_result}")
        else:
            logger.info(f"🔍 result: {result}")
        
        # If callback_url provided, post completion payload
        try:
            # More flexible callback condition: send callback if we have a callback_url and the result doesn't indicate an error
            should_send_callback = (
                result_callback_url and 
                isinstance(result, dict) and 
                result.get("status") != "error"  # Send callback unless explicitly an error
            )
            
            logger.info(f"🔍 CALLBACK CONDITION EVALUATION:")
            logger.info(f"🔍   result_callback_url: {result_callback_url}")
            logger.info(f"🔍   result_callback_url is truthy: {bool(result_callback_url)}")
            logger.info(f"🔍   result is dict: {isinstance(result, dict)}")
            logger.info(f"🔍   result status: {result.get('status') if isinstance(result, dict) else 'N/A'}")
            logger.info(f"🔍   result status != 'error': {result.get('status') != 'error' if isinstance(result, dict) else 'N/A'}")
            logger.info(f"🔍   Should send callback: {should_send_callback}")
            
            # Break down the condition for debugging
            condition1 = bool(result_callback_url)
            condition2 = isinstance(result, dict)
            condition3 = result.get("status") != "error" if isinstance(result, dict) else False
            
            logger.info(f"🔍 CALLBACK CONDITION BREAKDOWN:")
            logger.info(f"🔍   Condition 1 (callback_url exists): {condition1}")
            logger.info(f"🔍   Condition 2 (result is dict): {condition2}")
            logger.info(f"🔍   Condition 3 (status != error): {condition3}")
            logger.info(f"🔍   Final result (all conditions): {condition1 and condition2 and condition3}")
            
            if should_send_callback:
                import requests
                payload = {
                    "story_id": story_id,
                    "user_id": user_id,
                    "voice_id": voice_id,
                    "voice_name": voice_name,
                    "audio_url": result.get("firebase_url") or result.get("audio_url") or result.get("audio_path"),
                    "storage_path": result.get("firebase_path"),
                    "r2_path": result.get("r2_path") or result.get("firebase_path"),  # Explicit R2 path for callback validation
                    "language": language,
                    "metadata": {
                        **({} if not isinstance(api_metadata, dict) else api_metadata),
                        "generation_time": result.get("generation_time"),
                    },
                }
                
                logger.info(f"📤 TTS callback URL: {result_callback_url}")
                logger.info(f"📤 TTS callback payload type: {type(payload)}")
                logger.info(f"📤 TTS callback payload keys: {list(payload.keys())}")
                
                # Create a clean version of payload without raw data
                clean_payload = {k: v for k, v in payload.items() if k not in ['audio_data']}
                if 'audio_data' in payload:
                    clean_payload['audio_data'] = f"[BASE64 DATA] Length: {len(payload['audio_data'])} chars"
                logger.info(f"📤 TTS callback clean payload: {clean_payload}")
                
                # Log nested metadata in callback
                callback_metadata = payload.get("metadata", {})
                logger.info(f"📤 TTS callback metadata: {callback_metadata}")
                logger.info(f"📤 TTS callback metadata type: {type(callback_metadata)}")
                if isinstance(callback_metadata, dict):
                    logger.info(f"📤 TTS callback metadata keys: {list(callback_metadata.keys())}")
                    for key, value in callback_metadata.items():
                        logger.info(f"📤   TTS callback metadata {key}: {value} (type: {type(value)})")
                
                try:
                    logger.info(f"🔍 CALLBACK SENDING:")
                    logger.info(f"🔍   URL: {result_callback_url}")
                    logger.info(f"🔍   Payload keys: {list(payload.keys())}")
                    logger.info(f"🔍   Payload size: {len(str(payload))} characters")
                    
                    # Attempt signed callback with retry; if it ultimately fails, write metadata JSON as fallback
                    callback_success = False
                    try:
                        _post_signed_callback_with_retry(result_callback_url, payload)
                        callback_success = True
                        logger.info(f"✅ TTS callback POST {result_callback_url} -> signed and sent")
                    except Exception as final_cb_e:
                        logger.error(f"❌ Final callback failed after 4 retries: {final_cb_e}")
                        logger.error(f"❌ Callback failure will trigger GPU shutdown to prevent resource waste")
                        
                        # CRITICAL: Send error callback before shutting down GPU so UI can update
                        try:
                            # Derive error callback URL from success callback URL
                            error_callback_url = None
                            if result_callback_url:
                                if "/api/tts/callback" in result_callback_url:
                                    error_callback_url = result_callback_url.replace("/api/tts/callback", "/api/tts/error-callback")
                                elif "/callback" in result_callback_url:
                                    error_callback_url = result_callback_url.rsplit("/", 1)[0] + "/error-callback"
                                else:
                                    base_url = result_callback_url.rstrip("/")
                                    error_callback_url = f"{base_url}/error-callback"
                            
                            if error_callback_url:
                                logger.info(f"📤 Sending error callback for callback failure: {error_callback_url}")
                                notify_error_callback(
                                    error_callback_url=error_callback_url,
                                    story_id=story_id,
                                    error_message=f"Failed to send success callback after 4 retries: {final_cb_e}",
                                    user_id=user_id,
                                    voice_id=voice_id,
                                    error_details=str(final_cb_e),
                                    job_id=input.get('job_id'),
                                    metadata=payload.get("metadata") or {},
                                )
                                logger.info(f"✅ Error callback sent for callback failure")
                        except Exception as error_cb_e:
                            logger.error(f"❌ Failed to send error callback for callback failure: {error_cb_e}")
                        
                        try:
                            _write_metadata_json(
                                (result.get("firebase_path") or ""),
                                {
                                    "story_id": story_id,
                                    "user_id": user_id,
                                    "voice_id": voice_id,
                                    "voice_name": voice_name,
                                    "audio_url": payload.get("audio_url"),
                                    "storage_path": payload.get("storage_path"),
                                    "language": language,
                                    "status": "success",
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "metadata": payload.get("metadata") or {},
                                },
                            )
                        except Exception:
                            pass
                        # CRITICAL: Stop GPU usage by raising exception - this will cause RunPod to stop the worker
                        raise RuntimeError(f"Callback failed after 4 retries. Stopping GPU to prevent resource waste. Error: {final_cb_e}")
                except Exception as cb_e:
                    logger.warning(f"⚠️ TTS callback POST failed: {cb_e}")
                    logger.warning(f"⚠️ TTS callback exception type: {type(cb_e)}")
                    logger.warning(f"⚠️ TTS callback exception details: {str(cb_e)}")
                    import traceback
                    logger.warning(f"⚠️ TTS callback traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.warning(f"⚠️ Error preparing TTS callback: {e}")
            logger.warning(f"⚠️ TTS callback preparation exception type: {type(e)}")
        
        logger.info("🔍 ===== END TTS SUCCESS CALLBACK PAYLOAD =====")
        
        # Final callback status summary
        logger.info(f"🔍 CALLBACK SUMMARY:")
        logger.info(f"🔍   Original callback_url: {callback_url}")
        logger.info(f"🔍   Result callback_url: {result_callback_url}")
        logger.info(f"🔍   Callback sent: {should_send_callback if 'should_send_callback' in locals() else 'Unknown'}")
        logger.info(f"🔍   Result status: {result.get('status') if isinstance(result, dict) else 'N/A'}")
        
        return _return_with_cleanup(result)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        
        # Send error callback if callback_url is available
        try:
            if callback_url:
                # Construct error callback URL from success callback URL
                # Handle different possible callback URL formats
                if "/api/tts/callback" in callback_url:
                    error_callback_url = callback_url.replace("/api/tts/callback", "/api/tts/error-callback")
                elif "/api/tts/" in callback_url:
                    # If it's a different TTS endpoint, replace the last part
                    error_callback_url = callback_url.rsplit("/", 1)[0] + "/error-callback"
                else:
                    # Fallback: append error-callback to the base URL
                    base_url = callback_url.rstrip("/")
                    error_callback_url = f"{base_url}/error-callback"
                
                # Use metadata variables already extracted at the top of the function
                
                # Send error callback
                notify_error_callback(
                    error_callback_url=error_callback_url,
                    story_id=story_id or "unknown",
                    error_message=str(e),
                    error_details=f"Unexpected error in TTS handler: {type(e).__name__}",
                    user_id=user_id,
                    voice_id=voice_id,
                    job_id=event.get("id"),  # RunPod job ID
                    metadata={
                        "language": language,
                        "story_type": story_type,
                        "story_name": story_name,
                        "voice_name": voice_name,
                        "text_length": len(text) if text else 0,
                        "error_type": type(e).__name__
                    }
                )
        except Exception as callback_error:
            logger.error(f"❌ Failed to send error callback: {callback_error}")
        
        return _return_with_cleanup({"status": "error", "error": str(e)})

def handle_file_download(input):
    """Handle file download requests"""
    file_path = input.get("file_path")
    if not file_path:
        return {"status": "error", "error": "file_path is required"}
    
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return {
            "status": "success",
            "file_data": base64.b64encode(file_data).decode('utf-8'),
            "file_size": len(file_data)
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to read file: {e}"}

def list_available_files():
    """List all available files in the directories"""
    files = {}
    
    for directory_name, directory_path in [
        ("voice_profiles", VOICE_PROFILES_DIR),
        ("voice_samples", VOICE_SAMPLES_DIR),
        ("tts_generated", TTS_GENERATED_DIR),
        ("temp_voice", TEMP_VOICE_DIR)
    ]:
        if directory_path.exists():
            files[directory_name] = [f.name for f in directory_path.glob("*")]
        else:
            files[directory_name] = []
    
    return {"status": "success", "files": files}

if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    logger.info("✅ TTS Handler ready")
    runpod.serverless.start({'handler': handler })

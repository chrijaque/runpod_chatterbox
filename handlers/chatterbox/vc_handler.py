import runpod
import time  
import os
import base64
import logging
import sys
import glob
import pathlib
import shutil
import hmac
import hashlib
import json
from urllib.parse import urlparse, urlunparse, urljoin
from urllib.request import Request, HTTPRedirectHandler, build_opener
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch

# Configure logging (default WARNING; opt-in verbose via VERBOSE_LOGS=true)
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""Minimal, production-focused VC handler for RunPod runtime."""

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

        # Prefer HF_HOME/HF_HUB_CACHE; set TRANSFORMERS_CACHE for compatibility
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
        # Non-fatal: proceed without centralized caches
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
    """Log disk usage summary. Only logs when VERBOSE_LOGS is enabled."""
    if not _VERBOSE_LOGS:
        return
    try:
        logger.info(f"Disk usage summary {('(' + context + ')') if context else ''}:")
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
            logger.info(f"  {label:28} {_bytes_human(size_b):>10}  -> {pstr}")
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
    - Always clear temp voice dirs aggressively (older-than policy).
    - If force=True or disk is low, also prune model/tool caches.
    """
    try:
        # Disabled by default; enable via ENABLE_STORAGE_MAINTENANCE=true
        if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() != "true":
            return
        # Temp/work directories
        temp_dirs = [
            Path("/temp_voice"),
            Path("/voice_samples"),
            Path("/voice_profiles"),
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
        logger.info(f"Cleanup done. Free space: {_bytes_human(free_after)}")
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
        if _VERBOSE_LOGS:
            logger.info(f"Free disk before check: {_bytes_human(free_before)}")
        if free_before < int(min_free_gb * (1024 ** 3)):
            logger.warning(f"Low disk space detected (<{min_free_gb} GB). Running cleanup...")
            cleanup_runtime_storage(force=True)
    except Exception:
        pass

# Initialize cache env as early as possible
_ensure_cache_env_dirs()

# Early, pre-import disk headroom preflight (runs before any model downloads)
try:
    if os.getenv("ENABLE_STORAGE_MAINTENANCE", "false").lower() == "true":
        _pre_free = _disk_free_bytes("/")
        if _VERBOSE_LOGS:
            logger.info(f"Free disk early preflight: {_bytes_human(_pre_free)}")
        _min_gb = float(os.getenv("MIN_FREE_GB", "10"))
        if _pre_free < int(_min_gb * (1024 ** 3)):
            logger.warning(f"Low disk space detected in preflight (<{_min_gb} GB). Running cleanup...")
            log_disk_usage_summary("preflight_before_cleanup")
            cleanup_runtime_storage(force=True)
            log_disk_usage_summary("preflight_after_cleanup")
except Exception:
    pass

def clear_python_cache():
    """Clear Python caches and loaded chatterbox modules to ensure fresh load."""
    try:
        # Remove .pyc files
        for pyc_file in glob.glob("/workspace/**/*.pyc", recursive=True):
            try:
                os.remove(pyc_file)
            except Exception:
                pass
        # Remove __pycache__ directories
        for pycache_dir in pathlib.Path("/workspace").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
            except Exception:
                pass
        # Clear chatterbox entries from sys.modules
        to_clear = [name for name in list(sys.modules.keys()) if 'chatterbox' in name]
        for name in to_clear:
            del sys.modules[name]
    except Exception:
        pass

# Clear cache BEFORE importing any chatterbox modules
clear_python_cache()

# Import the models from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    from chatterbox.tts import ChatterboxTTS
    FORKED_HANDLER_AVAILABLE = True
    logger.info("Successfully imported ChatterboxVC and ChatterboxTTS from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"Could not import models from forked repository: {e}")

# Initialize models once at startup
vc_model = None
tts_model = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples") 
TEMP_VOICE_DIR = Path("/temp_voice")

# Create directories if they don't exist (RunPod deployment)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

# Ensure generic temp dir is under our managed directory
os.environ.setdefault("TMPDIR", str(TEMP_VOICE_DIR))

if _VERBOSE_LOGS:
    logger.info(f"Using directories:")
    logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
    logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
    logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# -------------------------------------------------------------
# Device selection with robust fallback
# -------------------------------------------------------------
def _select_device() -> str:
    """Choose execution device honoring env overrides and runtime availability."""
    try:
        forced = (os.getenv("VC_DEVICE") or os.getenv("DAEZEND_DEVICE") or os.getenv("DEVICE") or "").lower()
        if forced in ("cpu", "cuda", "mps"):
            return forced
        if forced == "auto":
            forced = ""
    except Exception:
        forced = ""

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"

# Initialize models
ensure_disk_headroom()
logger.info("Initializing models...")
try:
    if FORKED_HANDLER_AVAILABLE:
        _device = _select_device()
        logger.info(f"Selected device: {_device}")
        try:
            tts_model = ChatterboxTTS.from_pretrained(device=_device)
            logger.info("ChatterboxTTS ready")
            vc_model = ChatterboxVC.from_pretrained(device=_device)
            logger.info("ChatterboxVC ready")
        except Exception as dev_e:
            logger.error(f"Init failed on {_device}: {dev_e}. Retrying on CPU…")
            try:
                tts_model = ChatterboxTTS.from_pretrained(device='cpu')
                vc_model = ChatterboxVC.from_pretrained(device='cpu')
                logger.info("Models initialized on CPU")
            except Exception as cpu_e:
                logger.error(f"CPU fallback init failed: {cpu_e}")
                vc_model = None
                tts_model = None
    else:
        logger.error("Forked repository models not available")
        vc_model = None
        tts_model = None
        
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    vc_model = None
    tts_model = None

# -------------------------------------------------------------------
# R2 download helper
# -------------------------------------------------------------------
def download_from_r2(source_key: str) -> Optional[bytes]:
    """
    Download data from Cloudflare R2 using boto3 S3 client.
    
    :param source_key: Source key/path in R2 (e.g., "private/users/{user_id}/voices/{lang}/recorded/{file}.wav")
    :return: Binary data or None if failed
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Get R2 credentials from environment
        r2_account_id = os.getenv('R2_ACCOUNT_ID')
        r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
        r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
        r2_endpoint = os.getenv('R2_ENDPOINT')
        r2_bucket_name = os.getenv('R2_BUCKET_NAME', 'daezend-public-content')
        
        if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
            logger.error("R2 credentials not configured")
            return None
        
        # Create S3 client for R2
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            region_name='auto'
        )
        
        # Download from R2
        response = s3_client.get_object(
            Bucket=r2_bucket_name,
            Key=source_key
        )
        
        data = response['Body'].read()
        logger.info(f"Downloaded from R2: {source_key} ({len(data)} bytes)")
        return data
        
    except Exception as e:
        logger.error(f"R2 download failed: {e}")
        if _VERBOSE_LOGS:
            import traceback
            logger.error(f"R2 download traceback: {traceback.format_exc()}")
        return None

def get_voice_id(name):
    """Generate a unique ID for a voice based on the name"""
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name.lower().replace(' ', '_'))
    return f"voice_{clean_name}"

# -------------------------------------------------------------------
# Callback helpers
# -------------------------------------------------------------------
def send_error_callback(callback_url: Optional[str], user_id: Optional[str], voice_id: str,
                       voice_name: str, language: str, error: str) -> None:
    """Send error callback if URL is provided. Silently fails on errors."""
    if not callback_url:
        return
    
    try:
        payload = {
            'status': 'error',
            'user_id': user_id,
            'voice_id': voice_id,
            'voice_name': voice_name,
            'language': language,
            'error': error,
        }
        _post_signed_callback(callback_url, payload)
        if _VERBOSE_LOGS:
            logger.info(f"Error callback sent for voice_id={voice_id}")
    except Exception as e:
        logger.warning(f"Failed to send error callback: {e}")

def send_success_callback(callback_url: Optional[str], result: Dict[str, Any], user_id: Optional[str],
                         voice_id: str, voice_name: str, language: str, is_kids_voice: bool,
                         input_data: Dict[str, Any], audio_path: Optional[str] = None) -> None:
    """Send success callback if URL is provided. Silently fails on errors."""
    if not callback_url or not isinstance(result, dict) or result.get("status") == "error":
        return
    
    try:
        profile_storage_path = result.get('profile_storage_path')
        sample_storage_path = result.get('sample_storage_path')
        
        # Build storage paths - use R2 paths if available, otherwise fallback to old format
        kids_segment = 'kids/' if is_kids_voice else ''
        target_profile_name = result.get('profile_filename') or f"{voice_id}.npy"
        target_sample_name = result.get('sample_filename') or f"{voice_id}.mp3"
        
        profile_path = profile_storage_path or f"audio/voices/{language}/{kids_segment}profiles/{target_profile_name}"
        sample_path = sample_storage_path or f"audio/voices/{language}/{kids_segment}samples/{target_sample_name}"
        
        payload = {
            'status': 'success',
            'user_id': user_id,
            'voice_id': voice_id,
            'voice_name': voice_name,
            'language': language,
            'is_kids_voice': bool(is_kids_voice),
            'model_type': input_data.get('model_type') or 'chatterbox',
            'profile_path': profile_path,
            'sample_path': sample_path,
            'r2_profile_path': profile_storage_path,
            'r2_sample_path': sample_storage_path,
            'recorded_path': audio_path or result.get('recorded_audio_path', ''),
        }
        _post_signed_callback(callback_url, payload)
        if _VERBOSE_LOGS:
            logger.info(f"Success callback sent for voice_id={voice_id}")
    except Exception as e:
        logger.warning(f"Failed to send success callback: {e}")

# -------------------------------------------------------------------
# Metadata extraction
# -------------------------------------------------------------------
def extract_request_metadata(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all request metadata."""
    meta_top = input_data.get('metadata', {}) if isinstance(input_data.get('metadata'), dict) else {}
    
    return {
        'callback_url': input_data.get('callback_url') or meta_top.get('callback_url'),
        'user_id': input_data.get('user_id') or meta_top.get('user_id'),
        'voice_id': input_data.get('voice_id') or meta_top.get('voice_id'),
        'profile_filename': input_data.get('profile_filename') or meta_top.get('profile_filename'),
        'sample_filename': input_data.get('sample_filename') or meta_top.get('sample_filename'),
    }

# -------------------------------------------------------------------
# Audio file preparation
# -------------------------------------------------------------------
def _download_audio_file(audio_path: str, voice_id: str) -> Path:
    """Download audio from R2. Returns Path to temp file."""
    # Infer extension from path
    lower = str(audio_path).lower()
    ext = ".wav"
    if lower.endswith(".mp3"):
        ext = ".mp3"
    elif lower.endswith(".ogg"):
        ext = ".ogg"
    elif lower.endswith(".m4a"):
        ext = ".m4a"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}{ext}"
    
    if _VERBOSE_LOGS:
        logger.info(f"Downloading audio from R2: {audio_path}")
    
    # Download from R2
    audio_data = download_from_r2(audio_path)
    if audio_data is None:
        raise RuntimeError(f"Failed to download audio from R2: {audio_path}")
    
    # Write to temp file
    with open(temp_voice_file, 'wb') as f:
        f.write(audio_data)
    
    logger.info(f"Downloaded audio from R2 and saved to temp file")
    return temp_voice_file

def _prepare_audio_file(input_data: Dict[str, Any], voice_id: str) -> Path:
    """Prepare audio file from either base64 or R2 path."""
    audio_data = input_data.get('audio_data')
    audio_path = input_data.get('audio_path')
    audio_format = input_data.get('audio_format', 'wav')
    
    if audio_path:
        # Download from R2
        return _download_audio_file(audio_path, voice_id)
    else:
        # Decode base64 data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
        audio_bytes = base64.b64decode(audio_data)
        with open(temp_voice_file, 'wb') as f:
            f.write(audio_bytes)
        logger.info(f"Saved temporary voice file from base64 data")
        return temp_voice_file

# -------------------------------------------------------------------
# API metadata preparation
# -------------------------------------------------------------------
def _build_api_metadata(input_data: Dict[str, Any], voice_id: str, name: str,
                       language: str, is_kids_voice: bool, audio_path: Optional[str],
                       metadata: Dict[str, Any], target_profile_name: str,
                       target_sample_name: str) -> Dict[str, Any]:
    """Build API metadata for VC model. Uploads go to R2 at:
    - daezend-public-content/private/users/{user_id}/voices/{lang}/profiles/{voice_id}.npy
    - daezend-public-content/private/users/{user_id}/voices/{lang}/samples/{voice_id}.mp3
    """
    return {
        'user_id': input_data.get('user_id'),
        'project_id': input_data.get('project_id'),
        'voice_type': input_data.get('voice_type'),
        'quality': input_data.get('quality'),
        'language': language,
        'is_kids_voice': is_kids_voice,
        'recorded_path': audio_path if audio_path else None,
        'profile_filename': target_profile_name,
        'sample_filename': target_sample_name,
        'storage_metadata': {
            'user_id': input_data.get('user_id') or '',
            'voice_id': voice_id,
            'voice_name': name,
            'language': language,
            'is_kids_voice': str(is_kids_voice).lower(),
            'model_type': input_data.get('model_type') or 'chatterbox',
        }
    }

# -------------------------------------------------------------------
# VC model call
# -------------------------------------------------------------------
def call_vc_model_create_voice_clone(audio_file_path: Path, voice_id: str, voice_name: str,
                                    api_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implement voice cloning using available model methods.
    
    Uses the VC model's create_voice_clone method to create voice profiles.
    """
    global vc_model, tts_model
    
    start_time = time.time()
    
    try:
        # Check if models are available
        if vc_model is None or tts_model is None:
            logger.error("Models not available")
            return {
                "status": "error",
                "error": "Models not available",
                "generation_time": time.time() - start_time
            }
        
        if not hasattr(vc_model, 'create_voice_clone'):
            return {
                "status": "error",
                "error": "VC model missing create_voice_clone",
                "generation_time": time.time() - start_time
            }
        
        result = vc_model.create_voice_clone(
            audio_file_path=str(audio_file_path),
            voice_id=voice_id,
            voice_name=voice_name,
            metadata=api_metadata
        )
        generation_time = time.time() - start_time
        logger.info(f"Voice clone completed in {generation_time:.2f}s")
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"Voice clone failed after {generation_time:.2f}s: {e}")
        return {
            "status": "error",
            "error": str(e),
            "generation_time": generation_time
        }

# -------------------------------------------------------------------
# Main handler
# -------------------------------------------------------------------
def handle_voice_clone_request(input_data: Dict[str, Any], response_format: str) -> Dict[str, Any]:
    """Handle voice cloning requests - orchestrates the flow.
    
    Flow:
    1. User sends request with audio_path (R2 path) or audio_data (base64)
    2. RunPod downloads audio from R2 if audio_path provided
    3. Creates voice profile (.npy) and audio sample
    4. VC model uploads to R2 at:
       - daezend-public-content/private/users/{user_id}/voices/{lang}/profiles/{voice_id}.npy
       - daezend-public-content/private/users/{user_id}/voices/{lang}/samples/{voice_id}.mp3
    """
    global vc_model
    
    # Ensure we have disk headroom before doing any significant work
    ensure_disk_headroom()
    
    if _VERBOSE_LOGS:
        logger.debug("VC handler input", extra={"input_keys": list(input_data.keys())})
    
    # Extract metadata once
    metadata = extract_request_metadata(input_data)
    
    # Validate inputs
    name = input_data.get('name')
    audio_data = input_data.get('audio_data')
    audio_path = input_data.get('audio_path')
    
    if not name or (not audio_data and not audio_path):
        return {"status": "error", "error": "name and either audio_data or audio_path are required"}
    
    # Check if VC model is available
    if vc_model is None:
        logger.error("VC model not available")
        send_error_callback(
            metadata['callback_url'], metadata['user_id'],
            input_data.get('voice_id', 'unknown'), name or 'unknown',
            input_data.get('language', 'en'), "VC model not available"
        )
        return {"status": "error", "error": "VC model not available"}
    
    language = input_data.get('language', 'en')
    is_kids_voice = input_data.get('is_kids_voice', False)
    
    logger.info(f"Voice clone request: name={name}, language={language}, kids_voice={is_kids_voice}")
    
    try:
        # Determine voice_id
        voice_id = metadata['voice_id'] or get_voice_id(name)
        
        # Determine target filenames
        target_profile_name = metadata['profile_filename'] or f"{voice_id}.npy"
        target_sample_name = metadata['sample_filename'] or f"{voice_id}.mp3"
        
        # Prepare audio file (downloads from R2 if audio_path provided, otherwise decodes base64)
        temp_audio_file = _prepare_audio_file(input_data, voice_id)
        
        # Build API metadata
        api_metadata = _build_api_metadata(
            input_data, voice_id, name, language, is_kids_voice,
            audio_path, metadata, target_profile_name, target_sample_name
        )
        
        # Call VC model
        result = call_vc_model_create_voice_clone(
            audio_file_path=temp_audio_file,
            voice_id=voice_id,
            voice_name=name,
            api_metadata=api_metadata
        )
        
        # Cleanup temp file
        try:
            temp_audio_file.unlink(missing_ok=True)
        except Exception:
            pass
        
        # Store callback_url in result for reliable access
        if isinstance(result, dict):
            result["callback_url"] = metadata['callback_url']
        
        # Handle callbacks
        if isinstance(result, dict) and result.get("status") == "error":
            send_error_callback(
                metadata['callback_url'], metadata['user_id'], voice_id,
                name, language, result.get('error', 'Unknown error')
            )
        else:
            # Post-process result: if caller supplied audio_path, ensure recorded_audio_path reflects it
            if isinstance(result, dict) and audio_path:
                result.setdefault('metadata', {})
                result['recorded_audio_path'] = audio_path
            
            send_success_callback(
                metadata['callback_url'], result, metadata['user_id'],
                voice_id, name, language, is_kids_voice, input_data, audio_path
            )
        
        # Opportunistic cleanup after job
        try:
            cleanup_runtime_storage(force=False)
        except Exception:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Voice clone request failed: {e}", exc_info=True)
        
        # Cleanup temp file if it exists
        if 'temp_audio_file' in locals():
            try:
                temp_audio_file.unlink(missing_ok=True)
            except Exception:
                pass
        
        send_error_callback(
            metadata['callback_url'], metadata['user_id'],
            metadata.get('voice_id', ''), name or 'unknown',
            language, str(e)
        )
        
        try:
            cleanup_runtime_storage(force=False)
        except Exception:
            pass
        
        return {"status": "error", "error": str(e)}

def handler(event, responseFormat="base64"):
    """RunPod handler entry point."""
    input_data = event['input']
    
    # This handler is for voice cloning only
    result = handle_voice_clone_request(input_data, responseFormat)
    
    try:
        # Emit a structured error line for system log collectors if error
        if isinstance(result, dict) and result.get("status") == "error":
            req_id = event.get('id') or event.get('requestId')
            msg = {
                "requestId": req_id,
                "message": f"Error: {result.get('error', 'Unknown error')}",
                "level": "ERROR",
            }
            print(json.dumps(msg, ensure_ascii=False))
    except Exception:
        pass
    
    return result

# -------------------------------------------------------------------
# Callback POST implementation
# -------------------------------------------------------------------
def _canonicalize_callback_url(url: str) -> str:
    """Canonicalize callback URL to avoid 307 redirects (prefer www.daezend.app)."""
    try:
        p = urlparse(url)
        scheme = p.scheme or 'https'
        netloc = p.netloc
        if netloc == 'daezend.app':
            netloc = 'www.daezend.app'
        if not netloc and p.path:
            return f'https://www.daezend.app{p.path}'
        return urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return url

def _post_signed_callback(callback_url: str, payload: Dict[str, Any]):
    """POST JSON payload to callback_url with HMAC headers compatible with app callback."""
    logger.info(f"📤 Posting callback to {callback_url}")
    if _VERBOSE_LOGS:
        logger.debug(f"Posting callback to {callback_url}")
    
    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
    if not secret:
        logger.warning("DAEZEND_API_SHARED_SECRET not set; unsigned callback")
    
    canonical_url = _canonicalize_callback_url(callback_url)
    parsed = urlparse(canonical_url)
    path_for_signing = parsed.path or '/api/voice-clone/callback'  # Fixed default path
    logger.info(f"📤 Callback signing path: {path_for_signing}, canonical URL: {canonical_url}")
    ts = str(int(time.time() * 1000))
    
    body_bytes = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    
    if secret:
        prefix = f"POST\n{path_for_signing}\n{ts}\n".encode('utf-8')
        message = prefix + body_bytes
        sig = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
        headers.update({
            'X-Daezend-Timestamp': ts,
            'X-Daezend-Signature': sig,
        })
    
    # Configure HTTP opener to follow redirects
    redirect_handler = HTTPRedirectHandler()
    opener = build_opener(redirect_handler)
    req = Request(canonical_url, data=body_bytes, headers=headers, method='POST')
    
    try:
        resp = opener.open(req, timeout=15)
        response_data = resp.read()
        logger.info(f"✅ Callback response status: {resp.status}, URL: {canonical_url}")
        if _VERBOSE_LOGS:
            logger.debug(f"Callback response status: {resp.status}")
        resp.close()
    except HTTPError as http_err:
        # Explicitly handle 307/308 by re-posting to Location
        code = getattr(http_err, 'code', None)
        loc = None
        try:
            loc = http_err.headers.get('Location') if hasattr(http_err, 'headers') and http_err.headers else None
        except Exception:
            loc = None
        
        logger.error(f"❌ Callback HTTP error: {code} for URL: {canonical_url}, Location: {loc}")
        if code in (307, 308) and loc:
            try:
                follow_url = urljoin(canonical_url, loc)
                logger.info(f"Following {code} redirect to: {follow_url}")
                req2 = Request(follow_url, data=body_bytes, headers=headers, method='POST')
                resp2 = opener.open(req2, timeout=15)
                resp2.read()
                resp2.close()
                logger.info(f"✅ Callback redirect succeeded: {follow_url}")
                return
            except Exception as follow_e:
                logger.error(f"Redirect follow failed: {type(follow_e).__name__}: {follow_e}")
                raise
        else:
            logger.error(f"HTTP request failed: {type(http_err).__name__}: {http_err}, URL: {canonical_url}")
            raise
    except Exception as e:
        logger.error(f"❌ Callback request failed: {e}, URL: {canonical_url}")
        raise

if __name__ == '__main__':
    logger.info("Voice Clone Handler starting...")
    logger.info("Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler})

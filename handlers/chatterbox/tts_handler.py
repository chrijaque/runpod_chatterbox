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
    
    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
    if not secret:
        logger.error("❌ DAEZEND_API_SHARED_SECRET not set; cannot sign callback")
        raise RuntimeError('DAEZEND_API_SHARED_SECRET not set; cannot sign callback')
    
    logger.info(f"🔍 DAEZEND_API_SHARED_SECRET exists: {bool(secret)}")
    logger.info(f"🔍 DAEZEND_API_SHARED_SECRET length: {len(secret) if secret else 0}")

    # Canonicalize callback URL to avoid 307 redirects (prefer www.daezend.app)
    def _canonicalize_callback_url(url: str) -> str:
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
        'X-Daezend-Timestamp': ts,
        'X-Daezend-Signature': sig,
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
    from chatterbox.storage import resolve_bucket_name, is_r2_bucket, upload_to_r2, download_from_r2
    logger.info("✅ Successfully imported storage utilities from chatterbox")
except ImportError as e:
    logger.warning(f"⚠️ Could not import storage utilities: {e}")
    # Fallback: define functions locally if import fails
    def resolve_bucket_name(bucket_name: Optional[str] = None, country_code: Optional[str] = None) -> str:
        return os.getenv('R2_BUCKET_NAME', 'daezend-public-content')
    def is_r2_bucket(bucket_name: str) -> bool:
        return bucket_name == 'daezend-public-content' or bucket_name.startswith('r2://')
    def upload_to_r2(data: bytes, destination_key: str, content_type: str = "application/octet-stream", metadata: dict = None, bucket_name: Optional[str] = None) -> Optional[str]:
        logger.error("Storage utilities not available - upload_to_r2 not implemented")
        return None
    def download_from_r2(source_key: str) -> Optional[bytes]:
        logger.error("Storage utilities not available - download_from_r2 not implemented")
        return None

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
logger.info("🔧 Initializing models...")
try:
    if FORKED_HANDLER_AVAILABLE:
        # Initialize TTS model first (needed for s3gen)
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS ready")
        
        # Initialize VC model using the correct method
        vc_model = ChatterboxVC.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxVC ready")
        
    else:
        logger.error("❌ Forked repository models not available")
        vc_model = None
        tts_model = None
        
except Exception as e:
    logger.error(f"❌ Failed to initialize models: {e}")
    vc_model = None
    tts_model = None

# -------------------------------------------------------------------
# 🐞  Firebase / GCS credential debug helper
# -------------------------------------------------------------------
def _debug_gcs_creds():
    """Comprehensive Firebase credential check and validation."""
    logger.info("🔍 ===== TTS FIREBASE CREDENTIAL VALIDATION =====")
    try:
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        logger.info(f"🔑 Firebase secret present: {bool(firebase_secret_path)}")
        logger.info(f"🔑 Firebase secret length: {len(firebase_secret_path) if firebase_secret_path else 0}")
        
        if firebase_secret_path:
            # Check if it's JSON content
            if firebase_secret_path.startswith('{'):
                logger.info("🔑 Firebase secret appears to be JSON content")
                try:
                    import json
                    cred_data = json.loads(firebase_secret_path)
                    logger.info(f"🔑 JSON validation: SUCCESS")
                    logger.info(f"🔑 Project ID: {cred_data.get('project_id', 'NOT FOUND')}")
                    logger.info(f"🔑 Client Email: {cred_data.get('client_email', 'NOT FOUND')}")
                    logger.info(f"🔑 Private Key ID: {cred_data.get('private_key_id', 'NOT FOUND')}")
                    logger.info(f"🔑 Type: {cred_data.get('type', 'NOT FOUND')}")
                    
                    # Check for required fields
                    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                    missing_fields = [field for field in required_fields if field not in cred_data]
                    if missing_fields:
                        logger.error(f"❌ Missing required credential fields: {missing_fields}")
                    else:
                        logger.info("✅ All required credential fields present")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Firebase secret JSON is invalid: {e}")
                except Exception as e:
                    logger.error(f"❌ Error parsing Firebase secret: {e}")
            else:
                logger.info("🔑 Firebase secret appears to be a file path")
                if os.path.exists(firebase_secret_path):
                    logger.info(f"✅ Firebase secret file exists: {firebase_secret_path}")
                    try:
                        with open(firebase_secret_path, 'r') as f:
                            content = f.read()
                            logger.info(f"🔑 File content length: {len(content)}")
                            logger.info(f"🔑 File content preview: {content[:100]}...")
                    except Exception as e:
                        logger.error(f"❌ Error reading Firebase secret file: {e}")
                else:
                    logger.error(f"❌ Firebase secret file does not exist: {firebase_secret_path}")
        else:
            logger.error("❌ RUNPOD_SECRET_Firebase environment variable not set")
            
        # Check bucket identifier
        bucket_name_raw = os.getenv('GCS_BUCKET_US') or os.getenv('FIREBASE_STORAGE_BUCKET') or ''
        bucket_name = bucket_name_raw if bucket_name_raw else 'not-set'
        logger.info(f"🔑 Bucket identifier: {bucket_name}")
        if bucket_name != 'not-set':
            # Normalize to show project ID
            normalized = bucket_name.replace('.firebasestorage.app', '').replace('.appspot.com', '')
            logger.info(f"🔑 Bucket project ID: {normalized}")
        
    except Exception as e:
        logger.error(f"❌ Firebase credential validation failed: {e}")
    
    logger.info("🔍 ===== END TTS FIREBASE CREDENTIAL VALIDATION =====")

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
            extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
        
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
    
    # Extract user_id and story_id from metadata if not provided explicitly
    if not user_id:
        user_id = api_metadata.get("user_id") if isinstance(api_metadata, dict) else ""
    if not story_id:
        story_id = api_metadata.get("story_id") if isinstance(api_metadata, dict) else ""
    
    logger.info(f"🎯 ===== CALLING TTS GENERATION =====")
    logger.info(f"🔍 Parameters:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  voice_name: {voice_name}")
    logger.info(f"  language: {language}")
    logger.info(f"  story_type: {story_type}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  text_length: {len(text)} characters")
    logger.info(f"  profile_path: {profile_path}")
    logger.info(f"  user_id: {user_id}")
    logger.info(f"  story_id: {story_id}")
    
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
                    story_id=story_id
                )
                generation_time = time.time() - start_time
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
    profile_path = event["input"].get("profile_path") or (api_metadata.get("profile_path") if isinstance(api_metadata, dict) else None)
    callback_url = api_metadata.get("callback_url") or (event["metadata"].get("callback_url") if isinstance(event.get("metadata"), dict) else None)
    
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
    
    # ===== FIREBASE CREDENTIAL VALIDATION =====
    _debug_gcs_creds()
    
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
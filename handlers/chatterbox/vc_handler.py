import runpod
import time  
import os
import tempfile
import base64
import logging
import sys
import glob
import pathlib
import shutil
import hmac
import hashlib
import json
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from typing import Optional
import torch

def resolve_bucket_name(bucket_name: Optional[str] = None, country_code: Optional[str] = None) -> str:
    """
    Normalize and resolve the target GCS bucket name for uploads.
    Priority:
      1) Explicit bucket_name (strip gs:// prefix and Firebase Storage domain suffixes)
      2) AU if country_code == 'AU' and AU env present
      3) US default
    
    Normalizes Firebase Storage domain formats to actual GCS bucket names:
    - godnathistorie-a25fa.firebasestorage.app -> godnathistorie-a25fa
    - godnathistorie-a25fa.appspot.com -> godnathistorie-a25fa
    - australia-a25fa -> australia-a25fa
    """
    if bucket_name:
        bn = str(bucket_name).replace('gs://', '')
    elif (country_code or '').upper() == 'AU':
        bn = os.getenv('GCS_BUCKET_AU') or os.getenv('FIREBASE_STORAGE_BUCKET_AU') or ''
    else:
        bn = os.getenv('GCS_BUCKET_US') or os.getenv('FIREBASE_STORAGE_BUCKET') or ''
    bn = (bn or '').strip()
    # Basic validation: forbid slashes and stray prefixes
    if bn.startswith('gs://'):
        bn = bn.replace('gs://', '')
    # Strip protocol if present
    if bn.startswith('https://') or bn.startswith('http://'):
        bn = bn.split('://', 1)[1]
    # If URL-like, take host part only
    if '/' in bn:
        bn = bn.split('/')[0]
    # Strip Firebase Storage domain suffixes (GCS client needs actual bucket name)
    if bn.endswith('.firebasestorage.app'):
        bn = bn.replace('.firebasestorage.app', '')
    if bn.endswith('.appspot.com'):
        bn = bn.replace('.appspot.com', '')
    if '/' in bn or '\\' in bn:
        raise ValueError(f"Invalid bucket name (contains slash): {bn}")
    if not bn:
        raise ValueError("Bucket name could not be resolved from inputs or environment")
    return bn

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

# Keep model downloads simple: no monkey patches to huggingface_hub

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
VOICE_SAMPLES_DIR = Path("/voice_samples") 
TEMP_VOICE_DIR = Path("/temp_voice")

# Create directories if they don't exist (RunPod deployment)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

# Ensure generic temp dir is under our managed directory
os.environ.setdefault("TMPDIR", str(TEMP_VOICE_DIR))

logger.info(f"Using directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Initialize Firebase storage client
storage_client = None
bucket = None

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
logger.info("🔧 Initializing models...")
try:
    if FORKED_HANDLER_AVAILABLE:
        _device = _select_device()
        logger.info(f"🖥️ Selected device: {_device}")
        try:
            tts_model = ChatterboxTTS.from_pretrained(device=_device)
            logger.info("✅ ChatterboxTTS ready")
            vc_model = ChatterboxVC.from_pretrained(device=_device)
            logger.info("✅ ChatterboxVC ready")
        except Exception as dev_e:
            logger.error(f"❌ Init failed on {_device}: {dev_e}. Retrying on CPU…")
            try:
                tts_model = ChatterboxTTS.from_pretrained(device='cpu')
                vc_model = ChatterboxVC.from_pretrained(device='cpu')
                logger.info("✅ Models initialized on CPU")
            except Exception as cpu_e:
                logger.error(f"❌ CPU fallback init failed: {cpu_e}")
                vc_model = None
                tts_model = None

        
        
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
    logger.info("🔍 ===== FIREBASE CREDENTIAL VALIDATION =====")
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
    
    logger.info("🔍 ===== END FIREBASE CREDENTIAL VALIDATION =====")

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    try:
        firebase_secret = os.getenv('RUNPOD_SECRET_Firebase')
        firebase_secret_path = firebase_secret
        if firebase_secret and firebase_secret.startswith('{'):
            import json
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(json.loads(firebase_secret), tmp_file)
                firebase_secret_path = tmp_file.name
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = firebase_secret_path
        if firebase_secret_path and os.path.exists(firebase_secret_path):
            client = storage.Client.from_service_account_json(firebase_secret_path)
        else:
            client = storage.Client()
        storage_client = client
        # Normalize bucket name: strip .firebasestorage.app suffix for GCS client
        bucket_name_raw = os.getenv('GCS_BUCKET_US') or os.getenv('FIREBASE_STORAGE_BUCKET') or ''
        if not bucket_name_raw:
            raise ValueError("GCS_BUCKET_US or FIREBASE_STORAGE_BUCKET environment variable must be set")
        # Normalize: strip gs://, https://, .firebasestorage.app, .appspot.com
        bucket_name = bucket_name_raw.replace('gs://', '').replace('https://', '').replace('http://', '')
        if '/' in bucket_name:
            bucket_name = bucket_name.split('/')[0]
        bucket_name = bucket_name.replace('.firebasestorage.app', '').replace('.appspot.com', '')
        bucket = storage_client.bucket(bucket_name)
        logger.info(f"✅ Firebase storage client ready (bucket: {bucket_name})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase storage: {e}")
        return False

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """
    Upload data directly to Firebase Storage with metadata
    
    :param data: Binary data to upload
    :param destination_blob_name: Destination path in Firebase
    :param content_type: MIME type of the file
    :param metadata: Optional metadata to store with the file
    :return: Public URL or None if failed
    """
    try:
        from google.cloud import storage
        
        # Resolve region-aware bucket from metadata hints
        bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
        country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
        resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)
        
        logger.info(f"🔍 Resolved bucket: {resolved_bucket} (from bucket_hint={bucket_hint}, country_hint={country_hint})")
        
        # Initialize Firebase storage client and bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(resolved_bucket)
        
        logger.info(f"🔍 Creating blob: {destination_blob_name}")
        logger.info(f"🔍 Uploading {len(data)} bytes to bucket {resolved_bucket}...")
        
        blob = bucket.blob(destination_blob_name)
        
        # Set metadata if provided
        if metadata:
            blob.metadata = metadata
            logger.info(f"🔍 Set metadata: {metadata}")
        
        # Set content type
        blob.content_type = content_type
        logger.info(f"🔍 Set content type: {content_type}")
        
        # Upload the data
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"🔍 Upload completed, making public...")
        
        # Make the blob publicly accessible
        blob.make_public()
        
        # CRITICAL: Patch metadata to ensure persistence
        if metadata:
            try:
                blob.patch()
                logger.info(f"✅ Metadata patched successfully for: {destination_blob_name}")
            except Exception as patch_e:
                logger.error(f"❌ Failed to patch metadata for {destination_blob_name}: {patch_e}")
        
        public_url = blob.public_url
        logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url} (bucket: {resolved_bucket})")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Firebase upload failed: {e}")
        import traceback
        logger.error(f"❌ Firebase upload traceback: {traceback.format_exc()}")
        return None

def rename_in_firebase(src_path: str, dest_path: str, *, metadata: Optional[dict] = None, content_type: Optional[str] = None) -> Optional[str]:
    """
    Copy a blob to a new destination (rename), set metadata, make public, then delete the old blob.
    Returns new public URL or None.
    """
    global bucket
    try:
        if bucket is None and not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot rename")
            return None
        src_blob = bucket.blob(src_path)
        if not src_blob.exists():
            logger.warning(f"⚠️ Source blob does not exist: {src_path}")
            return None
        # Perform copy
        new_blob = bucket.copy_blob(src_blob, bucket, dest_path)
        # Set metadata if provided
        if metadata:
            new_blob.metadata = metadata
            logger.info(f"🔍 Set metadata on renamed blob: {metadata}")
        # Set content type if provided
        if content_type:
            new_blob.content_type = content_type
        new_blob.make_public()
        
        # CRITICAL: Patch metadata to ensure persistence
        if metadata:
            try:
                new_blob.patch()
                logger.info(f"✅ Metadata patched successfully for renamed blob: {dest_path}")
            except Exception as patch_e:
                logger.error(f"❌ Failed to patch metadata for renamed blob {dest_path}: {patch_e}")
        
        # Delete original
        try:
            src_blob.delete()
        except Exception as del_e:
            logger.warning(f"⚠️ Could not delete original blob {src_path}: {del_e}")
        logger.info(f"✅ Renamed {src_path} → {dest_path}")
        return new_blob.public_url
    except Exception as e:
        logger.error(f"❌ Rename failed {src_path} → {dest_path}: {e}")
        return None

def get_voice_id(name):
    """Generate a unique ID for a voice based on the name"""
    # Create a clean, filesystem-safe voice ID from the name
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name.lower().replace(' ', '_'))
    return f"voice_{clean_name}"

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def call_vc_model_create_voice_clone(audio_file_path, voice_id, voice_name, language="en", is_kids_voice=False, api_metadata=None):
    """
    Implement voice cloning using available model methods.
    
    Uses the TTS model's save_voice_clone method to create voice profiles.
    """
    global vc_model, tts_model
    
    logger.info("VC clone: voice_id=%s, name=%s, lang=%s, kids=%s", voice_id, voice_name, language, is_kids_voice)
    
    start_time = time.time()
    
    try:
        # Check if models are available
        if vc_model is None or tts_model is None:
            logger.error("❌ Models not available")
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
        logger.info("✅ Voice clone completed in %.2fs", generation_time)
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Voice clone failed after {generation_time:.2f}s: {e}")
        return {
            "status": "error",
            "error": str(e),
            "generation_time": generation_time
        }

def handler(event, responseFormat="base64"):
    input = event['input']    
    
    # This handler is for voice cloning only
    result = handle_voice_clone_request(input, responseFormat)
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

def handle_voice_clone_request(input, responseFormat):
    """Pure API orchestration: Handle voice cloning requests"""
    global vc_model
    
    # Ensure we have disk headroom before doing any significant work
    ensure_disk_headroom()

    # ===== COMPREHENSIVE INPUT PARAMETER LOGGING =====
    logger.info("🔍 ===== VC HANDLER INPUT PARAMETERS =====")
    logger.info(f"📥 Raw input keys: {list(input.keys())}")
    logger.info(f"📥 Input type: {type(input)}")
    
    # Log all input parameters
    for key, value in input.items():
        if key == 'audio_data' and value:
            logger.info(f"📥 {key}: [BASE64 DATA] Length: {len(value)} chars")
        elif isinstance(value, dict):
            logger.info(f"📥 {key}: {type(value)} with keys: {list(value.keys())}")
        else:
            logger.info(f"📥 {key}: {value}")
    
    # ===== FIREBASE CREDENTIAL VALIDATION =====
    _debug_gcs_creds()
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "error": "Failed to initialize Firebase storage"}
    
    # Extract callback_url early so we can send error callbacks for early failures
    meta_top = input.get('metadata', {}) if isinstance(input.get('metadata'), dict) else {}
    callback_url = input.get('callback_url') or meta_top.get('callback_url')
    user_id = input.get('user_id') or meta_top.get('user_id')
    
    # Check if VC model is available
    if vc_model is None:
        logger.error("❌ VC model not available")
        
        # Send error callback if callback_url is available
        if callback_url:
            try:
                # Send error callback
                payload = {
                    'status': 'error',
                    'user_id': user_id,
                    'voice_id': input.get('voice_id', 'unknown'),
                    'voice_name': input.get('name', 'unknown'),
                    'language': input.get('language', 'en'),
                    'error': 'VC model not available',
                }
                
                logger.info(f"📤 Error callback URL: {callback_url}")
                logger.info(f"📤 Error callback payload: {payload}")
                
                _post_signed_callback(callback_url, payload)
                logger.info("✅ Error callback sent successfully")
            except Exception as callback_error:
                logger.error(f"❌ Failed to send error callback: {callback_error}")
        
        return {"status": "error", "error": "VC model not available"}
    
    logger.info("✅ Using pre-initialized VC model")
    
    # Handle voice generation request only
    name = input.get('name')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_path = input.get('audio_path')  # Firebase Storage path e.g. audio/voices/en/recorded/uid_ts.wav
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio
    responseFormat = input.get('responseFormat', 'base64')  # Response format from frontend
    language = input.get('language', 'en')  # Language for storage organization
    is_kids_voice = input.get('is_kids_voice', False)  # Kids voice flag
    # Naming hints (optional)
    profile_filename_hint = input.get('profile_filename') or meta_top.get('profile_filename')
    sample_filename_hint = input.get('sample_filename') or meta_top.get('sample_filename')
    output_basename_hint = input.get('output_basename') or meta_top.get('output_basename')
    
    # Debug: Log callback_url immediately after extraction
    logger.info(f"🔍 EXTRACTED callback_url: {callback_url}")
    logger.info(f"🔍 EXTRACTED callback_url type: {type(callback_url)}")
    logger.info(f"🔍 EXTRACTED callback_url from input: {input.get('callback_url')}")
    logger.info(f"🔍 EXTRACTED callback_url from meta_top: {meta_top.get('callback_url')}")
    
    # ===== METADATA BREAKDOWN LOGGING =====
    logger.info("🔍 ===== METADATA BREAKDOWN =====")
    logger.info(f"📋 Top-level metadata: {meta_top}")
    logger.info(f"📋 Top-level metadata type: {type(meta_top)}")
    logger.info(f"📋 Top-level metadata keys: {list(meta_top.keys()) if isinstance(meta_top, dict) else 'Not a dict'}")
    
    # Log nested metadata if it exists
    nested_metadata = input.get('metadata', {})
    if isinstance(nested_metadata, dict):
        logger.info(f"📋 Nested metadata: {nested_metadata}")
        logger.info(f"📋 Nested metadata keys: {list(nested_metadata.keys())}")
        for key, value in nested_metadata.items():
            logger.info(f"📋   {key}: {value} (type: {type(value)})")
    else:
        logger.info(f"📋 Nested metadata: {nested_metadata} (type: {type(nested_metadata)})")
    
    logger.info("🔍 ===== END METADATA BREAKDOWN =====")

    if not name or (not audio_data and not audio_path):
        return {"status": "error", "error": "name and either audio_data or audio_path are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    logger.info(f"Language: {language}, Kids voice: {is_kids_voice}")
    
    try:
        # Honor server-provided voice_id; fallback to legacy generation only if missing
        voice_id = input.get('voice_id') or meta_top.get('voice_id')
        if not voice_id:
            voice_id = get_voice_id(name)
            logger.info(f"No voice_id provided; generated legacy id: {voice_id}")
        else:
            logger.info(f"Using provided voice_id: {voice_id}")

        # Enforce deterministic filenames: {voiceId}.npy|.mp3; recorded filename is only a local temp hint
        try:
            target_profile_name = profile_filename_hint or f"{voice_id}.npy"
            target_sample_name = sample_filename_hint or f"{voice_id}.mp3"
        except Exception:
            target_profile_name = profile_filename_hint or f"{voice_id}.npy"
            target_sample_name = sample_filename_hint or f"{voice_id}.mp3"
        
        # Prepare a local temp file from either base64 data or Firebase path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_voice_file = None
        if audio_path:
            # Infer extension from path
            lower = str(audio_path).lower()
            ext = ".wav"
            if lower.endswith(".mp3"): ext = ".mp3"
            elif lower.endswith(".ogg"): ext = ".ogg"
            elif lower.endswith(".m4a"): ext = ".m4a"
            temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}{ext}"
            try:
                if bucket is None and not initialize_firebase():
                    return {"status": "error", "error": "Failed to initialize Firebase storage"}
                blob = bucket.blob(audio_path)
                blob.download_to_filename(str(temp_voice_file))
                logger.info(f"Downloaded audio from Firebase {audio_path} to {temp_voice_file}")
            except Exception as dl_e:
                logger.error(f"❌ Failed to download audio_path {audio_path}: {dl_e}")
                return {"status": "error", "error": f"Failed to download audio_path: {dl_e}"}
        else:
            temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
            audio_bytes = base64.b64decode(audio_data)
            with open(temp_voice_file, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Saved temporary voice file to {temp_voice_file}")

        # Call the VC model's create_voice_clone method
        logger.info("🔄 Calling VC model's create_voice_clone method...")
        
        # ===== API METADATA PREPARATION LOGGING =====
        logger.info("🔍 ===== API METADATA PREPARATION =====")
        
        # Prepare API metadata including language and kids voice flag
        # Include bucket_name and country_code from top-level metadata for bucket resolution
        bucket_name = meta_top.get('bucket_name') or input.get('bucket_name')
        country_code = meta_top.get('country_code') or input.get('country_code')
        
        api_metadata = {
            'user_id': input.get('user_id'),
            'project_id': input.get('project_id'),
            'voice_type': input.get('voice_type'),
            'quality': input.get('quality'),
            'language': language,
            'is_kids_voice': is_kids_voice,
            # If request was pointer-based, pass the recorded_path to VC so it skips re-upload
            'recorded_path': audio_path if audio_path else None,
            # Explicit filenames (if provided/derived) so model uploads with the exact names
            'profile_filename': target_profile_name,
            'sample_filename': target_sample_name,
            # Bucket information for region-aware uploads
            'bucket_name': bucket_name,
            'country_code': country_code,
            # Strong metadata contract for Storage uploads downstream
            'storage_metadata': {
                'user_id': input.get('user_id') or '',
                'voice_id': voice_id,
                'voice_name': name,
                'language': language,
                'is_kids_voice': str(is_kids_voice).lower(),
                'model_type': input.get('model_type') or 'chatterbox',
                # Include bucket info in storage_metadata as well for upload functions
                'bucket_name': bucket_name,
                'country_code': country_code,
            }
        }
        
        logger.info(f"📋 API metadata prepared: {api_metadata}")
        logger.info(f"📋 API metadata type: {type(api_metadata)}")
        logger.info(f"📋 API metadata keys: {list(api_metadata.keys())}")
        
        # Log nested metadata structure
        nested_metadata = api_metadata.get('storage_metadata', {})
        logger.info(f"📋 Nested storage metadata: {nested_metadata}")
        logger.info(f"📋 Nested storage metadata type: {type(nested_metadata)}")
        logger.info(f"📋 Nested storage metadata keys: {list(nested_metadata.keys())}")
        
        # Log each metadata field with type information
        for key, value in nested_metadata.items():
            logger.info(f"📋   Storage metadata {key}: {value} (type: {type(value)})")
        
        logger.info("🔍 ===== END API METADATA PREPARATION =====")
        
        # Call the VC model - it handles everything!
        result = call_vc_model_create_voice_clone(
            audio_file_path=temp_voice_file,
            voice_id=voice_id,
            voice_name=name,
            api_metadata=api_metadata
        )
        
        # Store callback_url in result for reliable access
        if isinstance(result, dict):
            result["callback_url"] = callback_url
            logger.info(f"🔍 STORED callback_url in result: {callback_url}")
            logger.info(f"🔍 STORED callback_url type: {type(callback_url)}")
            logger.info(f"🔍 STORED callback_url in result keys: {list(result.keys())}")
        else:
            logger.warning(f"⚠️ Result is not a dict, cannot store callback_url. Result type: {type(result)}")
        
        # Clean up temporary voice file
        try:
            os.unlink(temp_voice_file)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")

        # Check if the voice clone operation failed
        if isinstance(result, dict) and result.get("status") == "error":
            logger.error(f"❌ Voice clone failed: {result.get('error', 'Unknown error')}")
            
            # Send error callback if callback_url is available
            try:
                if callback_url:
                    # Send error callback
                    payload = {
                        'status': 'error',
                        'user_id': user_id,
                        'voice_id': voice_id,
                        'voice_name': name,
                        'language': language,
                        'error': result.get('error', 'Unknown error'),
                    }
                    
                    logger.info(f"📤 Error callback URL: {callback_url}")
                    logger.info(f"📤 Error callback payload: {payload}")
                    
                    _post_signed_callback(callback_url, payload)
                    logger.info("✅ Error callback sent successfully")
            except Exception as callback_error:
                logger.error(f"❌ Failed to send error callback: {callback_error}")
            
            return result

        # Post-process result: if caller supplied audio_path, avoid duplicate recorded upload
        # and ensure recorded_audio_path reflects the original pointer
        try:
            if isinstance(result, dict) and audio_path:
                result.setdefault('metadata', {})
                result['recorded_audio_path'] = audio_path
        except Exception:
            pass
        logger.info(f"📤 Voice clone completed successfully")
        
        # No post-process renaming: model now uploads with standardized names directly

        # ===== SUCCESS CALLBACK LOGGING =====
        logger.info("🔍 ===== SUCCESS CALLBACK PAYLOAD =====")
        
        # Get callback_url from result (more reliable than variable scope)
        result_callback_url = result.get("callback_url") if isinstance(result, dict) else None
        logger.info(f"🔍 EXTRACTED callback_url from result: {result_callback_url}")
        logger.info(f"🔍 EXTRACTED callback_url from result type: {type(result_callback_url)}")
        logger.info(f"🔍 EXTRACTED callback_url from result exists: {bool(result_callback_url)}")
        if isinstance(result, dict):
            logger.info(f"🔍 EXTRACTED callback_url from result keys: {list(result.keys())}")
            logger.info(f"🔍 EXTRACTED callback_url from result has callback_url key: {'callback_url' in result}")
        
        # Attempt callback on success (only for successful operations)
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
                try:
                    # Build storage paths using the exact filenames
                    kids_segment = 'kids/' if is_kids_voice else ''
                    profile_path = f"audio/voices/{language}/{kids_segment}profiles/{target_profile_name}"
                    sample_path = f"audio/voices/{language}/{kids_segment}samples/{target_sample_name}"
                    payload = {
                        'status': 'success',
                        'user_id': user_id,
                        'voice_id': voice_id,
                        'voice_name': name,
                        'language': language,
                        'is_kids_voice': bool(is_kids_voice),
                        'model_type': input.get('model_type') or 'chatterbox',
                        'profile_path': profile_path,
                        'sample_path': sample_path,
                        'recorded_path': audio_path or (result.get('recorded_audio_path') if isinstance(result, dict) else ''),
                    }
                    
                    logger.info(f"📤 Success callback URL: {result_callback_url}")
                    logger.info(f"📤 Success callback payload: {payload}")
                    logger.info(f"📤 Success callback payload type: {type(payload)}")
                    logger.info(f"📤 Success callback payload keys: {list(payload.keys())}")
                    
                    try:
                        logger.info(f"🔍 CALLBACK SENDING:")
                        logger.info(f"🔍   URL: {result_callback_url}")
                        logger.info(f"🔍   Payload keys: {list(payload.keys())}")
                        logger.info(f"🔍   Payload size: {len(str(payload))} characters")
                        
                        _post_signed_callback(result_callback_url, payload)
                        logger.info(f"✅ VC callback POST {result_callback_url} -> signed and sent")
                    except Exception as cb_e:
                        logger.warning(f"⚠️ VC callback POST failed: {cb_e}")
                        logger.warning(f"⚠️ VC callback exception type: {type(cb_e)}")
                        logger.warning(f"⚠️ VC callback exception details: {str(cb_e)}")
                        import traceback
                        logger.warning(f"⚠️ VC callback traceback: {traceback.format_exc()}")
                except Exception as cb_e:
                    logger.warning(f"⚠️ Success callback failed: {cb_e}")
                    logger.warning(f"⚠️ Success callback exception type: {type(cb_e)}")
        except Exception as e:
            logger.warning(f"⚠️ Success callback preparation failed: {e}")
        
        logger.info("🔍 ===== END SUCCESS CALLBACK PAYLOAD =====")
        
        # Final callback status summary
        logger.info(f"🔍 CALLBACK SUMMARY:")
        logger.info(f"🔍   Original callback_url: {callback_url}")
        logger.info(f"🔍   Result callback_url: {result_callback_url}")
        logger.info(f"🔍   Callback sent: {should_send_callback if 'should_send_callback' in locals() else 'Unknown'}")
        logger.info(f"🔍   Result status: {result.get('status') if isinstance(result, dict) else 'N/A'}")

        # Opportunistic cleanup after job
        try:
            cleanup_runtime_storage(force=False)
        except Exception:
            pass
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if 'temp_voice_file' in locals():
            try:
                os.unlink(temp_voice_file)
            except:
                pass
        # ===== ERROR CALLBACK LOGGING =====
        logger.info("🔍 ===== ERROR CALLBACK PAYLOAD =====")
        
        # Attempt error callback
        try:
            if 'callback_url' in locals() and callback_url:
                try:
                    payload = {
                        'status': 'error',
                        'user_id': user_id,
                        'voice_id': input.get('voice_id') or meta_top.get('voice_id') or '',
                        'voice_name': name,
                        'language': language,
                        'error': str(e),
                    }
                    
                    logger.info(f"📤 Error callback URL: {callback_url}")
                    logger.info(f"📤 Error callback payload: {payload}")
                    logger.info(f"📤 Error callback payload type: {type(payload)}")
                    logger.info(f"📤 Error callback payload keys: {list(payload.keys())}")
                    
                    _post_signed_callback(callback_url, payload)
                    logger.info("✅ Error callback sent successfully")
                except Exception as cb_e:
                    logger.warning(f"⚠️ Error callback failed: {cb_e}")
                    logger.warning(f"⚠️ Error callback exception type: {type(cb_e)}")
        except Exception as callback_prep_e:
            logger.warning(f"⚠️ Error callback preparation failed: {callback_prep_e}")
        
        logger.info("🔍 ===== END ERROR CALLBACK PAYLOAD =====")
        try:
            cleanup_runtime_storage(force=False)
        except Exception:
            pass
        return {"status": "error", "error": str(e)}


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
        logger.warning("⚠️ DAEZEND_API_SHARED_SECRET not set; proceeding with UNSIGNED callback POST")

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

    canonical_url = _canonicalize_callback_url(callback_url)

    parsed = urlparse(canonical_url)
    # Default to voices success callback for signing if path is missing
    path_for_signing = parsed.path or '/api/voices/callback'
    ts = str(int(time.time() * 1000))
    
    logger.info(f"🔍 Parsed URL: {parsed}")
    logger.info(f"🔍 Path for signing: {path_for_signing}")
    logger.info(f"🔍 Timestamp: {ts}")
    
    body_bytes = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
    }
    if secret:
        prefix = f"POST\n{path_for_signing}\n{ts}\n".encode('utf-8')
        message = prefix + body_bytes
        sig = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
        logger.info(f"🔍 Body size: {len(body_bytes)} bytes")
        logger.info(f"🔍 Message size: {len(message)} bytes")
        logger.info(f"🔍 Signature: {sig[:20]}...")
        headers.update({
            'X-Daezend-Timestamp': ts,
            'X-Daezend-Signature': sig,
        })
    
    logger.info(f"🔍 Headers: {headers}")
    logger.info(f"🔍 Making POST request to: {canonical_url}")

    # Configure HTTP opener to follow redirects
    from urllib.request import HTTPRedirectHandler, build_opener
    
    # Create an opener that follows redirects
    redirect_handler = HTTPRedirectHandler()
    opener = build_opener(redirect_handler)
    
    req = Request(canonical_url, data=body_bytes, headers=headers, method='POST')
    
    try:
        resp = opener.open(req, timeout=15)
        response_data = resp.read()
        logger.info(f"🔍 Response status: {resp.status}")
        logger.info(f"🔍 Response headers: {dict(resp.headers)}")
        logger.info(f"🔍 Final URL after request: {getattr(resp, 'geturl', lambda: 'unknown')()}")
        logger.info(f"🔍 Response text: {response_data.decode('utf-8')[:200]}...")
        logger.info(f"✅ VC Callback POST successful: {resp.status}")
        resp.close()
    except HTTPError as http_err:
        # Explicitly handle 307/308 by re-posting to Location
        code = getattr(http_err, 'code', None)
        loc = None
        try:
            loc = http_err.headers.get('Location') if hasattr(http_err, 'headers') and http_err.headers else None
        except Exception:
            loc = None
        logger.warning(f"🔁 HTTPError encountered: code={code}, will inspect for redirect. Location={loc}")
        if code in (307, 308) and loc:
            try:
                # Build absolute URL if relative
                from urllib.parse import urljoin
                follow_url = urljoin(canonical_url, loc)
                logger.info(f"🔁 Following {code} redirect to: {follow_url}")
                # Reuse same signed headers and body (signature uses only path + body + timestamp)
                req2 = Request(follow_url, data=body_bytes, headers=headers, method='POST')
                resp2 = opener.open(req2, timeout=15)
                response_data2 = resp2.read()
                logger.info(f"🔍 Redirected response status: {resp2.status}")
                logger.info(f"🔍 Redirected response headers: {dict(resp2.headers)}")
                logger.info(f"🔍 Final URL after redirect: {getattr(resp2, 'geturl', lambda: 'unknown')()}")
                logger.info(f"🔍 Redirected response text: {response_data2.decode('utf-8')[:200]}...")
                logger.info(f"✅ VC Callback POST successful after redirect: {resp2.status}")
                resp2.close()
                return
            except Exception as follow_e:
                logger.error(f"❌ Redirect follow failed: {type(follow_e).__name__}: {follow_e}")
                raise
        else:
            logger.error(f"❌ HTTP request failed (no redirect follow): {type(http_err).__name__}: {http_err}")
            raise
    except Exception as e:
        logger.error(f"❌ HTTP request failed: {e}")
        raise

if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

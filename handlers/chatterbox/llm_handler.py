import runpod
import os
import logging
import requests
import hmac
import hashlib
import time
from urllib.parse import urlparse, urlunparse
from typing import Optional, Dict, Any
import json
from datetime import datetime
import tempfile

# Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    # Logger not yet initialized, will log later if needed

# Configure logging
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""LLM handler for RunPod runtime using Qwen2.5-32B-Instruct-AWQ model."""

# Model initialization (lazy loading)
_vllm_engine = None  # vLLM engine (primary)
_model = None  # AutoAWQ/Transformers model (fallback)
_tokenizer = None
_device = None
_use_vllm = None  # Will be determined on first load

# Firebase initialization
_firebase_initialized = False
_firestore_db = None

# Check if vLLM is available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

def _initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials."""
    global _firebase_initialized, _firestore_db
    
    if _firebase_initialized:
        return _firestore_db
    
    if not FIREBASE_AVAILABLE:
        logger.warning("⚠️ Firebase Admin SDK not available, skipping Firestore initialization")
        return None
    
    try:
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            firebase_service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT")
            
            if not firebase_service_account:
                logger.warning("⚠️ FIREBASE_SERVICE_ACCOUNT not set, skipping Firestore initialization")
                return None
            
            logger.info("🔧 Initializing Firebase Admin SDK...")
            
            # Parse JSON credentials
            try:
                cred_data = json.loads(firebase_service_account)
            except json.JSONDecodeError:
                logger.error("❌ FIREBASE_SERVICE_ACCOUNT is not valid JSON")
                return None
            
            # Create temporary file with credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(cred_data, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                # Initialize Firebase
                cred = credentials.Certificate(tmp_path)
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firebase Admin SDK initialized successfully")
            except Exception as e:
                logger.error(f"❌ Firebase initialization failed: {e}")
                return None
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        
        # Initialize Firestore
        _firestore_db = firestore.client()
        _firebase_initialized = True
        logger.info("✅ Firestore client initialized")
        return _firestore_db
        
    except Exception as e:
        logger.error(f"❌ Firebase initialization error: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return None


def _extract_beats(content: str) -> list:
    """Extract beats from story content using separators."""
    beats = []
    try:
        # Primary method: Use ⁂ separator (expected format)
        if '⁂' in content:
            beats = content.split('\n\n⁂\n\n')
            beats = [beat.strip() for beat in beats if beat.strip()]
        # Fallback: Try beat labels
        elif '### Beat' in content or 'Beat' in content:
            import re
            # Match patterns like "### Beat 1:", "Beat 1:", etc.
            beat_pattern = r'(?:###\s*)?Beat\s*\d+\s*:'
            parts = re.split(beat_pattern, content, flags=re.IGNORECASE)
            beats = [beat.strip() for beat in parts if beat.strip()]
        else:
            # Last resort: Try to split intelligently if no separators found
            # This should rarely happen if prompt is correct
            logger.warning("⚠️ No beat separators found, attempting intelligent split")
            # Try splitting by double newlines as fallback
            potential_beats = content.split('\n\n')
            # Group into ~10 beats (for Qwen3) or 12 beats (for others)
            # Default to 10 for safety
            target_beat_count = 10
            target_beat_length = len(content) // target_beat_count
            current_beat = []
            current_length = 0
            for para in potential_beats:
                para = para.strip()
                if not para:
                    continue
                current_beat.append(para)
                current_length += len(para)
                if current_length >= target_beat_length and len(beats) < (target_beat_count - 1):
                    beats.append('\n\n'.join(current_beat))
                    current_beat = []
                    current_length = 0
            if current_beat:
                beats.append('\n\n'.join(current_beat))
        
        # Filter out empty beats and very short ones (likely not actual beats)
        beats = [beat for beat in beats if len(beat) > 50]
        
        # Note: Expected beat count depends on genre (10 for Qwen3, 12 for others)
        # This will be validated in the calling code
        
        return beats
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract beats: {e}")
        return []

def _log_story_generation_details(story_id: str, title: str, content: str, preview: str, 
                                  coverPrompt: str, coverHook: str, coverEssence: str, 
                                  genre: str, age_range: str):
    """Log comprehensive details about the generated story."""
    try:
        # Extract beats
        beats = _extract_beats(content)
        
        # Calculate content statistics
        word_count = len(content.split())
        char_count = len(content)
        preview_length = len(preview) if preview else 0
        
        # Log comprehensive story details
        logger.info("=" * 80)
        logger.info(f"📖 STORY GENERATION COMPLETE - Story ID: {story_id}")
        logger.info("=" * 80)
        
        # Title
        logger.info(f"📌 TITLE: {title}")
        logger.info(f"   Length: {len(title)} characters")
        
        # Content statistics
        logger.info(f"📝 CONTENT STATISTICS:")
        logger.info(f"   Total characters: {char_count:,}")
        logger.info(f"   Total words: {word_count:,}")
        logger.info(f"   Estimated reading time: ~{word_count // 200} minutes")
        
        # Content preview
        content_preview = content[:500] + "..." if len(content) > 500 else content
        logger.info(f"📄 CONTENT PREVIEW (first 500 chars):")
        logger.info(f"   {content_preview}")
        
        # Preview text
        if preview:
            logger.info(f"📋 PREVIEW TEXT:")
            logger.info(f"   {preview}")
            logger.info(f"   Length: {preview_length} characters")
        else:
            logger.warning(f"⚠️ Preview text is empty")
        
        # Beats
        logger.info(f"🎬 BEATS EXTRACTED: {len(beats)} beats found")
        if beats:
            for i, beat in enumerate(beats[:10], 1):  # Log first 10 beats
                beat_preview = beat[:200] + "..." if len(beat) > 200 else beat
                logger.info(f"   Beat {i}: {len(beat)} chars - {beat_preview}")
            if len(beats) > 10:
                logger.info(f"   ... and {len(beats) - 10} more beats")
            # Note: Expected beat count depends on genre (10 for Qwen3, 12 for others)
            if len(beats) not in [10, 12]:
                logger.warning(f"⚠️ Unexpected beat count: {len(beats)} (expected 10 for Qwen3 or 12 for others)")
        else:
            logger.warning(f"⚠️ No beats extracted from content")
        
        # Cover metadata
        if coverHook:
            logger.info(f"🎨 COVER HOOK: {coverHook}")
        if coverEssence:
            logger.info(f"💫 COVER ESSENCE: {coverEssence}")
        if coverPrompt:
            cover_preview = coverPrompt[:200] + "..." if len(coverPrompt) > 200 else coverPrompt
            logger.info(f"🖼️ COVER PROMPT ({len(coverPrompt)} chars):")
            logger.info(f"   {cover_preview}")
        else:
            logger.warning(f"⚠️ Cover prompt is empty")
        
        # Metadata
        logger.info(f"🏷️ METADATA:")
        logger.info(f"   Genre: {genre}")
        logger.info(f"   Age Range: {age_range}")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log story generation details: {e}")
        import traceback
        logger.warning(f"⚠️ Traceback: {traceback.format_exc()}")

def _save_story_to_firestore(story_id: str, user_id: str, content: str, metadata: Dict[str, Any]):
    """Save story directly to Firestore. Detects if it's a default story and saves to the correct collection."""
    try:
        db = _initialize_firebase()
        if not db:
            logger.warning("⚠️ Firestore not available, skipping story save")
            return False
        
        # Extract metadata
        language = metadata.get("language", "en")
        genre = metadata.get("genre", "")
        age_range = metadata.get("age_range", "")
        
        # Qwen3 is adult-oriented, not age-specific - use +18 as default
        if genre and genre.lower() in ['qwen3', 'qwen']:
            age_range = "+18"
        
        # Title, preview, and cover will be generated by OpenAI via callback endpoint
        # Set placeholder values - OpenAI will update these via callback
        logger.info(f"📝 Story generated. Title/preview/cover will be generated by OpenAI via callback.")
        title = "Untitled Story"
        preview = ""
        coverPrompt = ""
        coverHook = ""
        coverEssence = ""
        
        # Add flag indicating OpenAI post-processing is needed (only for Qwen3)
        needs_openai_processing = genre and genre.lower() in ['qwen3', 'qwen']
        
        # Check if this is a default story by checking defaultStories collection
        is_default_story = False
        try:
            default_story_ref = db.collection("defaultStories").document(story_id)
            default_snap = default_story_ref.get()
            is_default_story = default_snap.exists
        except Exception as e:
            logger.warning(f"⚠️ Could not check defaultStories collection: {e}")
        
        # Prepare story document (matching OpenAI story structure)
        story_data = {
            "title": title,
            "content": content,
            "preview": preview,
            "coverPrompt": coverPrompt,
            "coverHook": coverHook,
            "coverEssence": coverEssence,
            "ageRange": age_range,
            "language": language,
            "promptVersion": "12-beat@2025-01-09",
            "genre": [genre.lower()] if genre else [],
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "provider": "runpod",  # Mark as Runpod-generated
            "needsOpenAIProcessing": needs_openai_processing,  # Flag for OpenAI callback
        }
        
        # For default stories, use generationStatus instead of status
        # Set to "processing" initially - callback will update to "generated" after OpenAI processing
        if is_default_story:
            story_data["generationStatus"] = "processing"
        else:
            story_data["status"] = "ready"
        
        # Add user_id if available
        if user_id:
            story_data["userId"] = user_id
        
        # Save to the correct collection
        if is_default_story:
            story_ref = db.collection("defaultStories").document(story_id)
            logger.info(f"💾 Saving default story {story_id} to defaultStories collection")
        else:
            story_ref = db.collection("stories").document(story_id)
            logger.info(f"💾 Saving user story {story_id} to stories collection")
        
        story_ref.set(story_data, merge=True)
        
        logger.info(f"✅ Story {story_id} saved to Firestore ({'defaultStories' if is_default_story else 'stories'} collection)")
        logger.info(f"📊 Story fields saved: title={title[:50]}..., preview={len(preview)} chars, coverPrompt={len(coverPrompt)} chars")
        
        # Log comprehensive story generation details
        _log_story_generation_details(
            story_id=story_id,
            title=title,
            content=content,
            preview=preview,
            coverPrompt=coverPrompt,
            coverHook=coverHook,
            coverEssence=coverEssence,
            genre=genre,
            age_range=age_range
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save story to Firestore: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def _ensure_cuda_ready(max_retries=5, delay=2):
    """Ensure CUDA is ready before loading model. Retry if device is busy."""
    import torch
    if not torch.cuda.is_available():
        return False
    
    for attempt in range(max_retries):
        try:
            # Try to create a small tensor to check if CUDA is ready
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.synchronize()
            logger.info(f"✅ CUDA is ready (attempt {attempt + 1})")
            return True
        except RuntimeError as e:
            if "busy" in str(e).lower() or "unavailable" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ CUDA device busy, waiting {delay}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ CUDA device still busy after {max_retries} attempts")
                    raise
            else:
                raise
    return False

def _load_model():
    """Lazy load the model from network volume or HuggingFace using vLLM (primary method for AWQ models)."""
    global _vllm_engine, _model, _tokenizer, _device, _use_vllm
    
    # Return already loaded model
    if _vllm_engine is not None or _model is not None:
        if _use_vllm:
            return _vllm_engine, _tokenizer, _device
        else:
            return _model, _tokenizer, _device
    
    try:
        from transformers import AutoTokenizer
        import torch
        from pathlib import Path
        
        # Check for network volume path first
        model_path = os.getenv("MODEL_PATH", "/runpod-volume/models/Qwen2.5-32B-Instruct-AWQ")
        model_name = os.getenv("MODEL_NAME", "Qwen2.5-32B-Instruct-AWQ")  # Fallback model name
        
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure CUDA is ready before attempting to load model
        if _device == "cuda":
            _ensure_cuda_ready()
        
        # Check if network volume path exists
        model_path_obj = Path(model_path)
        use_local_path = model_path_obj.exists() and model_path_obj.is_dir()
        
        if use_local_path:
            logger.info(f"📦 Loading model from network volume: {model_path}")
            logger.info(f"🔍 Checking model directory contents...")
            try:
                model_files = list(model_path_obj.glob("*"))
                logger.info(f"📁 Found {len(model_files)} files/directories in model path")
                for f in model_files[:10]:  # Log first 10 files
                    logger.info(f"   - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)" if f.is_file() else f"   - {f.name}/ (dir)")
            except Exception as e:
                logger.warning(f"⚠️ Could not list model directory: {e}")
            
            # Check if this is an AWQ quantized model
            model_files = list(model_path_obj.iterdir())
            is_awq = any(
                'awq' in f.name.lower() or 
                f.name == 'quant_config.json' or 
                (f.is_dir() and 'awq' in f.name.lower())
                for f in model_files
            )
            
            # For AWQ models, vLLM is the only supported method (AutoAWQ is deprecated)
            if is_awq:
                if not VLLM_AVAILABLE:
                    raise RuntimeError(
                        "AWQ model requires vLLM, but vLLM is not available. "
                        "Please ensure vLLM is installed in the container."
                    )
                
                # Retry vLLM loading with exponential backoff for CUDA busy errors
                max_retries = 3
                base_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"🚀 Attempting to load AWQ model with vLLM (attempt {attempt + 1}/{max_retries})...")
                        logger.info(f"📦 Loading AWQ model from {model_path} using vLLM...")
                        
                        # Use awq_marlin quantization (as detected by vLLM automatically)
                        # vLLM will automatically detect and use awq_marlin for compatible models
                        _vllm_engine = LLM(
                            model=model_path,
                            quantization="awq",  # vLLM will auto-detect awq_marlin if compatible
                            trust_remote_code=True,
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.85,  # Slightly lower to avoid OOM
                            max_model_len=32768,  # Match Qwen2.5 context length
                        )
                        
                        # Load tokenizer separately for vLLM
                        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                        _use_vllm = True
                        logger.info("✅ AWQ model loaded successfully with vLLM")
                        return _vllm_engine, _tokenizer, _device
                        
                    except RuntimeError as vllm_error:
                        error_str = str(vllm_error).lower()
                        if ("busy" in error_str or "unavailable" in error_str) and attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)
                            logger.warning(f"⚠️ CUDA device busy, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            # Ensure CUDA is ready before retry
                            if _device == "cuda":
                                _ensure_cuda_ready()
                            continue
                        else:
                            logger.error(f"❌ vLLM loading failed: {vllm_error}")
                            if attempt == max_retries - 1:
                                raise RuntimeError(
                                    f"Failed to load AWQ model with vLLM after {max_retries} attempts. "
                                    f"Last error: {vllm_error}. "
                                    f"AutoAWQ fallback is not available (deprecated). "
                                    f"Please ensure CUDA is available and the model path is correct."
                                ) from vllm_error
                    except Exception as vllm_error:
                        logger.error(f"❌ vLLM loading failed: {vllm_error}")
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Failed to load AWQ model with vLLM after {max_retries} attempts. "
                                f"Last error: {vllm_error}"
                            ) from vllm_error
                        wait_time = base_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
            else:
                # Standard model loading (non-AWQ)
                logger.info("🔧 Loading standard (non-AWQ) model...")
                _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # Try vLLM for standard models too
                if VLLM_AVAILABLE:
                    max_retries = 3
                    base_delay = 5
                    
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"🚀 Attempting to load standard model with vLLM (attempt {attempt + 1}/{max_retries})...")
                            _vllm_engine = LLM(
                                model=model_path,
                                trust_remote_code=True,
                                tensor_parallel_size=1,
                                gpu_memory_utilization=0.85,
                            )
                            _use_vllm = True
                            logger.info("✅ Standard model loaded successfully with vLLM")
                            return _vllm_engine, _tokenizer, _device
                        except RuntimeError as e:
                            error_str = str(e).lower()
                            if ("busy" in error_str or "unavailable" in error_str) and attempt < max_retries - 1:
                                wait_time = base_delay * (2 ** attempt)
                                logger.warning(f"⚠️ CUDA device busy, waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                                if _device == "cuda":
                                    _ensure_cuda_ready()
                                continue
                            else:
                                logger.warning(f"⚠️ vLLM loading failed: {e}, falling back to transformers...")
                                break
                        except Exception as e:
                            logger.warning(f"⚠️ vLLM loading failed: {e}, falling back to transformers...")
                            if attempt < max_retries - 1:
                                wait_time = base_delay * (2 ** attempt)
                                time.sleep(wait_time)
                            else:
                                break
                
                # Fallback to transformers for non-AWQ models
                from transformers import AutoModelForCausalLM
                logger.info("🔧 Loading standard model with transformers...")
                _model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                    device_map="auto" if _device == "cuda" else None,
                    trust_remote_code=True,
                )
                _use_vllm = False
                logger.info("✅ Standard model loaded with transformers")
                return _model, _tokenizer, _device
            
            if _device == "cpu" and not hasattr(_model, 'device_map'):
                _model = _model.to(_device)
        else:
            # Fallback to HuggingFace if local path doesn't exist
            logger.info(f"📦 Loading model from HuggingFace: {model_name}")
            logger.warning(f"⚠️ Network volume path {model_path} not found, using HuggingFace fallback")
            
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # For HuggingFace models, try vLLM first (especially for AWQ models)
            if VLLM_AVAILABLE:
                is_awq_hf = "awq" in model_name.lower()
                max_retries = 3
                base_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"🚀 Attempting to load HuggingFace model with vLLM (attempt {attempt + 1}/{max_retries})...")
                        _vllm_engine = LLM(
                            model=model_name,
                            quantization="awq" if is_awq_hf else None,
                            trust_remote_code=True,
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.85,
                            max_model_len=32768 if is_awq_hf else None,
                        )
                        _use_vllm = True
                        logger.info("✅ HuggingFace model loaded successfully with vLLM")
                        return _vllm_engine, _tokenizer, _device
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if ("busy" in error_str or "unavailable" in error_str) and attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)
                            logger.warning(f"⚠️ CUDA device busy, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            if _device == "cuda":
                                _ensure_cuda_ready()
                            continue
                        else:
                            if attempt == max_retries - 1:
                                if is_awq_hf:
                                    raise RuntimeError(
                                        f"Failed to load AWQ model from HuggingFace with vLLM after {max_retries} attempts. "
                                        f"Last error: {e}. AutoAWQ fallback is not available (deprecated)."
                                    ) from e
                            logger.warning(f"⚠️ vLLM loading failed: {e}")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ vLLM loading failed: {e}")
                        if attempt == max_retries - 1 and is_awq_hf:
                            raise RuntimeError(
                                f"Failed to load AWQ model from HuggingFace with vLLM. "
                                f"Error: {e}. AutoAWQ fallback is not available (deprecated)."
                            ) from e
                        if attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)
                            time.sleep(wait_time)
                        else:
                            break
            
            # Fallback to transformers only for non-AWQ models
            if "awq" in model_name.lower():
                raise RuntimeError(
                    "AWQ models require vLLM. vLLM loading failed and AutoAWQ fallback is deprecated. "
                    "Please ensure vLLM is properly installed and CUDA is available."
                )
            
            # Only load non-AWQ models with transformers
            from transformers import AutoModelForCausalLM
            logger.info("🔧 Loading non-AWQ model with transformers...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                device_map="auto" if _device == "cuda" else None,
                trust_remote_code=True,
            )
            
            if _device == "cpu":
                _model = _model.to(_device)
            
            _use_vllm = False
        
        logger.info(f"✅ Model loaded successfully on {_device}")
        if _use_vllm:
            logger.info(f"📊 Using vLLM engine")
        else:
            logger.info(f"📊 Model type: {type(_model).__name__}")
        logger.info(f"📊 Tokenizer type: {type(_tokenizer).__name__}")
        
        if _use_vllm:
            return _vllm_engine, _tokenizer, _device
        else:
            return _model, _tokenizer, _device
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise

def _post_signed_callback(callback_url: str, payload: dict):
    """POST JSON payload to callback_url with HMAC headers compatible with app callback."""
    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
    if not secret:
        logger.error("❌ DAEZEND_API_SHARED_SECRET not set; cannot sign callback")
        raise RuntimeError('DAEZEND_API_SHARED_SECRET not set; cannot sign callback')
    
    def _canonicalize_callback_url(url: str) -> str:
        try:
            p = urlparse(url)
            scheme = p.scheme or 'https'
            netloc = p.netloc
            if netloc == 'daezend.app':
                netloc = 'www.daezend.app'
            return urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
        except Exception:
            return url
    
    canonical_url = _canonicalize_callback_url(callback_url)
    parsed = urlparse(canonical_url)
    path_for_signing = parsed.path or '/api/llm/callback'
    ts = str(int(time.time() * 1000))
    
    body_string = json.dumps(payload)
    body_buffer = body_string.encode('utf-8')
    
    # Create HMAC signature
    prefix = f"POST\n{path_for_signing}\n{ts}\n".encode('utf-8')
    message = prefix + body_buffer
    signature = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
    
    headers = {
        'Content-Type': 'application/json',
        'X-Daezend-Timestamp': ts,
        'X-Daezend-Signature': signature,
    }
    
    logger.info(f"📤 Sending callback to: {canonical_url}")
    response = requests.post(canonical_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    logger.info(f"✅ Callback sent successfully: {response.status_code}")

def _post_signed_callback_with_retry(callback_url: str, payload: dict, max_retries: int = 4):
    """POST callback with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            _post_signed_callback(callback_url, payload)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            logger.warning(f"⚠️ Callback attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
            time.sleep(wait_time)

def notify_success_callback(callback_url: str, story_id: str, content: str, **kwargs):
    """Send success callback to the main app when LLM generation succeeds."""
    payload = {
        "story_id": story_id,
        "content": content,
        "user_id": kwargs.get("user_id"),
        "metadata": kwargs.get("metadata", {})
    }
    
    try:
        logger.info(f"📤 Sending success callback to: {callback_url}")
        _post_signed_callback_with_retry(callback_url, payload)
        logger.info(f"✅ Success callback sent successfully for story {story_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send success callback: {e}")
        return False

def notify_error_callback(error_callback_url: str, story_id: str, error_message: str, **kwargs):
    """Send error callback to the main app when LLM generation fails."""
    payload = {
        "story_id": story_id,
        "error": error_message,
        "error_details": kwargs.get("error_details"),
        "user_id": kwargs.get("user_id"),
        "job_id": kwargs.get("job_id"),
        "metadata": kwargs.get("metadata", {})
    }
    
    try:
        logger.info(f"📤 Sending error callback to: {error_callback_url}")
        _post_signed_callback_with_retry(error_callback_url, payload)
        logger.info(f"✅ Error callback sent successfully for story {story_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send error callback: {e}")
        return False

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for LLM story generation.
    
    Expected input format:
    {
        "input": {
            "messages": [{"role": "user", "content": "..."}, ...],
            "temperature": 0.7,
            "max_tokens": 6000,
            "language": "English",
            "genre": "adventure",
            "age_range": "6-8",
            "user_id": "...",
            "story_id": "...",
            "metadata": {...}
        },
        "metadata": {
            "callback_url": "...",
            "user_id": "...",
            "story_id": "..."
        }
    }
    """
    try:
        input_data = event.get("input", {})
        metadata = event.get("metadata", {})
        
        # Extract parameters
        messages = input_data.get("messages", [])
        temperature = input_data.get("temperature", 0.7)
        language = input_data.get("language")
        genre = input_data.get("genre")
        age_range = input_data.get("age_range")
        
        # Default to 3200 for Qwen3, 4000 for others
        default_max_tokens = 6000 if genre and genre.lower() in ['qwen3', 'qwen'] else 6000
        max_tokens = input_data.get("max_tokens", default_max_tokens)
        
        # Qwen3 is adult-oriented, not age-specific - use +18 as default
        if genre and genre.lower() in ['qwen3', 'qwen']:
            age_range = "+18"
        
        user_id = input_data.get("user_id") or metadata.get("user_id")
        story_id = input_data.get("story_id") or metadata.get("story_id")
        callback_url = input_data.get("callback_url") or metadata.get("callback_url")
        
        logger.info("=" * 80)
        logger.info(f"📖 LLM GENERATION REQUEST RECEIVED")
        logger.info("=" * 80)
        logger.info(f"📊 Story ID: {story_id}")
        logger.info(f"👤 User ID: {user_id}")
        logger.info(f"📝 Messages count: {len(messages)}")
        logger.info(f"🌡️ Temperature: {temperature}")
        logger.info(f"🔢 Max tokens: {max_tokens}")
        logger.info(f"🏷️ Genre: {genre}")
        logger.info(f"👶 Age Range: {age_range}")
        logger.info(f"🌍 Language: {language}")
        
        # Log message preview
        if messages:
            logger.info(f"📝 Message preview:")
            for i, msg in enumerate(messages[:2], 1):  # Log first 2 messages
                content_preview = msg.get("content", "")[:200] + "..." if len(msg.get("content", "")) > 200 else msg.get("content", "")
                logger.info(f"   Message {i} ({msg.get('role', 'unknown')}): {content_preview}")
            if len(messages) > 2:
                logger.info(f"   ... and {len(messages) - 2} more messages")
        
        if not messages:
            raise ValueError("messages is required")
        
        # Load model if not already loaded
        model_or_engine, tokenizer, device = _load_model()
        use_vllm = _use_vllm
        
        # Format messages for Qwen model
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Generate using vLLM or transformers/AutoAWQ
        if use_vllm:
            # Use vLLM API
            logger.info("🚀 Generating story content with vLLM...")
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            
            # Generate
            logger.info(f"⏳ Starting story generation (vLLM)...")
            start_time = time.time()
            outputs = model_or_engine.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            generation_time = time.time() - start_time
            
            # Extract beats for logging
            beats = _extract_beats(generated_text)
            
            logger.info(f"✅ Generated content length: {len(generated_text)} characters (vLLM)")
            logger.info(f"⏱️ Generation time: {generation_time:.2f} seconds")
            logger.info(f"📊 Content statistics: {len(generated_text.split())} words, {len(generated_text)} chars")
            logger.info(f"🎬 Beats detected: {len(beats)} beats")
            
            # Log content preview
            content_preview = generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
            logger.info(f"📄 Generated content preview (first 300 chars):")
            logger.info(f"   {content_preview}")
            
            # Log beats preview
            if beats:
                logger.info(f"🎬 Beats preview:")
                for i, beat in enumerate(beats[:3], 1):  # Log first 3 beats
                    beat_preview = beat[:150] + "..." if len(beat) > 150 else beat
                    logger.info(f"   Beat {i} ({len(beat)} chars): {beat_preview}")
                if len(beats) > 3:
                    logger.info(f"   ... and {len(beats) - 3} more beats")
        else:
            # Use transformers/AutoAWQ API
            logger.info("🤖 Generating story content with transformers/AutoAWQ...")
            import torch
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = tokenizer([text], return_tensors="pt")
            # Move to device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model_or_engine.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                )
            
            # Decode
            generated_text = tokenizer.batch_decode(
                generated_ids[:, model_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # Extract beats for logging
            beats = _extract_beats(generated_text)
            
            logger.info(f"✅ Generated content length: {len(generated_text)} characters (transformers/AutoAWQ)")
            logger.info(f"📊 Content statistics: {len(generated_text.split())} words, {len(generated_text)} chars")
            logger.info(f"🎬 Beats detected: {len(beats)} beats")
            
            # Log content preview
            content_preview = generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
            logger.info(f"📄 Generated content preview (first 300 chars):")
            logger.info(f"   {content_preview}")
            
            # Log beats preview
            if beats:
                logger.info(f"🎬 Beats preview:")
                for i, beat in enumerate(beats[:3], 1):  # Log first 3 beats
                    beat_preview = beat[:150] + "..." if len(beat) > 150 else beat
                    logger.info(f"   Beat {i} ({len(beat)} chars): {beat_preview}")
                if len(beats) > 3:
                    logger.info(f"   ... and {len(beats) - 3} more beats")
        
        # Enforce character limit (8,500 max for Qwen3)
        max_chars = 8500 if genre and genre.lower() in ['qwen3', 'qwen'] else 12000
        if len(generated_text) > max_chars:
            logger.warning(f"⚠️ Story exceeds {max_chars} character limit ({len(generated_text)} chars), truncating...")
            # Try to truncate at a beat boundary
            beats = _extract_beats(generated_text)
            expected_beats = 10 if genre and genre.lower() in ['qwen3', 'qwen'] else 12
            if len(beats) >= expected_beats:
                # Truncate by removing content from last beat(s) if needed
                truncated_beats = beats[:expected_beats]
                # Rebuild content, ensuring we don't exceed max_chars
                rebuilt_content = '\n\n⁂\n\n'.join(truncated_beats)
                if len(rebuilt_content) > max_chars:
                    # Further truncate last beat
                    last_beat = truncated_beats[-1]
                    remaining_chars = max_chars - len('\n\n⁂\n\n'.join(truncated_beats[:-1])) - len('\n\n⁂\n\n')
                    truncated_beats[-1] = last_beat[:remaining_chars].rsplit('.', 1)[0] + '.' if '.' in last_beat[:remaining_chars] else last_beat[:remaining_chars]
                    rebuilt_content = '\n\n⁂\n\n'.join(truncated_beats)
                generated_text = rebuilt_content
                logger.info(f"✅ Truncated to {len(generated_text)} characters while maintaining beat structure")
            else:
                # Fallback: simple truncation
                generated_text = generated_text[:max_chars].rsplit('.', 1)[0] + '.' if '.' in generated_text[:max_chars] else generated_text[:max_chars]
                logger.warning(f"⚠️ Simple truncation applied: {len(generated_text)} characters")
        
        # Save story directly to Firestore
        if story_id and user_id:
            logger.info(f"💾 Saving story {story_id} to Firestore...")
            save_metadata = {
                "language": language,
                "genre": genre,
                "age_range": age_range,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            _save_story_to_firestore(story_id, user_id, generated_text, save_metadata)
        
        # Send callback if callback_url is provided
        if callback_url:
            try:
                # Construct error callback URL
                if "/api/llm/callback" in callback_url:
                    error_callback_url = callback_url.replace("/api/llm/callback", "/api/llm/error-callback")
                else:
                    base_url = callback_url.rstrip("/")
                    error_callback_url = f"{base_url}/error-callback"
                
                # Send success callback
                callback_sent = notify_success_callback(
                    callback_url=callback_url,
                    story_id=story_id or "unknown",
                    content=generated_text,
                    user_id=user_id,
                    metadata={
                        "language": language,
                        "genre": genre,
                        "age_range": age_range,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                
                if not callback_sent:
                    # If callback fails, try error callback
                    notify_error_callback(
                        error_callback_url=error_callback_url,
                        story_id=story_id or "unknown",
                        error_message="Failed to send success callback",
                        user_id=user_id,
                        metadata={"callback_failed": True}
                    )
            except Exception as callback_error:
                logger.error(f"❌ Callback error: {callback_error}")
                # Try to send error callback
                try:
                    if "/api/llm/callback" in callback_url:
                        error_callback_url = callback_url.replace("/api/llm/callback", "/api/llm/error-callback")
                    else:
                        base_url = callback_url.rstrip("/")
                        error_callback_url = f"{base_url}/error-callback"
                    
                    notify_error_callback(
                        error_callback_url=error_callback_url,
                        story_id=story_id or "unknown",
                        error_message=f"Callback failed: {str(callback_error)}",
                        user_id=user_id,
                        metadata={"callback_error": str(callback_error)}
                    )
                except Exception:
                    pass
        
        return {
            "status": "success",
            "content": generated_text,
            "metadata": {
                "language": language,
                "genre": genre,
                "age_range": age_range,
            }
        }
        
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Send error callback if callback_url is available
        try:
            input_data = event.get("input", {})
            metadata = event.get("metadata", {})
            callback_url = input_data.get("callback_url") or metadata.get("callback_url")
            story_id = input_data.get("story_id") or metadata.get("story_id")
            user_id = input_data.get("user_id") or metadata.get("user_id")
            
            if callback_url:
                if "/api/llm/callback" in callback_url:
                    error_callback_url = callback_url.replace("/api/llm/callback", "/api/llm/error-callback")
                else:
                    base_url = callback_url.rstrip("/")
                    error_callback_url = f"{base_url}/error-callback"
                
                notify_error_callback(
                    error_callback_url=error_callback_url,
                    story_id=story_id or "unknown",
                    error_message=str(e),
                    error_details=f"LLM handler error: {type(e).__name__}",
                    user_id=user_id,
                    job_id=event.get("id"),
                    metadata={"error_type": type(e).__name__}
                )
        except Exception as callback_error:
            logger.error(f"❌ Failed to send error callback: {callback_error}")
        
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == '__main__':
    logger.info("🚀 LLM Handler starting...")
    logger.info("✅ LLM Handler ready")
    runpod.serverless.start({"handler": handler})


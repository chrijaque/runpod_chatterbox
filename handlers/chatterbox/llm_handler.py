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
        
        # Qwen3 and Erotic are adult-oriented, not age-specific - use +18 as default
        if genre and genre.lower() in ['qwen3', 'qwen', 'erotic']:
            age_range = "+18"
        
        # Get generated metadata if available (from Step 5)
        title = metadata.get("generated_title", "Untitled Story")
        preview = metadata.get("generated_preview", "")
        coverPrompt = metadata.get("generated_coverPrompt", "")
        tags = metadata.get("generated_tags")
        coverHook = ""
        coverEssence = ""
        
        # No longer need OpenAI processing - everything is generated in Qwen
        needs_openai_processing = False
        
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
            "needsOpenAIProcessing": False,  # No longer needed - everything generated in Qwen
        }
        
        # Add tags if generated
        if tags:
            story_data["tags"] = tags
        
        # For default stories, use generationStatus instead of status
        # Set to "generated" since metadata is now generated in Qwen
        if is_default_story:
            story_data["generationStatus"] = "generated"
            story_data["approvalStatus"] = "pending"
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
        
        # Treat anything that looks like AWQ as AWQ, even if the quant config file naming differs.
        # vLLM can often detect awq_marlin compatibility even when the folder doesn't contain 'awq' in filenames.
        awq_hint = ("awq" in str(model_path).lower()) or ("awq" in str(model_name).lower())

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
            is_awq = awq_hint or any(
                'awq' in f.name.lower() or 
                f.name == 'quant_config.json' or 
                f.name == 'quantize_config.json' or
                f.name == 'quantization_config.json' or
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
                if _device != "cuda":
                    raise RuntimeError(
                        "AWQ model requires a CUDA GPU. "
                        "This worker appears to be CPU-only (torch.cuda.is_available() == False). "
                        "Fix: run this endpoint on a GPU instance / ensure the container has NVIDIA runtime."
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
                        # If CUDA/driver isn't available, do NOT fall back to transformers for AWQ.
                        if "cuda driver initialization failed" in error_str or "you might not have a cuda gpu" in error_str:
                            raise RuntimeError(
                                "CUDA driver initialization failed while loading an AWQ model with vLLM. "
                                "This usually means the RunPod worker has no usable GPU or the NVIDIA driver/runtime isn't available. "
                                "Fix: ensure the endpoint is configured with a GPU and the container uses the NVIDIA runtime."
                            ) from vllm_error
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
                
                # If this is actually an AWQ model (based on name/path hint), do NOT fall back to transformers.
                # Transformers+AWQ integration is brittle and may fail depending on package versions.
                if awq_hint:
                    raise RuntimeError(
                        "Model appears to be AWQ (AWQ hint in model path/name), but AWQ config wasn't detected in files. "
                        "vLLM loading failed and transformers fallback is disabled for AWQ. "
                        "Fix: ensure GPU is available and use vLLM for AWQ models."
                    )

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
    secret = os.getenv('MINSTRALY_API_SHARED_SECRET')
    if not secret:
        logger.error("❌ MINSTRALY_API_SHARED_SECRET not set; cannot sign callback")
        raise RuntimeError('MINSTRALY_API_SHARED_SECRET not set; cannot sign callback')
    
    def _canonicalize_callback_url(url: str) -> str:
        try:
            p = urlparse(url)
            scheme = p.scheme or 'https'
            netloc = p.netloc
            if netloc == 'minstraly.com':
                netloc = 'www.minstraly.com'
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
        'X-Minstraly-Timestamp': ts,
        'X-Minstraly-Signature': signature,
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

def _generate_content(
    messages: list,
    model_or_engine,
    tokenizer,
    use_vllm: bool,
    temperature: float,
    max_tokens: int,
    device: str
) -> tuple[str, float]:
    """
    Helper function to generate content from messages.
    Returns (generated_text, generation_time)
    """
    import torch
    
    # Format messages for Qwen model
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    if use_vllm:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Log the full formatted prompt that's sent to the model
        logger.info("=" * 80)
        logger.info("🤖 FULL FORMATTED PROMPT (after chat template):")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        
        # Generate
        start_time = time.time()
        outputs = model_or_engine.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        generation_time = time.time() - start_time
        
        return generated_text, generation_time
    else:
        # Use transformers API
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Log the full formatted prompt that's sent to the model
        logger.info("=" * 80)
        logger.info("🤖 FULL FORMATTED PROMPT (after chat template):")
        logger.info("=" * 80)
        logger.info(text)
        logger.info("=" * 80)
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt")
        # Move to device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # Generate
        start_time = time.time()
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
        generation_time = time.time() - start_time
        
        return generated_text, generation_time

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

_BEAT_SEPARATOR = "\n\n⁂\n\n"
_SOFT_MIN_CHARS = 9000
_HARD_MAX_CHARS = 11500
_STEP2_GATE_MIN = 8800
_STEP2_GATE_MAX = 11800
_BEAT_MIN = 780
_BEAT_TARGET = 850
_BEAT_MAX = 930

def _normalize_workflow_type(input_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Return normalized workflow type (accepts workflow_type or workflow_version)."""
    wt = (
        input_data.get("workflow_type")
        or metadata.get("workflow_type")
        or input_data.get("workflow_version")
        or metadata.get("workflow_version")
        or ""
    )
    try:
        return str(wt).strip().lower()
    except Exception:
        return ""

def _split_beats_strict(text: str) -> list:
    """Split a structured beat-labeled story into beats using the canonical separator."""
    parts = text.split(_BEAT_SEPARATOR) if _BEAT_SEPARATOR in text else [text]
    beats = [p.strip() for p in parts if p and p.strip()]
    return beats

def _validate_beats_strict(beats: list, expected_beats: int = 12) -> tuple[bool, str]:
    """Validate strict beat format: count, labels, and separator safety."""
    try:
        if len(beats) != expected_beats:
            return False, f"expected {expected_beats} beats, got {len(beats)}"
        for i in range(1, expected_beats + 1):
            b = beats[i - 1].lstrip()
            if not b.startswith(f"Beat {i}:"):
                return False, f"beat {i} missing required label prefix"
            if "⁂" in beats[i - 1]:
                return False, f"beat {i} contains separator glyph"
        return True, ""
    except Exception as e:
        return False, f"validation error: {e}"

def _ensure_structured_beats(
    text: str,
    expected_beats: int,
    *,
    model_or_engine,
    tokenizer,
    use_vllm: bool,
    device: str,
    step_name: str,
) -> str:
    """
    Ensure text is in strict Beat 1..N + separator format.
    If invalid, attempt a single 'format-only' repair pass.
    """
    beats = _split_beats_strict(text)
    ok, reason = _validate_beats_strict(beats, expected_beats=expected_beats)
    if ok:
        return text.strip()

    logger.warning(f"⚠️ {step_name}: invalid structured beats ({reason}), attempting format-only repair")

    format_system = f"""You are a strict formatter.

TASK:
Reformat the provided text into a strict {expected_beats}-beat structure.

FORBIDDEN:
- Do NOT add new story content.
- Do NOT expand or embellish.
- Do NOT change meaning.
- Do NOT add meta-commentary.

FORMAT (STRICT):
- Output exactly {expected_beats} beats.
- Each beat MUST start with: Beat X: (X = 1..{expected_beats})
- Separate beats using \"\\n\\n⁂\\n\\n\" ONLY.
- Start immediately with \"Beat 1:\".
- Do NOT include the separator (⁂) inside any beat.

OUTPUT:
Return the reformatted beats only."""

    format_user = f"RAW TEXT TO REFORMAT:\n{text}"

    repaired, _t = _generate_content(
        [{"role": "system", "content": format_system}, {"role": "user", "content": format_user}],
        model_or_engine,
        tokenizer,
        use_vllm,
        temperature=0.2,
        max_tokens=2500,
        device=device,
    )

    repaired_beats = _split_beats_strict(repaired)
    ok2, reason2 = _validate_beats_strict(repaired_beats, expected_beats=expected_beats)
    if not ok2:
        logger.warning(f"⚠️ {step_name}: format-only repair failed ({reason2}); returning original output")
        return text.strip()

    logger.info(f"✅ {step_name}: format-only repair succeeded")
    return repaired.strip()

def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text and parse it."""
    try:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None

def _apply_transitions_to_beats(beats: list, transitions: list) -> list:
    """
    Apply 1-sentence transitions between beats in code, without rewriting existing beat text.
    Strategy: append transition as a micro-continuation at the end of the prior beat,
    avoiding a hard paragraph break so it doesn't read like a bolted-on bridge.
    """
    if not transitions:
        return beats

    # Transition safety filters: transitions must never "do the next beat".
    # They should only add state/anticipation, not action/anatomy/acts.
    FORBIDDEN_SUBSTRINGS = [
        # action verbs / procedural language
        "enter", "entered", "insert", "inserted", "thrust", "thrusts", "push", "pulled", "pulls",
        "guide", "guides", "guided", "turn", "turned", "move", "moves", "moved", "position", "positions",
        "spread", "spreads", "spreading", "reach", "reaches", "reached", "grip", "grips", "gripping",
        "kiss", "kisses", "kissing", "lick", "licks", "licking", "suck", "sucks", "sucking",
        # stage directions
        "transition to", "cut to", "shift to",
        # sex acts / escalation cues (keep broad)
        "anal", "oral", "blowjob", "rim", "rimming", "penetrat", "fuck", "fucking",
        # anatomy/contact terms
        "anus", "asshole", "sphincter", "rectum", "cheeks", "cock", "dick", "pussy", "clit", "nipple",
        "tongue", "mouth", "lips",
    ]
    FORBIDDEN_MULTIWORD = [
        "from behind",
        "hands and knees",
    ]

    def _is_safe_transition(txt: str) -> bool:
        t = (txt or "").strip()
        if not t:
            return False
        low = t.lower()
        if any(w in low for w in FORBIDDEN_MULTIWORD):
            return False
        if any(s in low for s in FORBIDDEN_SUBSTRINGS):
            return False
        # Avoid "Name + verb" starts that often cause mini-beats.
        # e.g. "Neil shifts..." / "Sarah turns..."
        try:
            import re
            if re.match(r"^[A-Z][a-z]+\s+\w+", t):
                first_two = " ".join(t.split()[:2]).lower()
                if any(v in first_two for v in ["shifts", "guides", "turns", "moves", "reaches", "spreads", "enters", "kisses"]):
                    return False
        except Exception:
            pass
        return True

    def _transition_bleeds_into_next(txt: str, next_beat: str) -> bool:
        """
        Cheap overlap heuristic: if the transition shares too many non-trivial words with the next beat,
        it's probably describing what happens next instead of anticipation.
        """
        try:
            import re
            stop = {"the","a","an","and","or","but","to","of","in","on","at","with","for","from","into","as","is","was","were","be","been","her","his","their","she","he","they","them","him"}
            t_words = [w for w in re.findall(r"[a-zA-Z']+", (txt or "").lower()) if len(w) >= 4 and w not in stop]
            n_words = [w for w in re.findall(r"[a-zA-Z']+", (next_beat or "").lower()) if len(w) >= 4 and w not in stop]
            if not t_words or not n_words:
                return False
            inter = set(t_words) & set(n_words)
            # If most of the transition's meaningful words already appear in the next beat, it's a spoiler.
            ratio = len(inter) / max(1, len(set(t_words)))
            return (len(inter) >= 5 and ratio >= 0.45)
        except Exception:
            return False

    def _clean_transition(txt: str) -> str:
        """
        Deterministic transition cleanup:
        - strip leading/trailing whitespace/quotes
        - remove leading numbering like "0:" / "1."
        - keep it short and stateful (prompt + filters enforce semantics)
        """
        if not txt:
            return ""
        t = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
        # Remove leading quote characters that sometimes appear
        t = t.lstrip(' "\'“”')
        # Remove leading numeric helper prefixes (e.g., "0: " / "1. ")
        import re
        t = re.sub(r"^\s*\d+\s*[:.)]\s*", "", t)
        # Collapse internal whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    by_pair: Dict[str, str] = {}
    for t in transitions:
        try:
            pair = t.get("between")
            txt = _clean_transition((t.get("text") or ""))
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a, b = int(pair[0]), int(pair[1])
            if a < 1 or b != a + 1:
                continue
            if not txt:
                continue
            # Hard caps: 1–2 short sentences, <= 180 chars total
            if len(txt) > 180:
                continue
            # Sentence count heuristic (avoid "..." being counted as multiple)
            import re
            sent_count = len([m for m in re.findall(r"[.!?]+", txt) if m != "..."])
            if sent_count == 0:
                # Allow a single sentence without terminal punctuation
                sent_count = 1
            if sent_count > 2:
                continue
            if "⁂" in txt or "Beat" in txt:
                continue
            if not _is_safe_transition(txt):
                continue
            by_pair[f"{a}-{b}"] = txt
        except Exception:
            continue

    out = list(beats)
    for i in range(1, len(beats)):  # between i and i+1
        key = f"{i}-{i+1}"
        if key in by_pair:
            prior = out[i - 1].rstrip()
            transition = by_pair[key].strip()
            # If the transition likely bleeds into Beat N+1, skip it.
            if _transition_bleeds_into_next(transition, out[i]):
                continue
            # Apply as a soft separate paragraph (keeps prose readable, avoids duplicating next beat mechanics).
            out[i - 1] = prior + "\n\n" + transition
    return out

def _safe_trim_to_max(cleaned_text: str, max_chars: int) -> str:
    """
    Trim text to max_chars trying to cut at a sentence boundary.
    Used as an absolute final enforcement (hard max).
    """
    if len(cleaned_text) <= max_chars:
        return cleaned_text
    try:
        truncated = cleaned_text[:max_chars]
        # Try to cut at last sentence end in the truncated window
        for sep in [".", "!", "?"]:
            idx = truncated.rfind(sep)
            if idx > max_chars * 0.85:
                return truncated[: idx + 1].rstrip()
        # Fallback: last newline
        nl = truncated.rfind("\n")
        if nl > max_chars * 0.85:
            return truncated[:nl].rstrip()
        return truncated.rstrip()
    except Exception:
        return cleaned_text[:max_chars].rstrip()

def _build_keyword_constraints_text(beat_keyword_map: Dict[int, list], keyword_constraints: list) -> str:
    """
    Build a strict, beat-scoped constraints section for Step 2.
    Uses both semantic instructions and lexical anchors (when provided) to force specificity.
    """
    if not beat_keyword_map:
        return ""
    by_id: Dict[str, Dict[str, Any]] = {}
    for k in keyword_constraints or []:
        try:
            kid = str(k.get("id"))
            if not kid:
                continue
            by_id[kid] = k
        except Exception:
            continue

    lines = ["KEYWORD CONSTRAINTS (STRICT):"]
    lines.append("- Each listed keyword MUST be clearly and explicitly depicted in its assigned beat.")
    lines.append("- Do NOT satisfy a keyword by vague euphemism.")
    lines.append("- If a keyword is assigned to Beat X, do NOT introduce that act in other beats unless explicitly listed.")
    lines.append("")

    for beat_num in sorted(beat_keyword_map.keys()):
        ids = beat_keyword_map.get(beat_num) or []
        if not ids:
            continue
        # Special handling: if both anal + ass_to_mouth are in same beat, enforce strict intra-beat order.
        has_anal = any(str(x).strip().lower() == "anal" for x in ids)
        has_atm = any(str(x).strip().lower() == "ass_to_mouth" for x in ids)
        pretty = []
        if has_anal and has_atm:
            pretty.append(f"- Beat {beat_num} MUST include BOTH Anal and Ass-to-Mouth in this strict order:")
            pretty.append(f"  1) Anal penetration (with preparation + explicit anatomy).")
            pretty.append(f"  2) Withdrawal/cleanup moment (brief, non-climax).")
            pretty.append(f"  3) Ass-to-mouth contact after anal (explicitly described; no euphemism).")
            pretty.append(f"  - Do NOT move climax/release into this beat.")
        for kid in ids:
            meta = by_id.get(kid, {})
            name = meta.get("name") or kid
            instr = (meta.get("instruction") or "").strip()
            anchors = meta.get("anchors") if isinstance(meta.get("anchors"), list) else []
            if instr and anchors:
                pretty.append(f"- Beat {beat_num} MUST include: {name}. {instr} Use explicit lexical anchors like: {', '.join(anchors)}.")
            elif instr:
                pretty.append(f"- Beat {beat_num} MUST include: {name}. {instr}")
            else:
                pretty.append(f"- Beat {beat_num} MUST include: {name}.")
        lines.extend(pretty)

    return "\n".join(lines).strip()

def _validate_keyword_coverage(beats: list, beat_keyword_map: Dict[int, list], keyword_constraints: list) -> Dict[int, list]:
    """
    Return missing keywords by beat number.
    Uses lexical anchors when available, otherwise falls back to keyword id/name presence.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for k in keyword_constraints or []:
        try:
            kid = str(k.get("id"))
            if kid:
                by_id[kid] = k
        except Exception:
            continue

    missing: Dict[int, list] = {}
    for beat_num, ids in (beat_keyword_map or {}).items():
        if beat_num < 1 or beat_num > len(beats):
            continue
        text = beats[beat_num - 1].lower()
        for kid in ids or []:
            meta = by_id.get(kid, {})
            anchors = meta.get("anchors") if isinstance(meta.get("anchors"), list) else []
            ok = False
            # Special-case ass_to_mouth: accept "ass[- ]to[- ]mouth" even if anchors differ.
            if str(kid).lower() == "ass_to_mouth":
                try:
                    import re
                    if re.search(r"ass[\s\-]*to[\s\-]*mouth", text):
                        ok = True
                except Exception:
                    pass
            if anchors:
                ok = any(a.lower() in text for a in anchors if isinstance(a, str) and a.strip())
            if not ok:
                # fallback to id or name substring
                name = str(meta.get("name") or "").lower()
                if kid.lower() in text or (name and name.lower() in text):
                    ok = True
            if not ok:
                if beat_num not in missing:
                    missing[beat_num] = []
                missing[beat_num].append(kid)
    return missing

def _assign_keywords_to_beats_with_llm(
    outline_text: str,
    keyword_constraints: list,
    *,
    model_or_engine,
    tokenizer,
    use_vllm: bool,
    device: str,
) -> Dict[int, list]:
    """
    JSON-only keyword-to-beat assignment step.
    Returns a dict beat_num -> [keyword_id].
    Falls back to deterministic mapping if JSON parsing fails.
    """
    if not keyword_constraints:
        return {}

    # Compact keyword list for model
    kw_list = []
    for k in keyword_constraints:
        try:
            kw_list.append({
                "id": k.get("id"),
                "category": k.get("category"),
                "name": k.get("name"),
            })
        except Exception:
            continue

    system = """You are a story planner.

TASK:
Assign each keyword to the most appropriate beat(s) in a 12-beat outline.

RULES:
- Do NOT write story text.
- Do NOT invent new keywords.
- Each keyword must be assigned to at least one beat.
- Sexual acts/positions should be assigned only to beats where sexual activity occurs (typically Beats 6–11).
- Do NOT assign orgasm-only keywords outside Beat 11.

FORMAT (JSON ONLY):
{
  "beatKeywords": {
    "7": ["anal"]
  }
}"""

    user = f"""Outline:
{outline_text}

Keywords:
{json.dumps(kw_list)}"""

    out_text, _t = _generate_content(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_or_engine,
        tokenizer,
        use_vllm,
        temperature=0.2,
        max_tokens=800,
        device=device,
    )
    obj = _parse_json_from_text(out_text) or {}
    beat_keywords = obj.get("beatKeywords") if isinstance(obj.get("beatKeywords"), dict) else {}

    # Convert keys to int
    mapping: Dict[int, list] = {}
    try:
        for k, v in beat_keywords.items():
            bn = int(k)
            if not isinstance(v, list):
                continue
            ids = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    ids.append(item.strip())
            if ids:
                mapping[bn] = ids
    except Exception:
        mapping = {}

    # Deterministic fallback: assign sexual activities to Beat 7/8, positions to Beat 9, others ignored
    if not mapping:
        mapping = {}
        acts = [k.get("id") for k in kw_list if str(k.get("category", "")).lower() == "sexual activity"]
        positions = [k.get("id") for k in kw_list if str(k.get("category", "")).lower() == "sexual position"]
        acts = [a for a in acts if isinstance(a, str) and a]
        positions = [p for p in positions if isinstance(p, str) and p]
        if acts:
            mapping[7] = [acts[0]]
            if len(acts) > 1:
                mapping[8] = [acts[1]]
        if positions:
            mapping[9] = [positions[0]]
    return mapping

def _normalize_keyword_id(token: str, keyword_constraints: list) -> Optional[str]:
    """
    Map model outputs like 'ass to mouth' -> 'ass_to_mouth' using known ids/names.
    """
    if not token:
        return None
    t = str(token).strip().lower()
    t = t.replace("-", "_").replace(" ", "_")
    # Build lookup from constraints
    ids = set()
    name_map: Dict[str, str] = {}
    for k in keyword_constraints or []:
        try:
            kid = str(k.get("id") or "").strip().lower()
            if kid:
                ids.add(kid)
            nm = str(k.get("name") or "").strip().lower()
            if nm:
                name_map[nm.replace(" ", "_").replace("-", "_")] = kid
        except Exception:
            continue
    if t in ids:
        return t
    if t in name_map and name_map[t]:
        return name_map[t]
    return None

def _ensure_all_keywords_assigned(beat_keyword_map: Dict[int, list], keyword_constraints: list) -> Dict[int, list]:
    """
    Ensure every selected keyword id appears at least once in beat_keyword_map.
    Also resolve dependency: ass_to_mouth should co-locate with anal if both selected.
    """
    selected_ids = []
    for k in keyword_constraints or []:
        try:
            kid = str(k.get("id") or "").strip()
            if kid:
                selected_ids.append(kid.lower())
        except Exception:
            continue

    # Normalize any existing tokens to canonical ids
    normalized: Dict[int, list] = {}
    for bn, toks in (beat_keyword_map or {}).items():
        ids = []
        for tok in toks or []:
            kid = _normalize_keyword_id(tok, keyword_constraints)
            if kid:
                ids.append(kid)
        if ids:
            # de-dupe while preserving order
            seen = set()
            out = []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            normalized[int(bn)] = out

    present = set()
    for ids in normalized.values():
        for kid in ids:
            present.add(kid)

    missing = [kid for kid in selected_ids if kid not in present]

    # Capability heuristics: place sexual activities into Beat 10 by default (pre-climax, explicit allowed),
    # positions into Beat 9, everything else into Beat 4/5 if needed.
    def _default_beat_for(kid: str) -> int:
        cat = ""
        for k in keyword_constraints or []:
            try:
                if str(k.get("id") or "").strip().lower() == kid:
                    cat = str(k.get("category") or "").strip().lower()
                    break
            except Exception:
                continue
        if cat == "sexual position":
            return 9
        if cat == "sexual activity":
            return 10
        return 5

    for kid in missing:
        bn = _default_beat_for(kid)
        normalized.setdefault(bn, [])
        normalized[bn].append(kid)

    # Dependency rule: if both anal and ass_to_mouth, co-locate in the same beat (Beat 10 by default)
    if "anal" in selected_ids and "ass_to_mouth" in selected_ids:
        # find where anal is
        anal_beat = None
        for bn, ids in normalized.items():
            if "anal" in ids:
                anal_beat = bn
                break
        if anal_beat is None:
            anal_beat = 10
            normalized.setdefault(anal_beat, []).append("anal")
        # ensure ass_to_mouth in same beat
        # remove from other beats first
        for bn, ids in list(normalized.items()):
            if bn != anal_beat and "ass_to_mouth" in ids:
                normalized[bn] = [x for x in ids if x != "ass_to_mouth"]
                if not normalized[bn]:
                    del normalized[bn]
        normalized.setdefault(anal_beat, [])
        if "ass_to_mouth" not in normalized[anal_beat]:
            normalized[anal_beat].append("ass_to_mouth")

    # final de-dupe per beat
    for bn, ids in list(normalized.items()):
        seen = set()
        out = []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        normalized[bn] = out
    return normalized

def _micro_add_to_reach_min_v2(
    structured_text: str,
    min_chars: int,
    *,
    model_or_engine,
    tokenizer,
    use_vllm: bool,
    device: str,
) -> str:
    """
    Delta-only micro-add pass to reach soft minimum.
    Produces JSON with short append sentences (no dialogue, no new actions) and applies in code.
    """
    cleaned_len = len(clean_story(structured_text))
    if cleaned_len >= min_chars:
        return structured_text

    beats = _split_beats_strict(structured_text)
    ok, _reason = _validate_beats_strict(beats, expected_beats=12)
    if not ok:
        return structured_text

    delta = min_chars - cleaned_len
    # Add at most 1–2 sentences to up to 10 beats (avoid Beat 11 climax, avoid Beat 12 no-sex beat).
    # This is a cheap bump; a stronger per-beat recovery pass exists later if we're far below min.
    max_targets = 10
    approx_per = max(60, min(140, int((delta / max_targets) + 20)))
    # Choose number of beats to modify (assume ~120 chars per beat)
    beats_to_modify = min(max_targets, max(1, int((delta + 119) // 120)))
    target_ids = list(range(1, beats_to_modify + 1))

    system = f"""We are a micro-expansion editor.

TASK:
Provide short, non-plot-changing append sentences for specific beats to slightly increase length.

STRICT RULES:
- Do NOT add new actions or events.
- Do NOT add dialogue.
- Do NOT add new sexual acts.
- Only add sensory detail, internal thought, or environmental description.
- Each append must be 1–2 short sentences only.
- Each append must be ~{approx_per} characters (±40).
- Do NOT mention beat numbers in the sentence.

FORMAT:
Return ONLY valid JSON:
{{
  "appends": [
    {{ "beat": 1, "text": "..." }}
  ]
}}"""

    # Provide compact context per target beat (strip label content)
    context_lines = []
    for i in target_ids:
        b = beats[i - 1]
        body = b.split(":", 1)[1].strip() if ":" in b else b.strip()
        context_lines.append(f"Beat {i} context: {body[:260]}")
    user = "We are below the minimum length and need small, safe additions.\n\n" + \
           f"Provide appends for beats: {', '.join(str(i) for i in target_ids)}.\n\n" + \
           "Return only JSON.\n\n" + \
           "\n".join(context_lines)

    out_text, _t = _generate_content(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_or_engine,
        tokenizer,
        use_vllm,
        temperature=0.4,
        max_tokens=800,
        device=device,
    )
    obj = _parse_json_from_text(out_text) or {}
    appends = obj.get("appends") if isinstance(obj.get("appends"), list) else []

    # Apply appends
    for item in appends:
        try:
            beat_id = int(item.get("beat"))
            txt = (item.get("text") or "").strip()
            if beat_id not in target_ids:
                continue
            if not txt or "⁂" in txt or "Beat" in txt:
                continue
            beats[beat_id - 1] = beats[beat_id - 1].rstrip() + "\n\n" + txt
        except Exception:
            continue

    rebuilt = _BEAT_SEPARATOR.join([b.strip() for b in beats])
    return rebuilt

def _trim_beat_to_max(beat_text: str, max_body_chars: int) -> str:
    """Trim a single beat body (keeping Beat X: label) to max_body_chars, sentence-aware."""
    beat_text = (beat_text or "").strip()
    if not beat_text:
        return beat_text
    label = ""
    body = beat_text
    if ":" in beat_text[:20]:
        label, body = beat_text.split(":", 1)
        label = label.strip() + ":"
        body = body.strip()
    if len(body) <= max_body_chars:
        return f"{label} {body}".strip() if label else body
    cut = body[:max_body_chars]
    for sep in [".", "!", "?"]:
        idx = cut.rfind(sep)
        if idx > max_body_chars * 0.85:
            cut = cut[:idx + 1]
            break
    cut = cut.rstrip()
    return f"{label} {cut}".strip() if label else cut

def _sanitize_structured_beats(text: str, expected_beats: int = 12) -> str:
    """
    Clean a structured Beat 1..N story without destroying the structure.
    - removes per-line numeric junk (e.g. "0:")
    - removes leading quotes/whitespace per line
    - removes nested Beat labels inside a beat body
    """
    beats = _split_beats_strict(text)
    ok, _reason = _validate_beats_strict(beats, expected_beats=expected_beats)
    if not ok:
        return text.strip()

    import re
    cleaned = []
    for i, beat in enumerate(beats, 1):
        if ":" in beat:
            label, body = beat.split(":", 1)
            label = f"Beat {i}:"
            body = body
        else:
            label = f"Beat {i}:"
            body = beat
        # Normalize newlines and strip whitespace/quotes per line
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        body = re.sub(r'(?m)^\s*["“”]\s*', '', body)
        body = re.sub(r'(?m)^\s*\d+\s*[:.)]\s*', '', body)  # remove "0:" etc
        body = "\n".join(line.strip() for line in body.split("\n"))
        # Remove accidental nested beat labels inside the body
        body = re.sub(r'(?im)\bBeat\s*\d+\s*:\s*', '', body)
        # Collapse blank lines
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        # Beat 1: strip expository framing if it leaked into the beat body
        if i == 1:
            low = body.lstrip().lower()
            forbidden = (
                "the story starts",
                "the story takes place",
                "the story opens",
                "this story",
                "the scene opens",
                "the narrative",
                "the following story",
            )
            if low.startswith(forbidden):
                # Drop first sentence
                parts = body.split(".", 1)
                if len(parts) > 1:
                    body = parts[1].lstrip()
                else:
                    body = ""
        cleaned.append(f"{label} {body}".strip())
    return _BEAT_SEPARATOR.join(cleaned).strip()

def _recover_length_per_beat_v2(
    structured_text: str,
    beat_keyword_map: Dict[int, list],
    keyword_constraints: list,
    *,
    model_or_engine,
    tokenizer,
    use_vllm: bool,
    device: str,
) -> str:
    """
    Per-beat densifier recovery. Only runs on beats that dropped below _BEAT_MIN.
    This defends length against entropy from editorial steps.
    """
    beats = _split_beats_strict(structured_text)
    ok, _reason = _validate_beats_strict(beats, expected_beats=12)
    if not ok:
        return structured_text

    # Compute beat body lengths
    body_lens: Dict[int, int] = {}
    for i, b in enumerate(beats, 1):
        body = b.split(":", 1)[1].strip() if ":" in b else b.strip()
        body_lens[i] = len(body)

    if min(body_lens.values()) >= _BEAT_MIN:
        return structured_text

    # Lookup for keyword anchors per beat (short constraint snippet)
    by_id: Dict[str, Dict[str, Any]] = {}
    for k in keyword_constraints or []:
        try:
            kid = str(k.get("id") or "").strip()
            if kid:
                by_id[kid] = k
        except Exception:
            continue

    def _beat_constraint_snippet(beat_num: int) -> str:
        ids = beat_keyword_map.get(beat_num) or []
        if not ids:
            return "- No special keyword constraints for this beat."
        parts = []
        for kid in ids:
            meta = by_id.get(kid, {})
            name = meta.get("name") or kid
            anchors = meta.get("anchors") if isinstance(meta.get("anchors"), list) else []
            if anchors:
                parts.append(f"- Must satisfy keyword: {name} (anchors: {', '.join(anchors)}).")
            else:
                parts.append(f"- Must satisfy keyword: {name}.")
        return "\n".join(parts)

    # Expand only beats under min
    for i in range(1, 13):
        if body_lens.get(i, 0) >= _BEAT_MIN:
            continue
        beat = beats[i - 1].strip()
        system = f"""You are a narrative densifier for one beat.

TASK:
Rewrite Beat {i} to increase descriptive density while preserving the same events and meaning.

STRICT RULES:
- Do NOT add new plot events.
- Do NOT add new sexual acts. Only elaborate what is already happening.
- Do NOT add dialogue.
- Do NOT add climax/release outside Beat 11.
- Preserve names, POV, tense, and continuity.
- You MAY remove non-narrative artifacts (e.g., stray numbering like ".0:", formatting remnants).

LENGTH SAFETY (CRITICAL):
- Beat body MUST be {_BEAT_MIN}–{_BEAT_MAX} characters (aim ~{_BEAT_TARGET}).

FORMAT:
- Output ONLY the revised Beat {i} text.
- Keep the label exactly: \"Beat {i}:\"."""

        user = f"""Constraints for Beat {i}:
{_beat_constraint_snippet(i)}

Beat {i} to densify:
{beat}

Return ONLY the revised Beat {i} text."""

        revised, _t = _generate_content(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model_or_engine,
            tokenizer,
            use_vllm,
            temperature=0.4,
            max_tokens=900,
            device=device,
        )
        revised = (revised or "").strip()
        if not revised.lstrip().startswith(f"Beat {i}:"):
            revised = f"Beat {i}: " + revised
        # Trim if wildly over
        revised = _trim_beat_to_max(revised, _BEAT_MAX)
        beats[i - 1] = revised

    return _BEAT_SEPARATOR.join([b.strip() for b in beats])

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
        
        # Extract callback_url early and log it
        # Check multiple locations: top-level input, top-level metadata, and nested input.metadata
        input_metadata = input_data.get("metadata", {})
        callback_url = (
            input_data.get("callback_url") 
            or metadata.get("callback_url") 
            or input_metadata.get("callback_url")
        )
        logger.info(f"🔔 Callback URL received: {callback_url if callback_url else 'NOT PROVIDED'}")
        logger.info(f"📋 Input data keys: {list(input_data.keys())}")
        logger.info(f"📋 Metadata keys: {list(metadata.keys())}")
        
        # Extract parameters
        messages = input_data.get("messages", [])
        temperature = input_data.get("temperature", 0.7)
        language = input_data.get("language")
        genre = input_data.get("genre")
        age_range = input_data.get("age_range")
        
        # Workflow selection
        workflow_type = _normalize_workflow_type(input_data, metadata)
        outline_messages = input_data.get("outline_messages", [])
        story_messages = input_data.get("story_messages", [])
        is_multi_step_v2 = workflow_type == "multi-step-v2"
        # Legacy two-step workflow (outline + story prompt injection)
        needs_two_step = (workflow_type == "two-step") or (outline_messages and story_messages and not is_multi_step_v2)
        
        # Default max_tokens (only used for single-step workflow)
        # For two-step workflow, we use configurable values: outline_max_tokens (default 5000) and max_tokens (default 5000 for story)
        # Token analysis:
        # - Outline: ~475-725 tokens needed, but can be longer with detailed beats, 5000 allows for expansion
        # - Story: ~3,025-3,425 tokens needed, 5000 is safer
        # - Single-step Qwen3: ~3,025-3,425 tokens needed, 4000 is tight but acceptable
        default_max_tokens = 4000 if genre and genre.lower() in ['qwen3', 'qwen'] else 4000
        max_tokens = input_data.get("max_tokens", default_max_tokens)
        outline_max_tokens = input_data.get("outline_max_tokens", 5000)  # Default 5000 for beat creation
        expansion_max_tokens = input_data.get("expansion_max_tokens", 4000)  # Default 4000 for Step 3 expansion
        finetune_max_tokens = input_data.get("finetune_max_tokens", 4000)  # Default 4000 for Step 4 finetuning
        
        # Qwen3 is adult-oriented, not age-specific - use +18 as default
        if genre and genre.lower() in ['qwen3', 'qwen']:
            age_range = "+18"
        
        user_id = input_data.get("user_id") or metadata.get("user_id")
        story_id = input_data.get("story_id") or metadata.get("story_id")
        # Re-check callback_url (may have been set earlier, but ensure we have the latest)
        callback_url = (
            input_data.get("callback_url") 
            or metadata.get("callback_url") 
            or input_metadata.get("callback_url")
        )
        
        # Get mode and tone for expansion prompt (if needed)
        mode = input_data.get("mode", "sensual")
        mode_tone_map = {
            "soft": "Gentle, romantic, and tender. Focus on emotional connection and sensuality.",
            "sensual": "Passionate, erotic, and detailed. Focus on physical sensations and desire.",
            "hardcore": "Explicit, intense, and graphic. Focus on raw physicality and explicit descriptions.",
            "extreme": "Most vulgar, explicit, and graphic. No moral boundaries. Use the crudest language and most explicit descriptions."
        }
        mode_tone = mode_tone_map.get(mode, mode_tone_map["sensual"])
        
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
        logger.info(f"🔄 Two-step workflow: {needs_two_step}")
        logger.info(f"🧩 Multi-step-v2 workflow: {is_multi_step_v2}")
        if needs_two_step:
            logger.info(f"📝 Outline messages count: {len(outline_messages)}")
            logger.info(f"📖 Story messages count: {len(story_messages)}")
            if outline_messages:
                logger.info("=" * 80)
                logger.info("📝 FULL OUTLINE PROMPT:")
                logger.info("=" * 80)
                for i, msg in enumerate(outline_messages, 1):
                    logger.info(f"\n--- Message {i} ({msg.get('role', 'unknown')}) ---")
                    logger.info(msg.get("content", ""))
                logger.info("=" * 80)
            if story_messages:
                logger.info("=" * 80)
                logger.info("📖 FULL STORY PROMPT:")
                logger.info("=" * 80)
                for i, msg in enumerate(story_messages, 1):
                    logger.info(f"\n--- Message {i} ({msg.get('role', 'unknown')}) ---")
                    logger.info(msg.get("content", ""))
                logger.info("=" * 80)
        
        # Log message preview
        if messages:
            logger.info("=" * 80)
            logger.info("📝 FULL MESSAGES PROMPT:")
            logger.info("=" * 80)
            for i, msg in enumerate(messages, 1):
                logger.info(f"\n--- Message {i} ({msg.get('role', 'unknown')}) ---")
                logger.info(msg.get("content", ""))
            logger.info("=" * 80)
        
        if not messages and not needs_two_step and not is_multi_step_v2:
            raise ValueError("messages is required (or outline_messages + story_messages for two-step)")
        
        # Load model if not already loaded
        model_or_engine, tokenizer, device = _load_model()
        use_vllm = _use_vllm

        # MULTI-STEP-V2 WORKFLOW: Deterministic 12-beat pipeline (Steps 2–8 orchestrated server-side)
        if is_multi_step_v2:
            expected_beats = 12
            enable_transitions = bool(input_data.get("enable_transitions", False))
            enable_dialogue = bool(input_data.get("enable_dialogue", False))
            enable_motifs = bool(input_data.get("enable_motifs", False))
            keyword_constraints = input_data.get("keyword_constraints", []) or []

            # Tight step budgets (can be overridden via input_data)
            outline_max_tokens_v2 = int(input_data.get("outline_max_tokens", 800) or 800)
            step2_max_tokens = int(input_data.get("max_tokens", 6000) or 6000)  # Step 2 budget
            step3_max_tokens = int(input_data.get("transitions_max_tokens", 800) or 800)
            step4_max_tokens = int(input_data.get("motifs_max_tokens", 1200) or 1200)
            step5_max_tokens = int(input_data.get("dialogue_max_tokens", 600) or 600)
            step6_max_tokens = int(input_data.get("climax_max_tokens", 800) or 800)
            step7_max_tokens = int(input_data.get("dedup_max_tokens", 7500) or 7500)
            step8a_max_tokens = int(input_data.get("metadata_max_tokens", 1500) or 1500)
            step8b_max_tokens = int(input_data.get("cover_max_tokens", 800) or 800)

            logger.info("=" * 80)
            logger.info("🧩 MULTI-STEP-V2 PIPELINE START")
            logger.info("=" * 80)
            logger.info(f"Flags: transitions={enable_transitions}, dialogue={enable_dialogue}, motifs={enable_motifs}")

            # Step 1: Generate outline (beats skeleton) using provided outline_messages
            if not outline_messages:
                raise ValueError("multi-step-v2 requires outline_messages (Step 1 prompt)")
            logger.info("=" * 80)
            logger.info("📝 STEP 1: Generating beat skeleton (outline)...")
            logger.info("=" * 80)
            outline_text, outline_time = _generate_content(
                outline_messages,
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens=outline_max_tokens_v2,
                device=device
            )
            outline_text = _ensure_structured_beats(
                outline_text,
                expected_beats,
                model_or_engine=model_or_engine,
                tokenizer=tokenizer,
                use_vllm=use_vllm,
                device=device,
                step_name="Step1_outline",
            )

            # NEW: Keyword → Beat binding (JSON-only) so keywords can't silently drop.
            beat_keyword_map: Dict[int, list] = {}
            if keyword_constraints:
                logger.info("=" * 80)
                logger.info("🏷️ STEP 1.5: Assigning keywords to beats (JSON-only)...")
                logger.info("=" * 80)
                beat_keyword_map = _assign_keywords_to_beats_with_llm(
                    outline_text,
                    keyword_constraints,
                    model_or_engine=model_or_engine,
                    tokenizer=tokenizer,
                    use_vllm=use_vllm,
                    device=device,
                )
                # Guarantee every selected keyword is assigned at least once (and normalize ids/names).
                beat_keyword_map = _ensure_all_keywords_assigned(beat_keyword_map, keyword_constraints)
                # Store for debugging
                try:
                    metadata["beatKeywordMap"] = {str(k): v for k, v in beat_keyword_map.items()}
                except Exception:
                    pass

            # Step 2: Physicalization (expanded beats, no climax except Beat 11)
            logger.info("=" * 80)
            logger.info("📖 STEP 2: Physicalization (expand beats; climax locked to Beat 11)...")
            logger.info("=" * 80)
            step2_system = """We are a precision erotic writer.

TASK:
Expand each beat into detailed narrative prose.

RULES:
- Expand each beat into a complete scene of ~850–900 characters.
- Keep most beats within 780–930 characters (small variance is allowed).
- Follow the outline exactly; do not change event order.
- Add concrete physical actions, positions, and spatial detail.
- Graphic description IS allowed.
- CRITICAL OPENING RULE: Beat 1 MUST start in-scene with observable action/posture (no expository framing like "The story takes place...").
- Introduce named adult protagonists early (Beat 1) and refer to them by name thereafter.
- Do NOT refer to characters as "the man", "the woman" except at first introduction.
- Do NOT write outline/stage-direction phrases like "Transition to..." / "They change positions..." / "The tension peaks...".
- Each beat must read as in-scene storytelling, not a bullet summary.
- If a beat has a keyword constraint for a sexual act, you MUST depict the act explicitly with act-specific anatomy and mechanics.
- Do not euphemize or generalize; include the required lexical anchors when provided.

CLIMAX CONTROL:
- A climax may ONLY occur in Beat 11.
- Beats 1–10 must NOT describe orgasm or release.
- Beat 12 must NOT contain sexual action.

STYLE RULES:
- Avoid abstract intensity words (e.g., "overwhelming", "crescendo").
- Prefer physical mechanics over emotional metaphors.

FORMAT (STRICT):
- Preserve beat labels ("Beat X:")
- Preserve "\\n\\n⁂\\n\\n" separators
- No titles, no commentary.

OUTPUT:
A fully expanded 12-beat story."""

            keyword_constraints_text = _build_keyword_constraints_text(beat_keyword_map, keyword_constraints)

            step2_user = f"""Expand the outline below into 12 expanded beats.

IMPORTANT:
- Preserve the beat labels exactly (Beat 1..Beat 12).
- Keep the same number of beats and the same beat order.
- Keep the same separator between beats: \\n\\n⁂\\n\\n.
- Aim for ~850–900 characters per beat. Keep most beats within 780–930.

{keyword_constraints_text if keyword_constraints_text else ''}

OUTLINE:
{outline_text}

Write the expanded beats now."""

            expanded_text, step2_time = _generate_content(
                [{"role": "system", "content": step2_system}, {"role": "user", "content": step2_user}],
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens=step2_max_tokens,
                device=device
            )
            expanded_text = _ensure_structured_beats(
                expanded_text,
                expected_beats,
                model_or_engine=model_or_engine,
                tokenizer=tokenizer,
                use_vllm=use_vllm,
                device=device,
                step_name="Step2_physicalization",
            )

            beats = _split_beats_strict(expanded_text)
            expanded_text = _sanitize_structured_beats(expanded_text, expected_beats=expected_beats)
            beats = _split_beats_strict(expanded_text)

            # Keyword validation after Step 2 (retry once if missing)
            if keyword_constraints and beat_keyword_map:
                missing = _validate_keyword_coverage(beats, beat_keyword_map, keyword_constraints)
                if missing:
                    logger.warning(f"⚠️ Step2 keyword validation failed (missing): {missing}. Retrying Step 2 once with stronger enforcement.")
                    missing_lines = []
                    for bn in sorted(missing.keys()):
                        missing_lines.append(f"- Beat {bn} missing: {', '.join(missing[bn])}")
                    enforcement = (
                        "\n\nCRITICAL RETRY NOTE:\n"
                        "Your previous output failed the keyword constraints.\n"
                        "You MUST explicitly depict the missing items in the specified beats, using explicit anatomy/mechanics and the lexical anchors when provided.\n"
                        + "\n".join(missing_lines)
                    )
                    step2_user_retry = step2_user + enforcement
                    expanded_text_retry2, step2_time_retry2 = _generate_content(
                        [{"role": "system", "content": step2_system}, {"role": "user", "content": step2_user_retry}],
                        model_or_engine,
                        tokenizer,
                        use_vllm,
                        temperature,
                        max_tokens=step2_max_tokens,
                        device=device
                    )
                    expanded_text_retry2 = _ensure_structured_beats(
                        expanded_text_retry2,
                        expected_beats,
                        model_or_engine=model_or_engine,
                        tokenizer=tokenizer,
                        use_vllm=use_vllm,
                        device=device,
                        step_name="Step2_physicalization_keyword_retry",
                    )
                    beats_retry2 = _split_beats_strict(expanded_text_retry2)
                    missing2 = _validate_keyword_coverage(beats_retry2, beat_keyword_map, keyword_constraints)
                    if missing2:
                        logger.warning(f"⚠️ Step2 keyword retry still missing: {missing2}. Proceeding; final story may not include all keywords.")
                        metadata["missingKeywordMap"] = {str(k): v for k, v in missing2.items()}
                    else:
                        expanded_text = expanded_text_retry2
                        step2_time = step2_time + step2_time_retry2
                        beats = beats_retry2

            # Step 2 checkpoint: reject and retry if out of allowed buffer.
            # We do NOT proceed to later steps if Step 2 fails the length contract.
            def _step2_clean_len(s: str) -> int:
                try:
                    return len(clean_story(s))
                except Exception:
                    return len(s)

            step2_len = _step2_clean_len(expanded_text)
            if step2_len < _STEP2_GATE_MIN or step2_len > _STEP2_GATE_MAX:
                logger.warning(
                    f"⚠️ Step2 checkpoint failed: cleaned_len={step2_len} (expected {_STEP2_GATE_MIN}–{_STEP2_GATE_MAX}). Retrying Step 2 once."
                )
                # One retry with the same prompt (we trust local constraints more than global targets)
                expanded_text_retry, step2_time_retry = _generate_content(
                    [{"role": "system", "content": step2_system}, {"role": "user", "content": step2_user}],
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature,
                    max_tokens=step2_max_tokens,
                    device=device
                )
                expanded_text_retry = _ensure_structured_beats(
                    expanded_text_retry,
                    expected_beats,
                    model_or_engine=model_or_engine,
                    tokenizer=tokenizer,
                    use_vllm=use_vllm,
                    device=device,
                    step_name="Step2_physicalization_retry",
                )
                step2_len_retry = _step2_clean_len(expanded_text_retry)
                if step2_len_retry < _STEP2_GATE_MIN or step2_len_retry > _STEP2_GATE_MAX:
                    logger.warning(
                        f"⚠️ Step2 retry still out of range: cleaned_len={step2_len_retry}. Continuing pipeline anyway, but final hard limits will be enforced."
                    )
                else:
                    expanded_text = expanded_text_retry
                    step2_time = step2_time + step2_time_retry
                    beats = _split_beats_strict(expanded_text)

            # Step 3 (optional): Transitions (JSON only; applied in code)
            if enable_transitions:
                logger.info("=" * 80)
                logger.info("🔗 STEP 3: Transitions (JSON-only; applied in code)...")
                logger.info("=" * 80)
                step3_system = """We are a narrative flow editor.

TASK:
Write short transition sentences that create anticipation between beats.

ABSOLUTE RULES:
- Do NOT describe physical actions.
- Do NOT describe sexual acts.
- Do NOT describe position changes.
- Do NOT describe anatomy or contact.
- Do NOT state what happens next.
- Do NOT add dialogue.
- Do NOT rewrite beats.

ALLOWED CONTENT:
- Emotional shift
- Physical proximity without contact
- Environmental change
- Internal anticipation
- Power dynamics
- Consent cues
- Silence, pauses, breath, eye contact

CONSTRAINTS:
- 1–2 short sentences.
- Max 180 characters total.
- Avoid stage-direction/meta phrasing ("Transition to...", "Cut to...", "Shift to...").
- Refer to characters by name/pronouns (avoid "the man", "the woman").

FORMAT:
Return ONLY valid JSON:
{
  "transitions": [
    { "between": [1, 2], "text": "..." }
  ]
}"""
                # Provide compact beat summaries to reduce context and avoid repetition loops
                summaries = []
                for i, b in enumerate(beats, 1):
                    b2 = b
                    if b2.lstrip().startswith(f"Beat {i}:"):
                        b2 = b2.split(":", 1)[1].strip()
                    summaries.append(f"Beat {i} summary: {b2[:220]}")
                step3_user = "Create one transition sentence between each adjacent beat (1->2 ... 11->12).\n\n" + "\n".join(summaries)
                transitions_text, _t3 = _generate_content(
                    [{"role": "system", "content": step3_system}, {"role": "user", "content": step3_user}],
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature=0.3,
                    max_tokens=step3_max_tokens,
                    device=device
                )
                transitions_json = _parse_json_from_text(transitions_text) or {}
                transitions_list = transitions_json.get("transitions") if isinstance(transitions_json.get("transitions"), list) else []
                metadata["transitions"] = transitions_list
                beats = _apply_transitions_to_beats(beats, transitions_list)

            # Step 4 (optional): Motif / keyword annotation (JSON only; no prose)
            if enable_motifs:
                logger.info("=" * 80)
                logger.info("🏷️ STEP 4: Motif annotation (JSON-only)...")
                logger.info("=" * 80)
                step4_system = """We are a story analyst.

TASK:
Assign thematic motifs to beats.

RULES:
- Do NOT generate prose.
- Do NOT rewrite text.
- Do NOT add actions.

FORMAT (STRICT JSON):
{
  "beatMotifs": {
    "6": ["authority", "instruction"],
    "7": ["submission"]
  }
}"""
                # Use outline only (cheaper and less prone to prose echo)
                step4_user = f"""Assign 1–3 motifs to any beats where it makes sense.

Return only JSON.

OUTLINE:
{outline_text}"""
                motifs_text, _t4 = _generate_content(
                    [{"role": "system", "content": step4_system}, {"role": "user", "content": step4_user}],
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature=0.3,
                    max_tokens=step4_max_tokens,
                    device=device
                )
                motifs_json = _parse_json_from_text(motifs_text) or {}
                metadata["beatMotifs"] = motifs_json.get("beatMotifs")

            # Step 5 (optional): Dialogue pass (per beat only; no new actions; no escalation)
            if enable_dialogue:
                logger.info("=" * 80)
                logger.info("🗣️ STEP 5: Dialogue pass (per beat)...")
                logger.info("=" * 80)
                step5_system = """We are a dialogue specialist.

TASK:
Add or refine dialogue for ONE beat only.

RULES:
- Do NOT add new physical actions.
- Do NOT escalate sexual intensity.
- Dialogue should reveal personality (voice, humor, insecurity, confidence) and consent cues.
- Max 240 additional characters total.
- Add at most 1–2 short lines of dialogue.
- Avoid generic labels ("the man", "the woman"); use names/pronouns.

OUTPUT:
Return ONLY the revised beat text."""
                revised_beats = []
                for i, beat in enumerate(beats, 1):
                    step5_user = f"""Revise ONLY the dialogue inside this beat.

CRITICAL:
- Do not add actions, positions, or new events.
- Do not escalate intensity.
- Keep the beat label intact: Beat {i}:

BEAT TO REVISE:
{beat}

Return ONLY the revised beat text."""
                    revised, _t5 = _generate_content(
                        [{"role": "system", "content": step5_system}, {"role": "user", "content": step5_user}],
                        model_or_engine,
                        tokenizer,
                        use_vllm,
                        temperature=0.5,
                        max_tokens=step5_max_tokens,
                        device=device
                    )
                    revised = revised.strip()
                    if not revised.lstrip().startswith(f"Beat {i}:"):
                        revised = f"Beat {i}: " + revised
                    revised_beats.append(revised)
                beats = revised_beats

            # Step 6: Climax pass (Beat 11 only)
            logger.info("=" * 80)
            logger.info("💥 STEP 6: Climax refinement (Beat 11 only)...")
            logger.info("=" * 80)
            step6_system = """We are an erotic climax specialist.

TASK:
Rewrite Beat 11 to deliver a single, sharp climax.

RULES:
- This is the ONLY beat allowed to climax.
- Do NOT extend length beyond +15%.
- LENGTH SAFETY (CRITICAL): Do NOT shrink the beat. Maintain the original beat length within ±5%.
- Do NOT repeat phrasing.
- No aftermath.

OUTPUT:
Return only Beat 11 text."""
            beat11 = beats[10] if len(beats) >= 11 else ""
            step6_user = f"""Rewrite ONLY Beat 11 for a single, sharp climax.

CRITICAL:
- Return only the revised Beat 11 text.
- Keep the label intact: Beat 11:

BEAT 11:
{beat11}"""
            beat11_revised, _t6 = _generate_content(
                [{"role": "system", "content": step6_system}, {"role": "user", "content": step6_user}],
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature=0.6,
                max_tokens=step6_max_tokens,
                device=device
            )
            beat11_revised = beat11_revised.strip()
            if not beat11_revised.lstrip().startswith("Beat 11:"):
                beat11_revised = "Beat 11: " + beat11_revised
            if len(beats) >= 11:
                beats[10] = beat11_revised

            # Re-assemble structured story (still beat-labeled + separators)
            structured_story = _BEAT_SEPARATOR.join([b.strip() for b in beats])

            # Step 7: Deduplication / polish (single full-story pass)
            logger.info("=" * 80)
            logger.info("🧼 STEP 7: Deduplication / polish (single full-story pass)...")
            logger.info("=" * 80)
            step7_system = """We are a professional prose editor.

TASK:
Remove repetition and improve variation.

STRICT RULES:
- Do NOT add new content.
- Do NOT intensify sexual acts.
- Do NOT add dialogue.
- LENGTH SAFETY (CRITICAL):
  - Do NOT reduce the length of any beat.
  - Maintain each beat length within ±5%.
  - If removing repetition, replace it with equivalent concrete physical/sensory detail.
- You MAY remove non-narrative artifacts (e.g., stray numbering like ".0:", formatting remnants, broken prefixes).

FORMAT:
Return the full story with identical structure:
- Preserve beat labels ("Beat X:") and order
- Preserve "\\n\\n⁂\\n\\n" separators
- No title, no headings, no meta commentary."""
            step7_user = f"""Edit the story below by removing repetition and improving variation.

CRITICAL:
- Do not add new content.
- Do not add dialogue.
- Do not intensify sexual acts.
- Keep the exact beat structure and separators.

STORY:
{structured_story}"""
            dedup_text, step7_time = _generate_content(
                [{"role": "system", "content": step7_system}, {"role": "user", "content": step7_user}],
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature=0.3,
                max_tokens=step7_max_tokens,
                device=device
            )
            dedup_text = _ensure_structured_beats(
                dedup_text,
                expected_beats,
                model_or_engine=model_or_engine,
                tokenizer=tokenizer,
                use_vllm=use_vllm,
                device=device,
                step_name="Step7_deduplicate",
            )
            dedup_text = _sanitize_structured_beats(dedup_text, expected_beats=expected_beats)
            generated_text = dedup_text
            generation_time = outline_time + step2_time + step7_time
            logger.info(f"✅ multi-step-v2 generated structured story: {len(generated_text)} chars")
            logger.info(f"⏱️ multi-step-v2 generation time (core): {generation_time:.2f}s")

            # Final length enforcement for multi-step-v2:
            # - soft min: 9000 (micro-add via deltas)
            # - hard max: 11500 (safe trim after cleaning)
            try:
                cleaned_len_final = len(clean_story(generated_text))
            except Exception:
                cleaned_len_final = len(generated_text)

            if cleaned_len_final < _SOFT_MIN_CHARS:
                logger.warning(f"⚠️ Below soft min after Step 7: cleaned_len={cleaned_len_final} (<{_SOFT_MIN_CHARS}). Applying micro-add deltas.")
                generated_text = _micro_add_to_reach_min_v2(
                    generated_text,
                    _SOFT_MIN_CHARS,
                    model_or_engine=model_or_engine,
                    tokenizer=tokenizer,
                    use_vllm=use_vllm,
                    device=device,
                )
                try:
                    cleaned_len_final = len(clean_story(generated_text))
                except Exception:
                    cleaned_len_final = len(generated_text)
                logger.info(f"✅ After micro-add: cleaned_len={cleaned_len_final}")

            # If we're still far below minimum or any beat collapsed below _BEAT_MIN,
            # run a stronger per-beat recovery pass (does not add new events/acts).
            try:
                beats_now = _split_beats_strict(generated_text)
                beat_body_lens = []
                for idx, b in enumerate(beats_now, 1):
                    body = b.split(":", 1)[1].strip() if ":" in b else b.strip()
                    beat_body_lens.append(len(body))
                any_beat_too_short = bool(beat_body_lens) and (min(beat_body_lens) < _BEAT_MIN)
            except Exception:
                any_beat_too_short = False

            if cleaned_len_final < _SOFT_MIN_CHARS or any_beat_too_short:
                logger.warning(
                    f"⚠️ Length still below target after micro-add (cleaned_len={cleaned_len_final}, any_beat_too_short={any_beat_too_short}). Running per-beat densifier recovery."
                )
                generated_text = _recover_length_per_beat_v2(
                    generated_text,
                    beat_keyword_map if isinstance(beat_keyword_map, dict) else {},
                    keyword_constraints if isinstance(keyword_constraints, list) else [],
                    model_or_engine=model_or_engine,
                    tokenizer=tokenizer,
                    use_vllm=use_vllm,
                    device=device,
                )
                try:
                    cleaned_len_final = len(clean_story(generated_text))
                except Exception:
                    cleaned_len_final = len(generated_text)
                logger.info(f"✅ After per-beat recovery: cleaned_len={cleaned_len_final}")

            # Hard max is enforced on cleaned text later, but we log here for visibility.
            if cleaned_len_final > _HARD_MAX_CHARS:
                logger.warning(f"⚠️ Above hard max after Step 7: cleaned_len={cleaned_len_final} (>{_HARD_MAX_CHARS}). Will trim after cleaning.")

            # Step 8: Metadata generation (fresh context; no beat labels)
            erotic_like = False
            try:
                genre_key = (genre or "").strip().lower()
                erotic_like = genre_key in ['erotic', 'advanced erotic', 'hardcore erotic', 'qwen3', 'qwen']
            except Exception:
                erotic_like = False

            if erotic_like:
                logger.info("=" * 80)
                logger.info("🧾 STEP 8: Generating title/preview/tags + cover prompt...")
                logger.info("=" * 80)

                story_for_metadata = clean_story(generated_text)
                story_for_metadata = story_for_metadata[:2500]

                title_guidelines_mature = input_data.get("title_guidelines_mature") or metadata.get("title_guidelines_mature") or {}
                title_guidelines_text = ""
                try:
                    if isinstance(title_guidelines_mature, dict) and title_guidelines_mature.get("style"):
                        ex = title_guidelines_mature.get("examples") if isinstance(title_guidelines_mature.get("examples"), list) else []
                        rules = title_guidelines_mature.get("rules") if isinstance(title_guidelines_mature.get("rules"), list) else []
                        title_guidelines_text = (
                            "\nTitle guidelines (mature):"
                            f"\n- Style: {title_guidelines_mature.get('style')}"
                            + (f"\n- Examples: {'; '.join([str(x) for x in ex[:6]])}" if ex else "")
                            + (f"\n- Rules: {' '.join([str(x) for x in rules])}" if rules else "")
                        )
                except Exception:
                    title_guidelines_text = ""

                # Step 8a: Title + preview + tags (JSON)
                allowed_themes = [
                    "betrayal", "redemption", "trust", "duty-vs-desire", "power-and-corruption",
                    "forbidden-knowledge", "mystery", "discovery", "legacy", "loss-and-grief"
                ]
                allowed_themes_text = f"\nAllowed canonical themes (prefer these ids): {', '.join(allowed_themes)}"
                available_tones = ["sensual", "noir", "tension", "passion", "mystery", "intimate", "forbidden"]
                tones_text = f"Available primary tones: {', '.join(available_tones)}"

                title_preview_system = """You are a creative storyteller and editor specializing in abstract, sophisticated narratives.

NOVELTY REQUIREMENTS (CRITICAL):
- The title must be distinct and story-specific; avoid generic templates.
- Do NOT use cliché template titles like "Whispers in the ___" / "___ in the Shadows" / "Embers in Twilight".
- Do NOT start with or include these overused noir title words/starts: "whispers", "shadows", "embers", "twilight", "midnight", "secrets", "desire".
- Prefer concrete nouns drawn from the story’s unique details (setting, artifact, motif, conflict).
- The title MUST contain at least one uncommon story-specific noun (not just mood words).
- Avoid repeating the same key noun twice.

Read the story below and generate:
1. A short, engaging title (2–6 words) that captures the story's essence.
   - Use abstract, evocative language
   - Reflect the sophisticated, mature tone
   - Avoid explicit sexual references
   - Make it memorable and suggestive

2. A 2–3 sentence preview (180–240 characters) that entices the reader.
   - Keep it spoiler-free
   - Use abstract, suggestive language
   - Focus on mood, atmosphere, and emotional tension
   - Avoid explicit sexual content

3. Tags for search and indexing:
   - themes: 2–4 canonical theme ids from the allowed list; if none fit, place the suggestion in "themeFreeform"
   - tones: 1–3 tonal tags (lowercase, concise)
   - settings: 1–3 location/setting tags (lowercase, concrete nouns)
   - artifacts: up to 2 key items (lowercase, concrete nouns)
   - conflicts: up to 2 conflicts/stakes (lowercase)
   - motifs: up to 2 motifs/imagery words (lowercase, concise)

Output ONLY valid JSON (no markdown, no code blocks):
{
  "title": "...",
  "preview": "...",
  "tags": {
    "themes": ["..."],
    "themeFreeform": ["..."],
    "tones": ["..."],
    "settings": ["..."],
    "artifacts": ["..."],
    "conflicts": ["..."],
    "motifs": ["..."]
  }
}"""

                title_preview_user = f"""Genre: {genre}{title_guidelines_text}
{tones_text}{allowed_themes_text}
Story:
{story_for_metadata}

Generate a non-explicit, abstract title, preview, and tags for this {genre} story."""

                title_preview_text, _t8a = _generate_content(
                    [{"role": "system", "content": title_preview_system}, {"role": "user", "content": title_preview_user}],
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature=0.7,
                    max_tokens=step8a_max_tokens,
                    device=device
                )
                title_preview_json = _parse_json_from_text(title_preview_text) or {}
                title = title_preview_json.get("title", "Untitled Story")
                preview = title_preview_json.get("preview", "")
                tags_raw = title_preview_json.get("tags", {}) if isinstance(title_preview_json.get("tags", {}), dict) else {}

                tags = None
                if tags_raw:
                    try:
                        tags = {
                            "themes": tags_raw.get("themes", []),
                            "themeFreeform": tags_raw.get("themeFreeform", []),
                            "tones": [t.lower() for t in tags_raw.get("tones", [])],
                            "settings": [s.lower() for s in tags_raw.get("settings", [])],
                            "artifacts": [a.lower() for a in tags_raw.get("artifacts", [])],
                            "conflicts": [c.lower() for c in tags_raw.get("conflicts", [])],
                            "motifs": [m.lower() for m in tags_raw.get("motifs", [])]
                        }
                        tags = {k: v for k, v in tags.items() if v}
                    except Exception:
                        tags = None

                # Step 8b: Cover prompt (text only)
                cover_system = """You create detailed image prompts for abstract, erotic noir-style book covers.

Generate a cover image prompt (25–60 words) in an abstract erotic noir style. The prompt should be:
- Non-explicit and suggestive rather than graphic
- Focused on mood, atmosphere, and sensual composition
- Using abstract visual elements
- Vertical portrait composition suitable for 1024x1536 output
- Include the book title as tasteful, readable cover typography that matches the art style

ANTI-SIMILARITY (CRITICAL):
- Avoid repeating a generic “silhouette + candle + wine” composition unless the story explicitly centers those objects.
- The prompt MUST include at least 2 story-specific visual details (a location feature + a concrete prop/motif) from the provided tags/context.
- Vary composition: specify one of (close-up hands, mid-shot embrace, distant corridor/room, mirror reflection, window backlight).
- Specify a distinct color palette (2–3 colors).
- Keep non-explicit; no visible sex acts or nudity.

Output ONLY the cover prompt text (25–60 words). No labels, no JSON, just the prompt."""

                cover_user = f"""Title: {title}
Genre: {genre}

Story context (first 1000 chars for mood reference):
{story_for_metadata[:1000]}

Story tags (use these to make the cover unique and story-specific):
{json.dumps(tags or {}, ensure_ascii=False)}

Create an abstract erotic noir cover prompt incorporating the required visual elements. Focus on moody, suggestive, non-explicit imagery. Integrate the exact title text as cover typography."""

                cover_prompt_text, _t8b = _generate_content(
                    [{"role": "system", "content": cover_system}, {"role": "user", "content": cover_user}],
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature=0.7,
                    max_tokens=step8b_max_tokens,
                    device=device
                )
                coverPrompt = (cover_prompt_text or "").strip()

                # Store metadata for Firestore and callback payload
                metadata["generated_title"] = title
                metadata["generated_preview"] = preview
                metadata["generated_coverPrompt"] = coverPrompt
                metadata["generated_tags"] = tags

                metadata["title"] = title
                metadata["preview"] = preview
                metadata["tags"] = tags
                metadata["cover_prompt"] = coverPrompt
                metadata["coverPrompt"] = coverPrompt
            else:
                metadata["generated_title"] = "Untitled Story"
                metadata["generated_preview"] = ""
                metadata["generated_coverPrompt"] = ""
                metadata["generated_tags"] = None
                metadata["title"] = "Untitled Story"
                metadata["preview"] = ""
                metadata["tags"] = None
                metadata["cover_prompt"] = ""
                metadata["coverPrompt"] = ""

        # TWO-STEP WORKFLOW: Generate outline first, then story (legacy)
        elif needs_two_step and outline_messages and story_messages:
            logger.info("=" * 80)
            logger.info("📝 STEP 1: Generating outline...")
            logger.info("=" * 80)
            
            # Step 1: Generate outline
            # Token analysis: ~475-725 tokens needed, but can be longer with detailed beats, 5000 allows for expansion
            outline_text, outline_time = _generate_content(
                outline_messages,
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens=outline_max_tokens,  # Configurable: default 5000 for detailed beat creation
                device=device
            )
            
            logger.info(f"✅ Generated outline ({len(outline_text)} chars) in {outline_time:.2f}s")
            logger.info(f"📄 Outline preview: {outline_text[:200]}...")
            
            # Inject outline into story messages
            logger.info("=" * 80)
            logger.info("📖 STEP 2: Generating story from outline...")
            logger.info("=" * 80)
            
            # Find user message in story_messages and inject outline
            # Replace "STORY STRUCTURE:" fallback section if present, otherwise prepend outline
            story_messages_with_outline = []
            for msg in story_messages:
                if msg.get("role") == "user":
                    original_content = msg.get("content", "")
                    msg_copy = msg.copy()
                    
                    # Check if content has "STORY STRUCTURE:" fallback section
                    # If so, replace it with the actual outline
                    if "STORY STRUCTURE:" in original_content:
                        # Replace the fallback section with actual outline
                        import re
                        # Match "STORY STRUCTURE:" and everything until the next major section
                        pattern = r'STORY STRUCTURE:.*?(?=\n\nSTORY REQUEST:|\n\nSTRICT REQUIREMENTS:|\Z)'
                        replacement = f"STORY OUTLINE:\n{outline_text}\n\nFollow this outline exactly - each scene matches one outline step."
                        updated_content = re.sub(pattern, replacement, original_content, flags=re.DOTALL)
                        msg_copy["content"] = updated_content
                        logger.info("✅ Replaced 'STORY STRUCTURE:' fallback with actual outline")
                    else:
                        # No fallback found, prepend outline
                        msg_copy["content"] = f"STORY OUTLINE:\n{outline_text}\n\nFollow this outline exactly - each scene matches one outline step.\n\n{original_content}"
                        logger.info("✅ Prepended outline to user message (no fallback found)")
                    
                    story_messages_with_outline.append(msg_copy)
                else:
                    story_messages_with_outline.append(msg)
            
            # Step 2: Generate story
            # Token analysis: ~3,025-3,425 tokens needed, 5000 is safer
            generated_text, story_time = _generate_content(
                story_messages_with_outline,
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens=5000,  # Story: system ~150-175 + user ~375-750 + output ~2,500 = ~3,025-3,425 total
                device=device
            )
            
            generation_time = outline_time + story_time
            logger.info(f"✅ Generated story ({len(generated_text)} chars) in {story_time:.2f}s")
            logger.info(f"⏱️ Total generation time: {generation_time:.2f}s")
            
            # Step 3: Expand story if < 8500 characters (synchronous)
            if len(generated_text) < 8500:
                logger.info("=" * 80)
                logger.info(f"📈 STEP 3: Expanding story (story is {len(generated_text)} chars, < 8500)...")
                logger.info("=" * 80)
                
                # Build expansion prompt dynamically
                expansion_system_prompt = f"""You're an award winning editor. You rewrite stories while preserving structure, pacing, and narrative balance.

You never over-expand any single section.

You distribute additional detail evenly across the entire story.

You maintain all scene breaks exactly as given.

CRITICAL: Preserve the beat labels ("Beat 1:", "Beat 2:", etc.) from the story below.

CRITICAL: Preserve the \\n\\n⁂\\n\\n separators between beats."""
                
                expansion_user_prompt = f"""Expand the following story by adding:
	•	more natural dialogue
	•	richer sensory descriptions
	•	deeper emotional reactions
	•	more specific physical detail
	•	smoother transitions

BUT:
	•	Keep the original plot and sequence exactly the same
	•	Keep all scene breaks (⁂) exactly where they are
	•	Keep all beat labels ("Beat 1:", "Beat 2:", etc.) exactly as they are
	•	Add only 30–40% more text overall, distributed evenly
	•	Do not expand only the first scenes
	•	Do not change POV, tense, or character roles
	•	Do not add new plot events

Here is the story to expand:

{generated_text}"""
                
                expansion_messages = [
                    {"role": "system", "content": expansion_system_prompt},
                    {"role": "user", "content": expansion_user_prompt}
                ]
                
                # Step 3: Expand story
                expanded_text, expansion_time = _generate_content(
                    expansion_messages,
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature,
                    max_tokens=expansion_max_tokens,
                    device=device
                )
                
                generated_text = expanded_text  # Use expanded version
                generation_time = outline_time + story_time + expansion_time
                logger.info(f"✅ Expanded story ({len(generated_text)} chars) in {expansion_time:.2f}s")
                logger.info(f"⏱️ Total generation time: {generation_time:.2f}s")
            else:
                logger.info(f"⏭️ Skipping Step 3 expansion (story is {len(generated_text)} chars, >= 8500)")
            
            # Step 4: Finetune story - rewrite duplicated dialogue and descriptions
            logger.info("=" * 80)
            logger.info("✨ STEP 4: Finetuning story (rewriting duplicated dialogue and descriptions)...")
            logger.info("=" * 80)
            
            # Build finetune prompt dynamically
            finetune_system_prompt = f"""You're an expert editor specializing in refining narrative prose. Your task is to improve story quality by rewriting duplicated, flat or repetitive dialogue and descriptions.

CRITICAL RULES:
- DO NOT change or remove any content
- DO NOT change the plot, sequence, or story structure
- DO NOT change character names, actions, or events
- DO NOT add new content or remove existing content
- Preserve all beat labels ("Beat 1:", "Beat 2:", etc.) exactly as they are
- Preserve all \\n\\n⁂\\n\\n separators between beats exactly as they are
- Only rewrite duplicated dialogue and descriptions to make them more varied and engaging
- Maintain the same meaning and context when rewriting"""
            
            finetune_user_prompt = f"""Rewrite the following story to improve quality by nuancing dialogue and descriptions.

Your task:
- Identify repeated phrases, dialogue patterns, or descriptions that appear multiple times
- Rewrite them to be more varied while keeping the same meaning
- Make dialogue more natural and less repetitive
- Vary descriptive language to avoid repetition
- Include dirty talk (dialogue) during sexual acts:
  - Use explicit sexual dialogue and dirty talk between characters
  - Include sounds like "ahh...", "ohh...", "yes...", "fuck...", "harder...", etc.
  - Mix dialogue with physical descriptions and sounds

CRITICAL CONSTRAINTS:
- Keep all scene breaks (⁂) exactly where they are
- Keep all beat labels ("Beat 1:", "Beat 2:", etc.) exactly as they are
- Do NOT change POV, tense, or character roles

Here is the story to finetune:

{generated_text}"""
            
            finetune_messages = [
                {"role": "system", "content": finetune_system_prompt},
                {"role": "user", "content": finetune_user_prompt}
            ]
            
            # Step 4: Finetune story
            finetuned_text, finetune_time = _generate_content(
                finetune_messages,
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens=finetune_max_tokens,
                device=device
            )
            
            generated_text = finetuned_text  # Use finetuned version
            generation_time = generation_time + finetune_time
            logger.info(f"✅ Finetuned story ({len(generated_text)} chars) in {finetune_time:.2f}s")
            logger.info(f"⏱️ Total generation time: {generation_time:.2f}s")
            
            # Step 5: Generate title, preview, tags, and cover prompt (for erotic-like stories)
            # IMPORTANT: This metadata must be propagated into BOTH:
            # - Firestore save metadata (used by _save_story_to_firestore)
            # - callback payload metadata (used by the Next.js app callback to enqueue cover image etc.)
            erotic_like = False
            try:
                genre_key = (genre or "").strip().lower()
                erotic_like = genre_key in ['erotic', 'advanced erotic', 'hardcore erotic', 'qwen3', 'qwen']
            except Exception:
                erotic_like = False

            if erotic_like:
                logger.info("=" * 80)
                logger.info("📝 STEP 5: Generating title, preview, tags, and cover prompt...")
                logger.info("=" * 80)
                
                # Get allowed themes for the genre
                # For now, we'll use a generic list - in production, this should come from the TypeScript catalog
                allowed_themes = [
                    "betrayal", "redemption", "trust", "duty-vs-desire", "power-and-corruption",
                    "forbidden-knowledge", "mystery", "discovery", "legacy", "loss-and-grief"
                ]
                allowed_themes_text = f"\nAllowed canonical themes (prefer these ids): {', '.join(allowed_themes)}"
                
                # Get available tones
                available_tones = ["sensual", "noir", "tension", "passion", "mystery", "intimate", "forbidden"]
                tones_text = f"Available primary tones: {', '.join(available_tones)}"
                
                # Build metadata generation prompt
                metadata_system_prompt = """You are a creative storyteller and editor specializing in abstract, sophisticated narratives.

Read the story below and generate:
1. A short, engaging title (2–6 words) that captures the story's essence.
   - Use abstract, evocative language
   - Reflect the sophisticated, mature tone
   - Avoid explicit sexual references
   - Make it memorable and suggestive

2. A 2–3 sentence preview (180–240 characters) that entices the reader.
   - Keep it spoiler-free
   - Use abstract, suggestive language
   - Focus on mood, atmosphere, and emotional tension
   - Avoid explicit sexual content
   - Emphasize the sophisticated, noir-like atmosphere

3. Tags for search and indexing:
   - themes: 2–4 canonical theme ids from the allowed list; if none fit, place the suggestion in "themeFreeform"
   - tones: 1–3 tonal tags (lowercase, concise) - e.g., "sensual", "noir", "tension", "passion", "mystery"
   - settings: 1–3 location/setting tags (lowercase, concrete nouns) - e.g., "hotel", "penthouse", "beach", "city"
   - artifacts: up to 2 key items (lowercase, concrete nouns) - e.g., "wine", "candle", "letter"
   - conflicts: up to 2 conflicts/stakes (lowercase) - e.g., "forbidden attraction", "power dynamic", "secret affair"
   - motifs: up to 2 motifs/imagery words (lowercase, concise) - e.g., "shadows", "silhouette", "tension"

4. A cover image prompt (25–60 words) in an abstract erotic noir style.
   - Non-explicit and suggestive rather than graphic
   - Focused on mood, atmosphere, and sensual composition
   - Using abstract visual elements: silhouettes, shadows of intertwined hands, dimly lit room with wine glasses and candlelight, soft curves of fabric, smoke, or shapes implying intimacy, backlit figure, moody lighting

Output ONLY valid JSON (no markdown, no code blocks):
{
  "title": "...",
  "preview": "...",
  "tags": {
    "themes": ["..."],
    "themeFreeform": ["..."],
    "tones": ["..."],
    "settings": ["..."],
    "artifacts": ["..."],
    "conflicts": ["..."],
    "motifs": ["..."]
  },
  "coverPrompt": "..."
}"""
                
                # Use cleaned story text for metadata generation (avoid Beat labels / separators leaking into title/preview)
                story_for_metadata = clean_story(generated_text)

                metadata_user_prompt = f"""Genre: {genre}
{tones_text}{allowed_themes_text}
Story:
{story_for_metadata}

Generate a non-explicit, abstract title, preview, tags, and cover prompt for this {genre} story."""
                
                metadata_messages = [
                    {"role": "system", "content": metadata_system_prompt},
                    {"role": "user", "content": metadata_user_prompt}
                ]
                
                # Step 5: Generate metadata
                metadata_response, metadata_time = _generate_content(
                    metadata_messages,
                    model_or_engine,
                    tokenizer,
                    use_vllm,
                    temperature=0.7,
                    max_tokens=1500,  # Increased for tags and cover prompt
                    device=device
                )
                
                generation_time = generation_time + metadata_time
                logger.info(f"✅ Generated metadata in {metadata_time:.2f}s")
                logger.info(f"⏱️ Total generation time: {generation_time:.2f}s")
                
                # Parse metadata JSON
                title = "Untitled Story"
                preview = ""
                coverPrompt = ""
                tags = None
                
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', metadata_response)
                    if json_match:
                        metadata_json = json.loads(json_match.group(0))
                        title = metadata_json.get("title", "Untitled Story")
                        preview = metadata_json.get("preview", "")
                        coverPrompt = metadata_json.get("coverPrompt", "")
                        tags_raw = metadata_json.get("tags", {})
                        
                        # Normalize tags (convert to proper format)
                        if tags_raw:
                            tags = {
                                "themes": tags_raw.get("themes", []),
                                "themeFreeform": tags_raw.get("themeFreeform", []),
                                "tones": [t.lower() for t in tags_raw.get("tones", [])],
                                "settings": [s.lower() for s in tags_raw.get("settings", [])],
                                "artifacts": [a.lower() for a in tags_raw.get("artifacts", [])],
                                "conflicts": [c.lower() for c in tags_raw.get("conflicts", [])],
                                "motifs": [m.lower() for m in tags_raw.get("motifs", [])]
                            }
                            # Remove empty arrays
                            tags = {k: v for k, v in tags.items() if v}
                        
                        logger.info(f"✅ Parsed metadata: title={title[:50]}..., preview={len(preview)} chars, coverPrompt={len(coverPrompt)} chars, tags={len(tags) if tags else 0} categories")
                    else:
                        logger.warning("⚠️ No JSON found in metadata response, using defaults")
                except Exception as parse_error:
                    logger.warning(f"⚠️ Failed to parse metadata JSON: {parse_error}")
                    logger.warning(f"⚠️ Raw metadata response: {metadata_response[:500]}")
                
                # Store metadata for later use in _save_story_to_firestore AND for callback payload.
                # _save_story_to_firestore currently reads generated_* keys.
                metadata["generated_title"] = title
                metadata["generated_preview"] = preview
                metadata["generated_coverPrompt"] = coverPrompt
                metadata["generated_tags"] = tags

                # Also include "app-facing" keys so the Next.js callback can consume them directly.
                # (The callback code looks for title/preview/cover_prompt/coverPrompt + tags.)
                metadata["title"] = title
                metadata["preview"] = preview
                metadata["tags"] = tags
                metadata["cover_prompt"] = coverPrompt
                metadata["coverPrompt"] = coverPrompt
            else:
                # For non-erotic stories, set defaults
                metadata["generated_title"] = "Untitled Story"
                metadata["generated_preview"] = ""
                metadata["generated_coverPrompt"] = ""
                metadata["generated_tags"] = None
                metadata["title"] = "Untitled Story"
                metadata["preview"] = ""
                metadata["tags"] = None
                metadata["cover_prompt"] = ""
                metadata["coverPrompt"] = ""
        
        # SINGLE-STEP WORKFLOW: Generate story directly
        else:
            # Format messages for Qwen model
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Generate using vLLM or transformers/AutoAWQ
            logger.info("🚀 Generating story content...")
            generated_text, generation_time = _generate_content(
                formatted_messages,
                model_or_engine,
                tokenizer,
                use_vllm,
                temperature,
                max_tokens,
                device
            )
            
            # For single-step workflow, metadata will be generated by OpenAI callback (if needed)
            metadata["generated_title"] = "Untitled Story"
            metadata["generated_preview"] = ""
            metadata["generated_coverPrompt"] = ""
            metadata["generated_tags"] = None
        
        # Keep the structured (beat-labeled) text for extraction/truncation;
        # only clean at the very end to avoid breaking beat-aware truncation.
        structured_text = generated_text

        # Extract beats for logging/truncation (from structured text)
        beats = _extract_beats(structured_text)
        
        logger.info(f"✅ Generated content length: {len(generated_text)} characters")
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
        
        # Enforce character limit on the FINAL (cleaned) text, but truncate using structured beats when possible.
        # Legacy Qwen/Qwen3 used 8,500; multi-step-v2 uses a hard max of 11,500.
        max_chars = _HARD_MAX_CHARS if workflow_type == "multi-step-v2" else (8500 if genre and genre.lower() in ['qwen3', 'qwen'] else 12000)
        cleaned_preview = clean_story(structured_text)
        if len(cleaned_preview) > max_chars:
            logger.warning(f"⚠️ Story exceeds {max_chars} character limit ({len(cleaned_preview)} chars cleaned), truncating...")
            expected_beats = 12 if workflow_type == "multi-step-v2" else (10 if genre and genre.lower() in ['qwen3', 'qwen'] else 12)
            if len(beats) >= expected_beats:
                truncated_beats = beats[:expected_beats]
                rebuilt_structured = '\n\n⁂\n\n'.join(truncated_beats)
                # If still too long after cleaning, truncate last beat content
                if len(clean_story(rebuilt_structured)) > max_chars:
                    last = truncated_beats[-1]
                    # Coarse truncation target (structured), then we will re-clean after
                    coarse_target = max(200, int(len(last) * 0.85))
                    truncated_beats[-1] = last[:coarse_target].rsplit('.', 1)[0] + '.' if '.' in last[:coarse_target] else last[:coarse_target]
                    rebuilt_structured = '\n\n⁂\n\n'.join(truncated_beats)
                structured_text = rebuilt_structured
                logger.info(f"✅ Truncated while maintaining beat structure (cleaned length: {len(clean_story(structured_text))})")
            else:
                # Fallback: simple truncation on cleaned text
                cleaned_preview = cleaned_preview[:max_chars].rsplit('.', 1)[0] + '.' if '.' in cleaned_preview[:max_chars] else cleaned_preview[:max_chars]
                structured_text = cleaned_preview
                logger.warning(f"⚠️ Simple truncation applied (cleaned): {len(cleaned_preview)} characters")

        # Clean story content: remove beat labels, separators, and unwanted formatting
        generated_text = clean_story(structured_text)
        if workflow_type == "multi-step-v2" and len(generated_text) > _HARD_MAX_CHARS:
            generated_text = _safe_trim_to_max(generated_text, _HARD_MAX_CHARS)
        
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
            # Merge in any generated metadata fields from Step 5 (if present)
            # NOTE: _save_story_to_firestore expects generated_* keys.
            try:
                for k in ["generated_title", "generated_preview", "generated_coverPrompt", "generated_tags"]:
                    if k in metadata:
                        save_metadata[k] = metadata.get(k)
            except Exception as e:
                logger.warning(f"⚠️ Failed to merge generated metadata into save_metadata: {e}")
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
                # Include generated metadata in callback payload so the main app can skip OpenAI fallback.
                callback_metadata = {
                    "language": language,
                    "genre": genre,
                    "age_range": age_range,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                try:
                    # Prefer app-facing keys (title/preview/cover_prompt/tags) if available
                    for k in ["title", "preview", "cover_prompt", "coverPrompt", "tags"]:
                        if k in metadata:
                            callback_metadata[k] = metadata.get(k)
                    # Also include generated_* for completeness/debugging
                    for k in ["generated_title", "generated_preview", "generated_coverPrompt", "generated_tags"]:
                        if k in metadata:
                            callback_metadata[k] = metadata.get(k)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to build callback metadata: {e}")

                callback_sent = notify_success_callback(
                    callback_url=callback_url,
                    story_id=story_id or "unknown",
                    content=generated_text,
                    user_id=user_id,
                    metadata=callback_metadata
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
            input_metadata = input_data.get("metadata", {})
            callback_url = (
                input_data.get("callback_url") 
                or metadata.get("callback_url") 
                or input_metadata.get("callback_url")
            )
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

def clean_story(text: str) -> str:
    """
    Clean story content by removing beat labels, separators, and unwanted formatting.
    This is the standard production solution for story-cleanup pipelines.
    
    Removes:
    - "⁂" dividers with surrounding whitespace
    - Beat labels like "Beat 8:", "Beat '8:", etc.
    - Accidental escaped newlines like "\\n"
    - Excessive blank lines (collapses to max 2 consecutive newlines)
    """
    import re
    
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip classic expository/meta openings deterministically (first line/sentence)
    # We cut (do not rewrite) to avoid introducing new content.
    def _strip_expository_opening(t: str) -> str:
        try:
            s = (t or "").lstrip()
            low = s.lower()
            forbidden = (
                "the story starts",
                "the story takes place",
                "the story opens",
                "this story",
                "the scene opens",
                "the narrative",
                "the following story",
            )
            if low.startswith(forbidden):
                # Drop first sentence
                parts = s.split(".", 1)
                if len(parts) > 1:
                    return parts[1].lstrip()
                return ""
            return t
        except Exception:
            return t

    text = _strip_expository_opening(text)

    # Remove leading quotes caused by model formatting (per line)
    text = re.sub(r'(?m)^\s*["“”]\s*', '', text)

    # Remove leading numeric junk (e.g. "0: ", "1. ", "2) ")
    text = re.sub(r'(?m)^\s*\d+\s*[:.)]\s*', '', text)

    # Trim leading/trailing whitespace per line (fix paragraphs starting with a space)
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove common helper phrases that sometimes leak into prose
    # (we do not rely on prompts to prevent these)
    helper_patterns = [
        r'\bThey change positions\b',
        r'\bTransition to\b',
        r'\bThe tension peaks\b',
        r'\bAs the scene shifts\b',
        r'\bThis leads to\b',
        r'\bIn the next moment\b',
    ]
    for p in helper_patterns:
        text = re.sub(r'(?im)^\s*' + p + r'.*$', '', text)

    # Remove "⁂" dividers with surrounding whitespace
    text = re.sub(r'\s*[⁂]+\s*', '\n\n', text, flags=re.MULTILINE)
    
    # Remove Beat labels like: Beat '8:, Beat 8:, etc.
    text = re.sub(r"Beat\s*['\"]?\d+['\"]?:?", '', text, flags=re.IGNORECASE)
    
    # Remove accidental escaped newlines like "\n\n"
    text = text.replace('\\n', '\n')
    
    # Collapse too many blank lines (max 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Trim leading/trailing whitespace
    return text.strip()

if __name__ == '__main__':
    logger.info("🚀 LLM Handler starting...")
    logger.info("✅ LLM Handler ready")
    runpod.serverless.start({"handler": handler})


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
        
        # Extract title from content (first line)
        title_match = content.split('\n')[0].strip() if content else None
        title = title_match[:120] if title_match else "Untitled Story"
        
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
            "preview": "",  # Will be generated by worker if needed
            "coverPrompt": "",  # Will be generated by worker if needed
            "coverHook": "",
            "coverEssence": "",
            "ageRange": age_range,
            "language": language,
            "promptVersion": "12-beat@2025-01-09",
            "genre": [genre.lower()] if genre else [],
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "provider": "runpod",  # Mark as Runpod-generated
        }
        
        # For default stories, use generationStatus instead of status
        if is_default_story:
            story_data["generationStatus"] = "ready"
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
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save story to Firestore: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def _load_model():
    """Lazy load the model from network volume or HuggingFace using vLLM (primary) or AutoAWQ (fallback)."""
    global _vllm_engine, _model, _tokenizer, _device, _use_vllm
    
    # Return already loaded model
    if _vllm_engine is not None or _model is not None:
        if _use_vllm:
            return _vllm_engine, _tokenizer, _device
        else:
            return _model, _tokenizer, _device
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from pathlib import Path
        
        # Check for network volume path first
        model_path = os.getenv("MODEL_PATH", "/runpod-volume/models/Qwen2.5-32B-Instruct-AWQ")
        model_name = os.getenv("MODEL_NAME", "Qwen2.5-32B-Instruct-AWQ")  # Fallback model name
        
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
            
            # Try vLLM first (primary method for AWQ models)
            if VLLM_AVAILABLE and is_awq:
                try:
                    logger.info("🚀 Attempting to load model with vLLM (primary method)...")
                    logger.info(f"📦 Loading AWQ model from {model_path} using vLLM...")
                    
                    _vllm_engine = LLM(
                        model=model_path,
                        quantization="awq",
                        trust_remote_code=True,
                        tensor_parallel_size=1,  # Adjust based on GPU count
                        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
                    )
                    
                    # Load tokenizer separately for vLLM
                    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    _use_vllm = True
                    logger.info("✅ Model loaded successfully with vLLM")
                    return _vllm_engine, _tokenizer, _device
                    
                except Exception as vllm_error:
                    logger.warning(f"⚠️ vLLM loading failed: {vllm_error}")
                    logger.info("🔄 Falling back to AutoAWQ...")
                    # Continue to fallback
            
            # Fallback to AutoAWQ if vLLM failed or not available
            if is_awq:
                logger.info("🔧 Loading AWQ model with AutoAWQ (fallback)...")
                # Load tokenizer from local path
                logger.info(f"🔧 Loading tokenizer from {model_path}...")
                _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                try:
                    from awq import AutoAWQForCausalLM
                    logger.info("✅ Using AutoAWQForCausalLM.from_quantized() for AWQ model")
                    _model = AutoAWQForCausalLM.from_quantized(
                        model_path,
                        fuse_layers=True,
                        trust_remote_code=True,
                        device_map="auto" if _device == "cuda" else None,
                    )
                    _use_vllm = False
                    logger.info("✅ AWQ model loaded successfully with AutoAWQ")
                    return _model, _tokenizer, _device
                except ImportError as e:
                    logger.warning(f"⚠️ AutoAWQ not available ({e}), trying standard loading...")
                    # Fallback to standard loading
                    _model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                        device_map="auto" if _device == "cuda" else None,
                        trust_remote_code=True,
                    )
                    _use_vllm = False
                    logger.info("✅ Model loaded with standard transformers")
                    return _model, _tokenizer, _device
                except Exception as e:
                    logger.error(f"❌ Failed to load AWQ model: {e}")
                    raise
            else:
                # Standard model loading (non-AWQ)
                logger.info("🔧 Loading standard (non-AWQ) model...")
                _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # Try vLLM for standard models too
                if VLLM_AVAILABLE:
                    try:
                        logger.info("🚀 Attempting to load standard model with vLLM...")
                        _vllm_engine = LLM(
                            model=model_path,
                            trust_remote_code=True,
                            tensor_parallel_size=1,
                            gpu_memory_utilization=0.9,
                        )
                        _use_vllm = True
                        logger.info("✅ Standard model loaded successfully with vLLM")
                        return _vllm_engine, _tokenizer, _device
                    except Exception as e:
                        logger.warning(f"⚠️ vLLM loading failed: {e}, falling back to transformers...")
                
                # Fallback to transformers
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
            
            # Try vLLM first for HuggingFace models
            if VLLM_AVAILABLE:
                try:
                    logger.info("🚀 Attempting to load HuggingFace model with vLLM...")
                    _vllm_engine = LLM(
                        model=model_name,
                        quantization="awq" if "awq" in model_name.lower() else None,
                        trust_remote_code=True,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.9,
                    )
                    _use_vllm = True
                    logger.info("✅ HuggingFace model loaded successfully with vLLM")
                    return _vllm_engine, _tokenizer, _device
                except Exception as e:
                    logger.warning(f"⚠️ vLLM loading failed: {e}, falling back to transformers...")
            
            # Fallback to transformers
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
            "max_tokens": 4000,
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
        max_tokens = input_data.get("max_tokens", 4000)
        language = input_data.get("language")
        genre = input_data.get("genre")
        age_range = input_data.get("age_range")
        user_id = input_data.get("user_id") or metadata.get("user_id")
        story_id = input_data.get("story_id") or metadata.get("story_id")
        callback_url = input_data.get("callback_url") or metadata.get("callback_url")
        
        logger.info(f"📖 LLM generation request received")
        logger.info(f"📊 Story ID: {story_id}")
        logger.info(f"👤 User ID: {user_id}")
        logger.info(f"📝 Messages count: {len(messages)}")
        logger.info(f"🌡️ Temperature: {temperature}")
        logger.info(f"🔢 Max tokens: {max_tokens}")
        
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
            outputs = model_or_engine.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"✅ Generated content length: {len(generated_text)} characters (vLLM)")
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
            
            logger.info(f"✅ Generated content length: {len(generated_text)} characters (transformers/AutoAWQ)")
        
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


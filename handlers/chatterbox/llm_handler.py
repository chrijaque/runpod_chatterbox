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

# Configure logging
_VERBOSE_LOGS = os.getenv("VERBOSE_LOGS", "false").lower() == "true"
_LOG_LEVEL = logging.INFO if _VERBOSE_LOGS else logging.WARNING
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

"""LLM handler for RunPod runtime using Qwen 2.5 Instruct model."""

# Model initialization (lazy loading)
_model = None
_tokenizer = None
_device = None

def _load_model():
    """Lazy load the Qwen 2.5 Instruct model."""
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device
    
    try:
        logger.info("🔧 Loading Qwen 2.5 Instruct model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"📦 Loading model: {model_name} on {_device}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            device_map="auto" if _device == "cuda" else None,
        )
        
        if _device == "cpu":
            _model = _model.to(_device)
        
        logger.info(f"✅ Model loaded successfully on {_device}")
        return _model, _tokenizer, _device
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
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
        model, tokenizer, device = _load_model()
        
        # Import torch for no_grad context
        import torch
        
        # Format messages for Qwen 2.5 Instruct
        # Qwen 2.5 uses chat template format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
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
        logger.info("🤖 Generating story content...")
        with torch.no_grad():
            generated_ids = model.generate(
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
        
        logger.info(f"✅ Generated content length: {len(generated_text)} characters")
        
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


import requests
import os
import logging
from typing import Dict, Any, Optional
import time
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

class RunPodClient:
    """RunPod API client for voice cloning, TTS, and LLM operations"""
    
    def __init__(self, api_key: str, voice_endpoint_id: str, tts_endpoint_id: str, llm_endpoint_id: Optional[str] = None):
        logger.info("🔍 ===== RUNPOD CLIENT INITIALIZATION =====")
        logger.info(f"📞 API Key provided: {bool(api_key)}")
        logger.info(f"📞 API Key length: {len(api_key) if api_key else 0}")
        logger.info(f"📞 Voice Endpoint ID: {voice_endpoint_id}")
        logger.info(f"📞 TTS Endpoint ID: {tts_endpoint_id}")
        logger.info(f"📞 LLM Endpoint ID: {llm_endpoint_id}")
        
        self.api_key = api_key
        self.voice_endpoint_id = voice_endpoint_id  # Default ChatterboxTTS
        self.tts_endpoint_id = tts_endpoint_id      # Default ChatterboxTTS
        self.llm_endpoint_id = llm_endpoint_id      # Optional LLM endpoint
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Import settings to access endpoint IDs
        from ..config import settings
        self.vc_cb_endpoint_id = settings.VC_CB_ENDPOINT_ID
        self.tts_cb_endpoint_id = settings.TTS_CB_ENDPOINT_ID
        self.vc_zonos_endpoint_id = getattr(settings, "VC_ZONOS_ENDPOINT_ID", "") or ""
        self.tts_zonos_endpoint_id = getattr(settings, "TTS_ZONOS_ENDPOINT_ID", "") or ""
        self.llm_cb_endpoint_id = settings.LLM_CB_ENDPOINT_ID
        
        logger.info(f"📞 Base URL: {self.base_url}")
        logger.info(f"📞 Headers configured: {bool(self.headers)}")
        logger.info(f"📞 ChatterboxTTS VC Endpoint: {self.vc_cb_endpoint_id}")
        logger.info(f"📞 ChatterboxTTS TTS Endpoint: {self.tts_cb_endpoint_id}")
        if self.vc_zonos_endpoint_id:
            logger.info(f"📞 Zonos VC Endpoint: {self.vc_zonos_endpoint_id}")
        if self.tts_zonos_endpoint_id:
            logger.info(f"📞 Zonos TTS Endpoint: {self.tts_zonos_endpoint_id}")
        logger.info(f"📞 ChatterboxTTS LLM Endpoint: {self.llm_cb_endpoint_id}")
        logger.info("🔍 ===== END RUNPOD CLIENT INITIALIZATION =====")
    
    def create_voice_clone(self, name: str, audio_base64: str | None, audio_format: str = "wav", response_format: str = "base64", 
                          language: str = "en", is_kids_voice: bool = False, model_type: str = "chatterbox",
                          user_id: Optional[str] = None, audio_path: Optional[str] = None,
                          profile_filename: Optional[str] = None, sample_filename: Optional[str] = None,
                          output_basename: Optional[str] = None, voice_id: Optional[str] = None,
                          callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a voice clone using RunPod
        
        :param name: Voice name
        :param audio_base64: Base64 encoded audio data
        :param audio_format: Audio format (wav, mp3, etc.)
        :param response_format: Response format (base64, binary)
        :return: RunPod response
        """
        try:
            logger.info("🔍 ===== RUNPOD CLIENT INTERNAL DEBUG =====")
            logger.info(f"📞 Method called with parameters:")
            logger.info(f"   name: {name}")
            logger.info(f"   audio_base64 type: {type(audio_base64)}")
            logger.info(f"   audio_base64 length: {len(audio_base64) if audio_base64 else 0}")
            logger.info(f"   audio_format: {audio_format}")
            logger.info(f"   response_format: {response_format}")
            logger.info(f"   model_type: {model_type}")
            logger.info(f"   audio_path: {audio_path}")
            
            # Debug audio data being sent to RunPod
            logger.info(f"🔍 Audio data details being sent to RunPod:")
            logger.info(f"   - Has audio data: {bool(audio_base64)}")
            logger.info(f"   - Audio data length: {len(audio_base64) if audio_base64 else 0}")
            logger.info(f"   - Audio format: {audio_format}")
            logger.info(f"   - Audio data preview: {audio_base64[:200] + '...' if audio_base64 and len(audio_base64) > 200 else audio_base64}")
            logger.info(f"   - Audio data end: {audio_base64[-100:] if audio_base64 and len(audio_base64) > 100 else audio_base64}")
            
            # Validate audio data before sending to RunPod
            if not audio_path:
                if not audio_base64 or len(audio_base64) < 1000:
                    logger.error(f"❌ Invalid audio data being sent to RunPod:")
                    logger.error(f"   - Has audio data: {bool(audio_base64)}")
                    logger.error(f"   - Audio data length: {len(audio_base64) if audio_base64 else 0}")
                    logger.error(f"   - Minimum expected: 1000")
                    raise Exception("Invalid audio data - audio file too small or empty")
            
            # Route to correct endpoint based on model type
            mt = (model_type or "chatterbox").lower()
            if mt in ["zonos", "zyphra"]:
                if not self.vc_zonos_endpoint_id:
                    raise Exception("VC_ZONOS_ENDPOINT_ID not configured")
                endpoint_id = self.vc_zonos_endpoint_id
                logger.info(f"🎯 Routing to Zonos VC endpoint: {endpoint_id}")
            else:
                endpoint_id = self.vc_cb_endpoint_id
                logger.info(f"🎯 Routing to ChatterboxTTS VC endpoint: {endpoint_id}")
            
            url = f"{self.base_url}/{endpoint_id}/run"

            # Canonicalize callback_url host to www.minstraly.com (avoid 307)
            cb_url = callback_url
            if cb_url:
                try:
                    p = urlparse(cb_url)
                    scheme = p.scheme or 'https'
                    netloc = p.netloc
                    if netloc == 'minstraly.com':
                        netloc = 'www.minstraly.com'
                    cb_url = urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
                except Exception:
                    pass
            
            payload = {
                # Some handlers read top-level metadata
                "metadata": {
                    "user_id": user_id,
                    "voice_id": voice_id,
                    "language": language,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type,
                    "callback_url": cb_url,
                    # Naming hints for downstream handler
                    "profile_filename": profile_filename,
                    "sample_filename": sample_filename,
                    "output_basename": output_basename,
                    "voice_name": name,
                },
                "input": {
                    "name": name,
                    "audio_data": audio_base64 or "",
                    "audio_path": audio_path,
                    "audio_format": audio_format,
                    "responseFormat": response_format,
                    "language": language,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type,
                    # Duplicate critical identifiers at the top level for handlers that don't read nested metadata
                    "user_id": user_id,
                    "voice_id": voice_id,
                    # Strongly pass explicit filenames so handler uses them verbatim
                    "profile_filename": profile_filename,
                    "sample_filename": sample_filename,
                    "output_basename": output_basename,
                    "voice_name": name,
                    # Also include a nested metadata object for handlers that expect it
                    "metadata": {
                        "user_id": user_id,
                        "voice_id": voice_id,
                        "language": language,
                        "is_kids_voice": is_kids_voice,
                        "model_type": model_type,
                        "callback_url": cb_url,
                        # New naming hints
                        "voice_name": name,
                        "profile_filename": profile_filename,
                        "sample_filename": sample_filename,
                        "output_basename": output_basename,
                    },
                }
            }
            
            logger.info(f"🚀 Creating voice clone for: {name}")
            logger.info(f"📡 Calling RunPod API: {url}")
            logger.info(f"📡 Endpoint ID: {endpoint_id}")
            logger.info(f"📡 Payload audio data length: {len(payload['input']['audio_data']) if payload['input']['audio_data'] else 0}")
            logger.info(f"📡 Headers: {self.headers}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"📡 Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod voice clone API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod API error: {response.text}")
            
            result = response.json()
            logger.info(f"✅ Voice clone job submitted successfully")
            logger.info(f"✅ Job ID: {result.get('id')}")
            logger.info(f"✅ Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Return quickly with job info; Firestore will be updated asynchronously by the worker
            if result.get('id'):
                return {"status": "IN_QUEUE", "id": result.get('id')}
            else:
                logger.error("❌ No job ID in response")
                raise Exception("No job ID in response")
            
        except Exception as e:
            logger.error(f"❌ Error creating voice clone: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
    def generate_tts(self, voice_id: str, text: str, profile_base64: str, response_format: str = "base64") -> Dict[str, Any]:
        """
        Generate TTS using RunPod (legacy method for voice clone samples)
        
        :param voice_id: Voice ID
        :param text: Text to synthesize
        :param profile_base64: Base64 encoded voice profile
        :param response_format: Response format (base64, binary)
        :return: RunPod response
        """
        return self.generate_tts_with_context(
            voice_id=voice_id,
            text=text,
            profile_base64=profile_base64,
            response_format=response_format
        )

    def generate_tts_with_context(self, voice_id: str, text: str, profile_base64: str, response_format: str = "base64", 
                                 language: str = "en", story_type: str = "user", is_kids_voice: bool = False, 
                                 model_type: str = "chatterbox", user_id: Optional[str] = None, story_id: Optional[str] = None,
                                 profile_path: Optional[str] = None, callback_url: Optional[str] = None,
                                 story_name: Optional[str] = None, output_basename: Optional[str] = None,
                                 output_filename: Optional[str] = None, voice_name: Optional[str] = None,
                                 genre: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate TTS using RunPod with story context
        
        :param voice_id: Voice ID
        :param text: Text to synthesize
        :param profile_base64: Base64 encoded voice profile
        :param response_format: Response format (base64, binary)
        :param language: Language code (en, es, fr, etc.)
        :param story_type: Story type (user, app, sample)
        :param is_kids_voice: Whether this is a kids voice
        :return: RunPod response
        """
        try:
            # Route to correct endpoint based on model type
            mt = (model_type or "chatterbox").lower()
            if mt in ["zonos", "zyphra"]:
                if not self.tts_zonos_endpoint_id:
                    raise Exception("TTS_ZONOS_ENDPOINT_ID not configured")
                endpoint_id = self.tts_zonos_endpoint_id
                logger.info(f"🎯 Routing to Zonos TTS endpoint: {endpoint_id}")
            else:
                endpoint_id = self.tts_cb_endpoint_id
                logger.info(f"🎯 Routing to ChatterboxTTS TTS endpoint: {endpoint_id}")
            
            url = f"{self.base_url}/{endpoint_id}/run"

            # Canonicalize callback_url host to www.minstraly.com (avoid 307)
            cb_url = callback_url
            if cb_url:
                try:
                    p = urlparse(cb_url)
                    scheme = p.scheme or 'https'
                    netloc = p.netloc
                    if netloc == 'minstraly.com':
                        netloc = 'www.minstraly.com'
                    cb_url = urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
                except Exception:
                    pass
            
            payload = {
                # Some handlers read top-level metadata
                "metadata": {
                    "user_id": user_id,
                    "story_id": story_id,
                    "language": language,
                    "story_type": story_type,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type,
                    "callback_url": cb_url,
                    "story_name": story_name,
                    "output_basename": output_basename,
                    "output_filename": output_filename,
                    "voice_name": voice_name,
                    "genre": genre,  # Add genre for TTS parameter adjustment
                },
                "input": {
                    "voice_id": voice_id,
                    "text": text,
                    "profile_base64": profile_base64,
                    "profile_path": profile_path,
                    "responseFormat": response_format,
                    "language": language,
                    "story_type": story_type,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type,
                    # Duplicate identifiers at top level for handlers that don't look into metadata
                    "user_id": user_id,
                    "story_id": story_id,
                    "story_name": story_name,
                    "output_basename": output_basename,
                    "output_filename": output_filename,
                    "voice_name": voice_name,
                    "genre": genre,  # Add genre at input level too
                    # Nested metadata for handlers that expect it
                    "metadata": {
                        "user_id": user_id,
                        "story_id": story_id,
                        "language": language,
                        "story_type": story_type,
                        "is_kids_voice": is_kids_voice,
                        "model_type": model_type,
                        "callback_url": cb_url,
                        "story_name": story_name,
                        "output_basename": output_basename,
                        "output_filename": output_filename,
                        "voice_name": voice_name,
                        "genre": genre,  # Add genre to nested metadata
                    },
                }
            }
            
            logger.info(f"🎵 Generating TTS for voice: {voice_id}")
            logger.info(f"📝 Text length: {len(text)} characters")
            logger.info(f"🌍 Language: {language}")
            logger.info(f"📚 Story Type: {story_type}")
            logger.info(f"👶 Kids Voice: {is_kids_voice}")
            logger.info(f"📡 Calling RunPod TTS API: {url}")
            logger.info(f"📊 Payload keys: {list(payload.keys())}")
            logger.info(f"📊 Input keys: {list(payload['input'].keys())}")
            logger.info(f"🔑 Headers: {self.headers}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            logger.info(f"📡 RunPod TTS API Response Status: {response.status_code}")
            logger.info(f"📡 RunPod TTS API Response Headers: {dict(response.headers)}")
            logger.info(f"📡 RunPod TTS API Response Text: {response.text[:500]}...")
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod TTS API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod TTS API error: {response.text}")
            
            result = response.json()
            logger.info(f"📊 Response type: {type(result)}")
            logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            if isinstance(result, dict):
                logger.info(f"🔍 Response status: {result.get('status', 'No status')}")
                logger.info(f"🔍 Response message: {result.get('message', 'No message')}")
                logger.info(f"🔍 Response keys: {list(result.keys())}")
                
                # Check if the TTS handler returned an error status
                if result.get('status') == 'error':
                    error_message = result.get('error', result.get('message', 'Unknown error from TTS handler'))
                    logger.error(f"❌ TTS handler returned error: {error_message}")
                    # Return the error response as-is so the API can handle it properly
                    return result
                else:
                    logger.info(f"✅ TTS generated successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating TTS: {e}")
            raise
    
    def get_job_status(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
        """
        Get job status from RunPod
        
        :param endpoint_id: RunPod endpoint ID
        :param job_id: Job ID
        :return: Job status
        """
        try:
            url = f"{self.base_url}/{endpoint_id}/status/{job_id}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod status API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod status API error: {response.text}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Error getting job status: {e}")
            raise
    
    def cancel_job(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
        """
        Cancel a RunPod job
        
        :param endpoint_id: RunPod endpoint ID
        :param job_id: Job ID
        :return: Cancellation result
        """
        try:
            url = f"{self.base_url}/{endpoint_id}/cancel/{job_id}"
            
            response = requests.post(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod cancel API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod cancel API error: {response.text}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Error canceling job: {e}")
            raise
    
    def wait_for_job_completion(self, endpoint_id: str, job_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a RunPod job to complete
        
        :param endpoint_id: RunPod endpoint ID
        :param job_id: Job ID
        :param timeout: Timeout in seconds
        :param poll_interval: Poll interval in seconds
        :return: Final job result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = self.get_job_status(endpoint_id, job_id)
                
                if status.get("status") == "COMPLETED":
                    logger.info(f"✅ Job {job_id} completed successfully")
                    # Return the output from the completed job
                    output = status.get("output", {})
                    logger.info(f"📦 Job output: {output}")
                    logger.info(f"📦 Job output type: {type(output)}")
                    logger.info(f"📦 Job output keys: {list(output.keys()) if isinstance(output, dict) else 'Not a dict'}")
                    return output
                elif status.get("status") == "FAILED":
                    error_message = status.get('error', 'Unknown error')
                    logger.error(f"❌ Job {job_id} failed: {error_message}")
                    # Return the error message instead of raising an exception
                    return {
                        "status": "error",
                        "message": error_message,
                        "job_id": job_id
                    }
                elif status.get("status") == "CANCELLED":
                    logger.warning(f"⚠️ Job {job_id} was cancelled")
                    return {
                        "status": "error",
                        "message": "Job was cancelled",
                        "job_id": job_id
                    }
                
                logger.info(f"⏳ Job {job_id} status: {status.get('status')} - waiting {poll_interval}s...")
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"❌ Error polling job status: {e}")
                raise
        
        # Timeout reached
        logger.error(f"❌ Job {job_id} timed out after {timeout} seconds")
        return {
            "status": "error",
            "message": f"Job timed out after {timeout} seconds",
            "job_id": job_id
        }
    
    def generate_llm_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 6000,
        language: Optional[str] = None,
        genre: Optional[str] = None,
        age_range: Optional[str] = None,
        user_id: Optional[str] = None,
        story_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        workflow_type: Optional[str] = None,
        outline_messages: Optional[list] = None,
        story_messages: Optional[list] = None,
        outline_max_tokens: Optional[int] = None,
        expansion_max_tokens: Optional[int] = None,
        finetune_max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
        preformatted_beats: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate LLM completion using RunPod Qwen 2.5 instruct model
        
        :param messages: List of message dicts with 'role' and 'content'
        :param temperature: Sampling temperature
        :param max_tokens: Maximum tokens to generate
        :param language: Language code (optional)
        :param genre: Genre name (optional)
        :param age_range: Age range (optional)
        :param user_id: User ID (optional)
        :param story_id: Story ID (optional)
        :param callback_url: Callback URL for async completion (optional)
        :return: RunPod response with status and content/job_id
        """
        try:
            if not self.llm_cb_endpoint_id:
                raise Exception("LLM endpoint ID not configured")
            
            endpoint_id = self.llm_cb_endpoint_id
            url = f"{self.base_url}/{endpoint_id}/run"
            
            # Canonicalize callback_url host to www.minstraly.com (avoid 307)
            cb_url = callback_url
            logger.info(f"🔔 Callback URL received in RunPod client: {cb_url}")
            logger.info(f"🔔 Story ID: {story_id}, User ID: {user_id}")
            if cb_url:
                try:
                    p = urlparse(cb_url)
                    scheme = p.scheme or 'https'
                    netloc = p.netloc
                    if netloc == 'minstraly.com':
                        netloc = 'www.minstraly.com'
                    cb_url = urlunparse((scheme, netloc, p.path, p.params, p.query, p.fragment))
                    logger.info(f"🔔 Callback URL canonicalized: {cb_url}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to canonicalize callback URL: {e}")
            else:
                logger.warning("⚠️ No callback URL provided to RunPod client - will use default from API app")
            
            payload = {
                "metadata": {
                    "user_id": user_id,
                    "story_id": story_id,
                    "language": language,
                    "genre": genre,
                    "age_range": age_range,
                    "callback_url": cb_url,
                    "workflow_type": workflow_type,
                },
                "input": {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "language": language,
                    "genre": genre,
                    "age_range": age_range,
                    "user_id": user_id,
                    "story_id": story_id,
                    "workflow_type": workflow_type,
                    "outline_messages": outline_messages,
                    "story_messages": story_messages,
                    "outline_max_tokens": outline_max_tokens,
                    "expansion_max_tokens": expansion_max_tokens,
                    "finetune_max_tokens": finetune_max_tokens,
                    "mode": mode,
                    "preformatted_beats": preformatted_beats,
                    "metadata": {
                        "user_id": user_id,
                        "story_id": story_id,
                        "language": language,
                        "genre": genre,
                        "age_range": age_range,
                        "callback_url": cb_url,
                        "workflow_type": workflow_type,
                    },
                }
            }
            
            logger.info(f"🤖 Generating LLM completion for story: {story_id}")
            logger.info(f"📡 Calling RunPod LLM API: {url}")
            logger.info(f"📊 Messages count: {len(messages)}")
            logger.info(f"🌡️ Temperature: {temperature}")
            logger.info(f"🔢 Max tokens: {max_tokens}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            logger.info(f"📡 Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod LLM API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod LLM API error: {response.text}")
            
            result = response.json()
            logger.info(f"📊 Response type: {type(result)}")
            logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Check if handler returned content directly (synchronous) or queued job (async)
            if isinstance(result, dict):
                if result.get('status') == 'error':
                    error_message = result.get('error', result.get('message', 'Unknown error from LLM handler'))
                    logger.error(f"❌ LLM handler returned error: {error_message}")
                    return result
                elif result.get('status') == 'success' and 'content' in result:
                    logger.info(f"✅ LLM generation completed synchronously")
                    return result
                elif result.get('id'):
                    logger.info(f"⏳ LLM job queued with ID: {result.get('id')}")
                    return {"status": "IN_QUEUE", "id": result.get('id')}
                elif 'content' in result:
                    # Handler returned content directly without status
                    logger.info(f"✅ LLM generation completed (implicit success)")
                    return {"status": "success", "content": result.get('content'), "metadata": result.get('metadata')}
            
            logger.warning(f"⚠️ Unexpected response format: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM completion: {e}")
            raise
    
    def is_configured(self) -> bool:
        """Check if RunPod is properly configured"""
        return all([
            self.api_key,
            self.voice_endpoint_id,
            self.tts_endpoint_id
        ]) 
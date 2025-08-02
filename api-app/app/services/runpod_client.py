import requests
import os
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class RunPodClient:
    """RunPod API client for voice cloning and TTS operations"""
    
    def __init__(self, api_key: str, voice_endpoint_id: str, tts_endpoint_id: str):
        logger.info("🔍 ===== RUNPOD CLIENT INITIALIZATION =====")
        logger.info(f"📞 API Key provided: {bool(api_key)}")
        logger.info(f"📞 API Key length: {len(api_key) if api_key else 0}")
        logger.info(f"📞 Voice Endpoint ID: {voice_endpoint_id}")
        logger.info(f"📞 TTS Endpoint ID: {tts_endpoint_id}")
        
        self.api_key = api_key
        self.voice_endpoint_id = voice_endpoint_id  # Default ChatterboxTTS
        self.tts_endpoint_id = tts_endpoint_id      # Default ChatterboxTTS
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Import settings to access endpoint IDs
        from ..config import settings
        self.vc_cb_endpoint_id = settings.VC_CB_ENDPOINT_ID
        self.tts_cb_endpoint_id = settings.TTS_CB_ENDPOINT_ID
        self.vc_higgs_endpoint_id = settings.VC_Higgs_ENDPOINT_ID
        self.tts_higgs_endpoint_id = settings.TTS_Higgs_ENDPOINT_ID
        
        logger.info(f"📞 Base URL: {self.base_url}")
        logger.info(f"📞 Headers configured: {bool(self.headers)}")
        logger.info(f"📞 ChatterboxTTS VC Endpoint: {self.vc_cb_endpoint_id}")
        logger.info(f"📞 ChatterboxTTS TTS Endpoint: {self.tts_cb_endpoint_id}")
        logger.info(f"📞 Higgs Audio VC Endpoint: {self.vc_higgs_endpoint_id}")
        logger.info(f"📞 Higgs Audio TTS Endpoint: {self.tts_higgs_endpoint_id}")
        logger.info("🔍 ===== END RUNPOD CLIENT INITIALIZATION =====")
    
    def create_voice_clone(self, name: str, audio_base64: str, audio_format: str = "wav", response_format: str = "base64", 
                          language: str = "en", is_kids_voice: bool = False, model_type: str = "chatterbox") -> Dict[str, Any]:
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
            
            # Debug audio data being sent to RunPod
            logger.info(f"🔍 Audio data details being sent to RunPod:")
            logger.info(f"   - Has audio data: {bool(audio_base64)}")
            logger.info(f"   - Audio data length: {len(audio_base64) if audio_base64 else 0}")
            logger.info(f"   - Audio format: {audio_format}")
            logger.info(f"   - Audio data preview: {audio_base64[:200] + '...' if audio_base64 and len(audio_base64) > 200 else audio_base64}")
            logger.info(f"   - Audio data end: {audio_base64[-100:] if audio_base64 and len(audio_base64) > 100 else audio_base64}")
            
            # Validate audio data before sending to RunPod
            if not audio_base64 or len(audio_base64) < 1000:
                logger.error(f"❌ Invalid audio data being sent to RunPod:")
                logger.error(f"   - Has audio data: {bool(audio_base64)}")
                logger.error(f"   - Audio data length: {len(audio_base64) if audio_base64 else 0}")
                logger.error(f"   - Minimum expected: 1000")
                raise Exception("Invalid audio data - audio file too small or empty")
            
            # Route to correct endpoint based on model type
            if model_type.lower() in ["chatterbox", "chatterboxtts", "cb"]:
                endpoint_id = self.vc_cb_endpoint_id
                logger.info(f"🎯 Routing to ChatterboxTTS VC endpoint: {endpoint_id}")
            elif model_type.lower() in ["higgs", "higgs_audio", "higgsaudio"]:
                endpoint_id = self.vc_higgs_endpoint_id
                logger.info(f"🎯 Routing to Higgs Audio VC endpoint: {endpoint_id}")
            else:
                # Default to ChatterboxTTS
                endpoint_id = self.vc_cb_endpoint_id
                logger.info(f"🎯 Defaulting to ChatterboxTTS VC endpoint: {endpoint_id}")
            
            url = f"{self.base_url}/{endpoint_id}/run"
            
            payload = {
                "input": {
                    "name": name,
                    "audio_data": audio_base64,
                    "audio_format": audio_format,
                    "responseFormat": response_format,
                    "language": language,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type
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
            
            # Wait for job completion
            if result.get('id'):
                logger.info(f"⏳ Waiting for job completion: {result['id']}")
                final_result = self.wait_for_job_completion(endpoint_id, result['id'])
                logger.info(f"✅ Job completed with result: {final_result}")
                logger.info(f"✅ Final result type: {type(final_result)}")
                logger.info(f"✅ Final result keys: {list(final_result.keys()) if isinstance(final_result, dict) else 'Not a dict'}")
                return final_result
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
                                 model_type: str = "chatterbox") -> Dict[str, Any]:
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
            if model_type.lower() in ["chatterbox", "chatterboxtts", "cb"]:
                endpoint_id = self.tts_cb_endpoint_id
                logger.info(f"🎯 Routing to ChatterboxTTS TTS endpoint: {endpoint_id}")
            elif model_type.lower() in ["higgs", "higgs_audio", "higgsaudio"]:
                endpoint_id = self.tts_higgs_endpoint_id
                logger.info(f"🎯 Routing to Higgs Audio TTS endpoint: {endpoint_id}")
            else:
                # Default to ChatterboxTTS
                endpoint_id = self.tts_cb_endpoint_id
                logger.info(f"🎯 Defaulting to ChatterboxTTS TTS endpoint: {endpoint_id}")
            
            url = f"{self.base_url}/{endpoint_id}/run"
            
            payload = {
                "input": {
                    "voice_id": voice_id,
                    "text": text,
                    "profile_base64": profile_base64,
                    "responseFormat": response_format,
                    "language": language,
                    "story_type": story_type,
                    "is_kids_voice": is_kids_voice,
                    "model_type": model_type
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
            logger.info(f"✅ TTS generated successfully")
            logger.info(f"📊 Response type: {type(result)}")
            logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            if isinstance(result, dict):
                logger.info(f"🔍 Response status: {result.get('status', 'No status')}")
                logger.info(f"🔍 Response message: {result.get('message', 'No message')}")
                logger.info(f"🔍 Response keys: {list(result.keys())}")
            
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
    
    def is_configured(self) -> bool:
        """Check if RunPod is properly configured"""
        return all([
            self.api_key,
            self.voice_endpoint_id,
            self.tts_endpoint_id
        ]) 
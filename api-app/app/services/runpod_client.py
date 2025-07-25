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
        self.voice_endpoint_id = voice_endpoint_id
        self.tts_endpoint_id = tts_endpoint_id
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"📞 Base URL: {self.base_url}")
        logger.info(f"📞 Headers configured: {bool(self.headers)}")
        logger.info("🔍 ===== END RUNPOD CLIENT INITIALIZATION =====")
    
    def create_voice_clone(self, name: str, audio_base64: str, audio_format: str = "wav", response_format: str = "base64") -> Dict[str, Any]:
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
            
            url = f"{self.base_url}/{self.voice_endpoint_id}/run"
            
            payload = {
                "input": {
                    "name": name,
                    "audio_data": audio_base64,
                    "audio_format": audio_format,
                    "responseFormat": response_format
                }
            }
            
            logger.info(f"🚀 Creating voice clone for: {name}")
            logger.info(f"📡 Calling RunPod API: {url}")
            logger.info(f"📡 Endpoint ID: {self.voice_endpoint_id}")
            logger.info(f"📡 Headers: {self.headers}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"📡 Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod voice clone API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod API error: {response.text}")
            
            result = response.json()
            logger.info(f"✅ Voice clone created successfully")
            logger.info(f"✅ Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error creating voice clone: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
    
    def generate_tts(self, voice_id: str, text: str, profile_base64: str, response_format: str = "base64") -> Dict[str, Any]:
        """
        Generate TTS using RunPod
        
        :param voice_id: Voice ID
        :param text: Text to synthesize
        :param profile_base64: Base64 encoded voice profile
        :param response_format: Response format (base64, binary)
        :return: RunPod response
        """
        try:
            url = f"{self.base_url}/{self.tts_endpoint_id}/run"
            
            payload = {
                "input": {
                    "voice_id": voice_id,
                    "text": text,
                    "profile_base64": profile_base64,
                    "responseFormat": response_format
                }
            }
            
            logger.info(f"🎵 Generating TTS for voice: {voice_id}")
            logger.info(f"📝 Text length: {len(text)} characters")
            logger.info(f"📡 Calling RunPod TTS API: {url}")
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"❌ RunPod TTS API error: {response.status_code} - {response.text}")
                raise Exception(f"RunPod TTS API error: {response.text}")
            
            result = response.json()
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
                    return status
                elif status.get("status") == "FAILED":
                    logger.error(f"❌ Job {job_id} failed: {status.get('error', 'Unknown error')}")
                    raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
                elif status.get("status") == "CANCELLED":
                    logger.warning(f"⚠️ Job {job_id} was cancelled")
                    raise Exception("Job was cancelled")
                
                logger.info(f"⏳ Job {job_id} status: {status.get('status')} - waiting {poll_interval}s...")
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"❌ Error polling job status: {e}")
                raise
        
        raise Exception(f"Job {job_id} timed out after {timeout} seconds")
    
    def is_configured(self) -> bool:
        """Check if RunPod is properly configured"""
        return all([
            self.api_key,
            self.voice_endpoint_id,
            self.tts_endpoint_id
        ]) 
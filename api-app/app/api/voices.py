from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import os
from ..services.runpod_client import RunPodClient
from ..services.firebase import FirebaseService
from ..services.redis_queue import RedisQueueService
from ..models.schemas import VoiceCloneRequest, VoiceCloneResponse, VoiceInfo
from ..config import settings
from ..middleware.security import verify_hmac

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
runpod_client = RunPodClient(
    api_key=settings.RUNPOD_API_KEY,
    voice_endpoint_id=settings.VC_CB_ENDPOINT_ID,
    tts_endpoint_id=settings.TTS_CB_ENDPOINT_ID
)

# Initialize Firebase service for library display
firebase_service = None
redis_queue: RedisQueueService | None = None

def get_firebase_service():
    """Get or initialize Firebase service"""
    global firebase_service
    
    if firebase_service is not None:
        return firebase_service
    
    logger.info(f"🔍 Firebase config debug:")
    logger.info(f"   - FIREBASE_LOCAL_CREDS_FILE: {settings.FIREBASE_LOCAL_CREDS_FILE}")
    logger.info(f"   - FIREBASE_LOCAL_CREDS_FILE exists: {os.path.exists(settings.FIREBASE_LOCAL_CREDS_FILE) if settings.FIREBASE_LOCAL_CREDS_FILE else False}")
    logger.info(f"   - FIREBASE_LOCAL_CREDS_FILE absolute path: {os.path.abspath(settings.FIREBASE_LOCAL_CREDS_FILE) if settings.FIREBASE_LOCAL_CREDS_FILE else 'None'}")
    logger.info(f"   - FIREBASE_CREDENTIALS: {'SET' if settings.FIREBASE_CREDENTIALS else 'NOT SET'}")
    logger.info(f"   - FIREBASE_STORAGE_BUCKET: {settings.FIREBASE_STORAGE_BUCKET}")
    
    # Try to use local Firebase credentials file first
    if settings.FIREBASE_LOCAL_CREDS_FILE and os.path.exists(settings.FIREBASE_LOCAL_CREDS_FILE):
        try:
            logger.info(f"🔍 Reading Firebase credentials from: {settings.FIREBASE_LOCAL_CREDS_FILE}")
            with open(settings.FIREBASE_LOCAL_CREDS_FILE, 'r') as f:
                credentials_json = f.read()
            logger.info(f"🔍 Credentials file size: {len(credentials_json)} characters")
            logger.info(f"🔍 Credentials loaded successfully")
            
            firebase_service = FirebaseService(
                credentials_json=credentials_json,
                bucket_name=settings.get_firebase_bucket_name()
            )
            logger.info("✅ Firebase service initialized with local credentials file")
            return firebase_service
        except Exception as e:
            logger.warning(f"⚠️ Firebase service initialization failed with local file: {e}")
            firebase_service = None
    # Fallback to environment variable
    elif settings.FIREBASE_CREDENTIALS and settings.FIREBASE_STORAGE_BUCKET:
        try:
            firebase_service = FirebaseService(
                credentials_json=settings.FIREBASE_CREDENTIALS,
                bucket_name=settings.get_firebase_bucket_name()
            )
            logger.info("✅ Firebase service initialized with environment credentials")
            return firebase_service
        except Exception as e:
            logger.warning(f"⚠️ Firebase service initialization failed: {e}")
            firebase_service = None
    else:
        logger.warning("⚠️ Firebase credentials not available - voice library will be empty")
        return None
    
    return firebase_service

def get_queue_service() -> RedisQueueService | None:
    global redis_queue
    if redis_queue is not None:
        return redis_queue
    try:
        redis_queue = RedisQueueService()
        logger.info("✅ Redis queue initialized")
        return redis_queue
    except Exception as e:
        logger.warning(f"⚠️ Redis not configured: {e}")
        return None

@router.post("/clone", response_model=VoiceCloneResponse, dependencies=[Depends(verify_hmac)])
async def clone_voice(request: VoiceCloneRequest, job_id: str | None = None):
    """
    Clone a voice using uploaded audio.
    """
    try:
        logger.info(f"🎤 Voice clone request received for: {request.name}")
        logger.info(f"📊 Request details: language={request.language}, kids_voice={request.is_kids_voice}")
        logger.info(f"🎯 Model type: {request.model_type}")
        
        # Debug audio data received from frontend
        logger.info(f"🔍 Audio data details from frontend:")
        logger.info(f"   - Has audio data: {bool(request.audio_data)}")
        logger.info(f"   - Audio data length: {len(request.audio_data) if request.audio_data else 0}")
        logger.info(f"   - Audio format: {request.audio_format}")
        logger.info(f"   - Audio data preview: {request.audio_data[:200] + '...' if request.audio_data and len(request.audio_data) > 200 else request.audio_data}")
        logger.info(f"   - Audio data end: {request.audio_data[-100:] if request.audio_data and len(request.audio_data) > 100 else request.audio_data}")
        
        # Validate identifiers and audio input (either audio_data or audio_path)
        if not request.user_id:
            raise HTTPException(status_code=400, detail="Missing user_id")
        if not (request.audio_data or request.audio_path):
            raise HTTPException(status_code=400, detail="Either audio_data or audio_path must be provided")
        
        # Queue disabled: call RunPod directly (synchronous)

        # Fallback: Call RunPod synchronously when Redis is not configured
        # If audio_path provided, pass pointer through; the RunPod handler will download it.
        if request.audio_path and "/recorded/" not in request.audio_path:
            raise HTTPException(status_code=400, detail="audio_path must be under recorded/")

        result = runpod_client.create_voice_clone(
            name=request.name,
            audio_base64=request.audio_data,
            audio_format=request.audio_format,
            language=request.language,
            is_kids_voice=request.is_kids_voice,
            model_type=request.model_type,
            user_id=request.user_id,
            audio_path=request.audio_path,
            profile_filename=request.profile_filename or (f"{request.voice_id}.npy" if request.voice_id else None),
            sample_filename=request.sample_filename or (f"{request.voice_id}.mp3" if request.voice_id else None),
            output_basename=request.output_basename or request.voice_id,
            voice_id=request.voice_id,
        )
        
        logger.info(f"🔍 RunPod result type: {type(result)}")
        logger.info(f"🔍 RunPod result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        logger.info(f"🔍 RunPod result status: {result.get('status')}")
        
        if result.get("status") in ["IN_QUEUE", "IN_PROGRESS"] and result.get("id"):
            # Return quickly after dispatching job; UI will listen to Firestore updates
            return VoiceCloneResponse(status="success", job_id=result.get("id"))
        elif result.get("status") == "success":
            # Legacy path if the client still waits for completion
            return VoiceCloneResponse(status="success", job_id=result.get("id"))
        elif result.get("status") == "error":
            # Handle error response from RunPod
            error_message = result.get("error", result.get("message", "Unknown error occurred"))
            logger.error(f"❌ RunPod job failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
        else:
            error_message = result.get("error", result.get("message", "Unknown error occurred"))
            logger.error(f"❌ Voice clone failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
    except Exception as e:
        logger.error(f"❌ Voice clone error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {str(e)}")

@router.post("/callback")
async def voice_clone_callback(request: dict):
    """
    Handle voice clone callbacks from RunPod workers.
    This endpoint receives status updates from the voice cloning process.
    """
    try:
        logger.info(f"📥 Voice clone callback received: {request}")
        
        # Extract callback data
        status = request.get('status')
        user_id = request.get('user_id')
        voice_id = request.get('voice_id')
        error = request.get('error')
        message = request.get('message')  # Some workers send 'message' instead of 'error'
        
        # Handle both error field formats
        error_message = error or message
        
        logger.info(f"📋 Callback details:")
        logger.info(f"   - Status: {status}")
        logger.info(f"   - User ID: {user_id}")
        logger.info(f"   - Voice ID: {voice_id}")
        logger.info(f"   - Error: {error_message}")
        
        # Update Firestore voice profile document based on status
        if voice_id and user_id:
            try:
                firebase_service = get_firebase_service()
                if firebase_service:
                    from google.cloud.firestore import SERVER_TIMESTAMP
                    
                    doc_ref = firebase_service.db.collection("voice_profiles").document(voice_id)
                    
                    if status == "success":
                        # Update to ready status
                        doc_ref.update({
                            "status": "ready",
                            "updatedAt": SERVER_TIMESTAMP,
                        })
                        logger.info(f"✅ Updated voice profile {voice_id} to ready status")
                    elif status == "error" or error_message:
                        # Update to failed status with error message
                        doc_ref.update({
                            "status": "failed",
                            "error": error_message,
                            "updatedAt": SERVER_TIMESTAMP,
                        })
                        logger.info(f"❌ Updated voice profile {voice_id} to failed status: {error_message}")
                    else:
                        logger.warning(f"⚠️ Unknown callback status: {status}")
                else:
                    logger.warning("⚠️ Firebase service not available for callback update")
            except Exception as e:
                logger.error(f"❌ Failed to update Firestore voice profile: {e}")
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"❌ Error handling voice clone callback: {e}")
        return {"success": False, "error": str(e)}

@router.get("/")
async def list_voices():
    """
    List all available voices.
    """
    try:
        logger.info("📚 Listing voices from Firebase...")
        
        # Get Firebase service
        firebase_service = get_firebase_service()
        
        # Check if Firebase service is available
        if firebase_service is None:
            logger.warning("⚠️ Firebase service not available - returning empty voice list")
            return {
                "status": "success",
                "voices": [],
                "language": "en",
                "is_kids_voice": False,
                "total": 0
            }
        
        # Get voices from Firebase
        voices = firebase_service.list_voices_by_language("en", False)
        
        # Debug: Log the raw Firebase data
        logger.info(f"🔍 Raw Firebase data for first voice: {voices[0] if voices else 'No voices found'}")
        
        # Convert to VoiceInfo objects with frontend-expected field names
        voice_list = []
        for voice in voices:
            # Get the first sample and profile URLs from the Firebase data
            samples = voice.get("samples", [])
            profiles = voice.get("profiles", [])
            sample_url = samples[0] if samples else None
            profile_url = profiles[0] if profiles else None
            
            logger.info(f"🔍 Voice {voice.get('voice_id')}: samples={samples}, profiles={profiles}")
            logger.info(f"🔍 Extracted URLs: sample_url={sample_url}, profile_url={profile_url}")
            
            voice_info = VoiceInfo(
                voice_id=voice.get("voice_id", ""),
                name=voice.get("name", ""),
                sample_file=sample_url,  # This will be mapped to sample_url by frontend
                embedding_file=profile_url,  # This will be mapped to profile_url by frontend
                created_date=voice.get("created_date"),
                language=voice.get("language"),
                is_kids_voice=voice.get("is_kids_voice")
            )
            voice_list.append(voice_info)
        
        logger.info(f"✅ Found {len(voice_list)} voices")
        
        # Return the format expected by the frontend
        return {
            "status": "success",
            "voices": voice_list,
            "language": "en",
            "is_kids_voice": False,
            "total": len(voice_list)
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@router.get("/{voice_id}/profile")
async def get_voice_profile(voice_id: str, language: str = "en", is_kids_voice: bool = False):
    """
    Get voice profile as base64 from Firebase (proxy endpoint to avoid CORS)
    """
    try:
        logger.info(f"🔍 Getting voice profile for: {voice_id}")
        logger.info(f"🔍 Parameters: language={language}, is_kids_voice={is_kids_voice}")
        
        # Get Firebase service
        firebase_service = get_firebase_service()
        
        # Check if Firebase service is available
        if firebase_service is None:
            logger.warning("⚠️ Firebase service not available - cannot get voice profile")
            raise HTTPException(status_code=503, detail="Firebase service not available")
        
        # Get voice profile from Firebase
        profile_base64 = firebase_service.get_voice_profile_base64(voice_id, language, is_kids_voice)
        
        if profile_base64:
            logger.info(f"✅ Found voice profile for {voice_id} (length: {len(profile_base64)})")
            return {
                "status": "success",
                "profile_base64": profile_base64,
                "voice_id": voice_id
            }
        else:
            logger.warning(f"⚠️ Voice profile not found for {voice_id}")
            raise HTTPException(status_code=404, detail="Voice profile not found")
            
    except Exception as e:
        logger.error(f"❌ Error getting voice profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get voice profile: {str(e)}")

@router.get("/by-language/{language}")
async def list_voices_by_language(language: str, is_kids_voice: bool = False):
    """
    List voices by language and kids voice flag.
    """
    try:
        logger.info(f"📚 Listing voices for language: {language}, is_kids_voice: {is_kids_voice}")
        
        # Get Firebase service
        firebase_service = get_firebase_service()
        
        # Check if Firebase service is available
        if firebase_service is None:
            logger.warning("⚠️ Firebase service not available - returning empty voice list")
            return {
                "status": "success",
                "voices": [],
                "language": language,
                "is_kids_voice": is_kids_voice,
                "total": 0
            }
        
        # Get voices from Firebase
        voices = firebase_service.list_voices_by_language(language, is_kids_voice)
        
        # Debug: Log the raw voices data
        logger.info(f"🔍 Raw voices from Firebase: {len(voices)} voices")
        for i, voice in enumerate(voices):
            logger.info(f"🔍 Voice {i+1}: {voice.get('voice_id')} - samples: {len(voice.get('samples', []))}, profiles: {len(voice.get('profiles', []))}")
        
        # Convert to VoiceInfo objects with frontend-expected field names
        voice_list = []
        for voice in voices:
            # Get the first sample and profile URLs from the Firebase data
            sample_url = voice.get("samples", [])[0] if voice.get("samples") else None
            profile_url = voice.get("profiles", [])[0] if voice.get("profiles") else None
            
            logger.info(f"🔍 Processing voice {voice.get('voice_id')}: sample_url={sample_url}, profile_url={profile_url}")
            
            voice_info = VoiceInfo(
                voice_id=voice.get("voice_id", ""),
                name=voice.get("name", ""),
                sample_file=sample_url,  # This will be mapped to sample_url by frontend
                embedding_file=profile_url,  # This will be mapped to profile_url by frontend
                created_date=voice.get("created_date"),
                language=voice.get("language"),
                is_kids_voice=voice.get("is_kids_voice")
            )
            voice_list.append(voice_info)
        
        logger.info(f"✅ Found {len(voice_list)} voices for language {language}")
        
        # Return the format expected by the frontend
        return {
            "status": "success",
            "voices": voice_list,
            "language": language,
            "is_kids_voice": is_kids_voice,
            "total": len(voice_list)
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing voices for language {language}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}") 

@router.get("/test-firebase-files")
async def test_firebase_files():
    """
    Test endpoint to list all files in Firebase
    """
    try:
        logger.info("🔍 Testing Firebase file listing...")
        
        # Get Firebase service
        firebase_service = get_firebase_service()
        
        # Check if Firebase service is available
        if firebase_service is None:
            logger.warning("⚠️ Firebase service not available - cannot test files")
            raise HTTPException(status_code=503, detail="Firebase service not available")
        
        # Test different prefixes
        prefixes = [
            "audio/voices/en/",
            "audio/voices/en/samples/",
            "audio/voices/en/profiles/",
            "audio/voices/en/recorded/"
        ]
        
        results = {}
        for prefix in prefixes:
            files = firebase_service.test_list_all_files(prefix)
            results[prefix] = files
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Firebase files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test Firebase files: {str(e)}") 
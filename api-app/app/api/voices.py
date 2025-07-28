from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from ..services.runpod_client import RunPodClient
from ..services.firebase import FirebaseService
from ..models.schemas import VoiceCloneRequest, VoiceCloneResponse, VoiceInfo
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
runpod_client = RunPodClient(
    api_key=settings.RUNPOD_API_KEY,
    voice_endpoint_id=settings.RUNPOD_ENDPOINT_ID,
    tts_endpoint_id=settings.TTS_ENDPOINT_ID
)
firebase_service = FirebaseService(
    credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
    bucket_name=settings.get_firebase_bucket_name()
)

@router.post("/clone", response_model=VoiceCloneResponse)
async def clone_voice(request: VoiceCloneRequest):
    """
    Clone a voice using uploaded audio.
    """
    try:
        logger.info(f"🎤 Voice clone request received for: {request.name}")
        logger.info(f"📊 Request details: language={request.language}, kids_voice={request.is_kids_voice}")
        
        # Call RunPod for voice cloning
        result = await runpod_client.create_voice_clone(
            name=request.name,
            audio_base64=request.audio_data,
            audio_format=request.audio_format,
            language=request.language,
            is_kids_voice=request.is_kids_voice
        )
        
        if result.get("status") == "success":
            logger.info("✅ Voice clone completed successfully")
            
            # Extract paths from RunPod response
            profile_path = result.get("profile_path")
            recorded_audio_path = result.get("recorded_audio_path")
            sample_audio_path = result.get("sample_audio_path")
            metadata = result.get("metadata", {})
            
            logger.info(f"📦 Profile path: {profile_path}")
            logger.info(f"🎵 Recorded audio path: {recorded_audio_path}")
            logger.info(f"🎵 Sample audio path: {sample_audio_path}")
            
            return VoiceCloneResponse(
                status="success",
                profile_path=profile_path,
                recorded_audio_path=recorded_audio_path,
                sample_audio_path=sample_audio_path,
                metadata=metadata
            )
        else:
            error_message = result.get("message", "Unknown error occurred")
            logger.error(f"❌ Voice clone failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
    except Exception as e:
        logger.error(f"❌ Voice clone error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {str(e)}")

@router.get("/")
async def list_voices():
    """
    List all available voices.
    """
    try:
        logger.info("📚 Listing voices from Firebase...")
        
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

@router.get("/by-language/{language}")
async def list_voices_by_language(language: str, is_kids_voice: bool = False):
    """
    List voices by language and kids voice flag.
    """
    try:
        logger.info(f"📚 Listing voices for language: {language}, is_kids_voice: {is_kids_voice}")
        
        # Get voices from Firebase
        voices = firebase_service.list_voices_by_language(language, is_kids_voice)
        
        logger.info(f"🔍 Firebase returned {len(voices)} voices")
        if voices:
            logger.info(f"🔍 First voice structure: {voices[0]}")
        
        # Debug: Log the raw Firebase data
        logger.info(f"🔍 Raw Firebase data for first voice: {voices[0] if voices else 'No voices found'}")
        
        # Convert to VoiceInfo objects with frontend-expected field names
        voice_list = []
        for voice in voices:
            # Get the first sample and profile URLs from the Firebase data
            sample_url = voice.get("samples", [])[0] if voice.get("samples") else None
            profile_url = voice.get("profiles", [])[0] if voice.get("profiles") else None
            
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
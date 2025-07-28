from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from ..services.runpod_client import RunPodClient
from ..services.firebase import FirebaseService
from ..models.schemas import VoiceCloneRequest, VoiceCloneResponse, VoiceInfo
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
runpod_client = RunPodClient()
firebase_service = FirebaseService()

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

@router.get("/", response_model=List[VoiceInfo])
async def list_voices():
    """
    List all available voices.
    """
    try:
        logger.info("📚 Listing voices from Firebase...")
        
        # Get voices from Firebase
        voices = firebase_service.list_voices_by_language("en", False)
        
        # Convert to VoiceInfo objects
        voice_list = []
        for voice in voices:
            voice_info = VoiceInfo(
                voice_id=voice.get("voice_id", ""),
                name=voice.get("name", ""),
                sample_file=voice.get("sample_file"),
                embedding_file=voice.get("embedding_file"),
                created_date=voice.get("created_date"),
                language=voice.get("language"),
                is_kids_voice=voice.get("is_kids_voice")
            )
            voice_list.append(voice_info)
        
        logger.info(f"✅ Found {len(voice_list)} voices")
        return voice_list
        
    except Exception as e:
        logger.error(f"❌ Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@router.get("/by-language/{language}", response_model=List[VoiceInfo])
async def list_voices_by_language(language: str):
    """
    List voices by language.
    """
    try:
        logger.info(f"📚 Listing voices for language: {language}")
        
        # Get voices from Firebase
        voices = firebase_service.list_voices_by_language(language, False)
        
        # Convert to VoiceInfo objects
        voice_list = []
        for voice in voices:
            voice_info = VoiceInfo(
                voice_id=voice.get("voice_id", ""),
                name=voice.get("name", ""),
                sample_file=voice.get("sample_file"),
                embedding_file=voice.get("embedding_file"),
                created_date=voice.get("created_date"),
                language=voice.get("language"),
                is_kids_voice=voice.get("is_kids_voice")
            )
            voice_list.append(voice_info)
        
        logger.info(f"✅ Found {len(voice_list)} voices for language {language}")
        return voice_list
        
    except Exception as e:
        logger.error(f"❌ Error listing voices for language {language}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}") 
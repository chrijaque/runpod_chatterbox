from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
import json
import logging
from datetime import datetime
import os

from ..models.schemas import (
    VoiceInfo, VoiceLibraryResponse, VoiceCloneResponse, VoiceCloneRequest, VoiceMetadata,
    ErrorResponse, SuccessResponse
)
from ..services.firebase import FirebaseService
from ..services.runpod_client import RunPodClient
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voices", tags=["voices"])

# Initialize services
firebase_service = FirebaseService(
    credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
    bucket_name=settings.get_firebase_bucket_name()
)

logger.info("🔍 ===== RUNPOD CLIENT CREATION DEBUG =====")
logger.info(f"📞 Creating RunPod client with:")
logger.info(f"   API Key: {'SET' if settings.RUNPOD_API_KEY else 'NOT SET'}")
logger.info(f"   Voice Endpoint ID: {settings.RUNPOD_ENDPOINT_ID}")
logger.info(f"   TTS Endpoint ID: {settings.TTS_ENDPOINT_ID}")

runpod_client = RunPodClient(
    api_key=settings.RUNPOD_API_KEY or "",
    voice_endpoint_id=settings.RUNPOD_ENDPOINT_ID or "",
    tts_endpoint_id=settings.TTS_ENDPOINT_ID or ""
)

logger.info(f"📞 RunPod client created: {type(runpod_client)}")
logger.info(f"📞 RunPod client configured: {runpod_client.is_configured()}")
logger.info("🔍 ===== END RUNPOD CLIENT CREATION DEBUG =====")

@router.get("/", response_model=VoiceLibraryResponse)
async def list_voices():
    """Get voice library with Firebase integration"""
    try:
        voices: List[VoiceInfo] = []
        
        # Since we're using Firebase for storage, return empty list
        # Voice listing should be done through Firebase endpoints
        return VoiceLibraryResponse(
            status="success",
            voices=[],
            total_voices=0
        )
        
    except Exception as e:
        logger.error(f"Error getting voice library: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clone", response_model=VoiceCloneResponse)
async def create_voice_clone(request: VoiceCloneRequest):
    """Create a new voice clone using RunPod with organized Firebase storage"""
    try:
        logger.info("🔍 ===== VOICE CLONE REQUEST DEBUG =====")
        logger.info(f"📥 Request type: {type(request)}")
        logger.info(f"📥 Request title: {request.title}")
        logger.info(f"📥 Request voices count: {len(request.voices) if request.voices else 0}")
        logger.info(f"📥 Request generated_sample: {bool(request.generated_sample)}")
        logger.info(f"📥 Request metadata: {request.metadata}")
        
        if not request.voices or len(request.voices) == 0:
            raise HTTPException(status_code=400, detail="At least one voice audio is required")
        
        # Extract metadata
        metadata = request.metadata
        language = metadata.get('language', 'en')
        is_kids_voice = metadata.get('isKidsVoice', False)
        user_id = metadata.get('userId', 'unknown')
        
        logger.info(f"🔄 Creating voice clone for: {request.title}")
        logger.info(f"   Language: {language}")
        logger.info(f"   Kids Voice: {is_kids_voice}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Audio files: {len(request.voices)}")
        
        # Generate voice ID from title
        import re
        clean_title = re.sub(r'[^a-zA-Z0-9_-]', '', request.title.lower().replace(' ', '_'))
        voice_id = f"voice_{clean_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process each voice file
        firebase_urls = {
            "recorded": [],
            "samples": [],
            "profiles": []
        }
        
        # Use the first voice for RunPod processing (you could extend this to process multiple)
        primary_audio = request.voices[0]
        
        # Check if we have a pre-generated sample to avoid duplicate RunPod calls
        if request.generated_sample:
            logger.info(f"🎯 Using pre-generated sample - skipping RunPod call for: {request.title}")
            
            # Use the pre-generated sample directly
            output = {
                "status": "success",
                "audio_base64": request.generated_sample,
                "profile_base64": None,  # We'll need to generate this separately if needed
                "metadata": {
                    "sample_file": f"/voice_samples/{voice_id}_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                    "profile_path": f"/voice_profiles/{voice_id}.npy",
                    "voice_id": voice_id,
                    "voice_name": request.title,
                    "generation_method": "pre_generated_sample"
                }
            }
            
            # Extract file information from our constructed response
            sample_file = output.get('metadata', {}).get('sample_file')
            profile_path = output.get('metadata', {}).get('profile_path')
            
            # Extract filename from path
            sample_filename = Path(sample_file).name
            profile_filename = f"{voice_id}.npy" if profile_path else None
            
        else:
            logger.info(f"🔄 No pre-generated sample - calling RunPod for: {request.title}")
            
            # Debug RunPod client call
            logger.info("🔍 ===== RUNPOD CLIENT DEBUG =====")
            logger.info(f"📞 RunPod client type: {type(runpod_client)}")
            logger.info(f"📞 RunPod client methods: {[m for m in dir(runpod_client) if not m.startswith('_')]}")
            logger.info(f"📞 RunPod client configured: {runpod_client.is_configured()}")
            
            # Check if create_voice_clone method exists
            if hasattr(runpod_client, 'create_voice_clone'):
                logger.info("✅ create_voice_clone method exists")
                import inspect
                sig = inspect.signature(runpod_client.create_voice_clone)
                logger.info(f"📞 Method signature: {sig}")
            else:
                logger.error("❌ create_voice_clone method does NOT exist")
                raise HTTPException(status_code=500, detail="RunPod client missing create_voice_clone method")
            
            # Prepare RunPod request
            runpod_request = {
                "name": request.title,
                "audio_data": primary_audio,
                "audio_format": "wav",
                "responseFormat": "base64"
            }
            
            logger.info(f"📤 Calling RunPod with parameters:")
            logger.info(f"   name: {request.title}")
            logger.info(f"   audio_base64 length: {len(primary_audio) if primary_audio else 0}")
            logger.info(f"   audio_format: wav")
            logger.info(f"   response_format: base64")
            
            # Submit to RunPod
            try:
                job = runpod_client.create_voice_clone(
                    name=request.title,
                    audio_base64=primary_audio,
                    audio_format="wav",
                    response_format="base64"
                )
                logger.info(f"✅ RunPod call successful, job: {job}")
            except Exception as e:
                logger.error(f"❌ RunPod call failed: {e}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"RunPod call failed: {str(e)}")
            if not job:
                raise HTTPException(status_code=500, detail="Failed to submit voice clone job to RunPod")
            
            # Wait for completion
            result = runpod_client.wait_for_job_completion(
                endpoint_id=settings.RUNPOD_ENDPOINT_ID or "",
                job_id=job['id']
            )
            if not result or result.get('status') != 'COMPLETED':
                error_msg = result.get('error', 'Unknown error') if result else 'Job failed'
                raise HTTPException(status_code=500, detail=f"Voice clone failed: {error_msg}")
            
            output = result.get('output', {})
            if output.get('status') != 'success':
                raise HTTPException(status_code=500, detail=f"Voice clone failed: {output.get('message', 'Unknown error')}")
            
            # Extract file information from RunPod response
            sample_file = output.get('metadata', {}).get('sample_file')
            profile_path = output.get('metadata', {}).get('profile_path')
            
            if not sample_file:
                raise HTTPException(status_code=500, detail="Invalid response from RunPod")
            
            # Extract filename from RunPod path
            sample_filename = Path(sample_file).name
            profile_filename = f"{voice_id}.npy" if profile_path else None
        
        # Upload user recordings to Firebase
        for i, audio_data in enumerate(request.voices):
            recording_filename = f"recording_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Build Firebase path based on language and kids voice
            if is_kids_voice:
                firebase_path = f"audio/voices/{language}/kids/recorded/{voice_id}_{recording_filename}"
            else:
                firebase_path = f"audio/voices/{language}/recorded/{voice_id}_{recording_filename}"
            
            # Upload base64 audio data directly to Firebase
            recording_url = firebase_service.upload_base64_audio(audio_data, firebase_path)
            if recording_url:
                firebase_urls['recorded'].append(recording_url)
                logger.info(f"✅ User recording uploaded to Firebase: {recording_url}")
            else:
                logger.error(f"❌ Failed to upload user recording {i+1}")
        
        # Upload voice sample to Firebase
        if sample_filename:
            if request.generated_sample:
                # Upload pre-generated sample directly to Firebase
                sample_filename_with_timestamp = f"{voice_id}_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                
                # Build Firebase path based on language and kids voice
                if is_kids_voice:
                    firebase_path = f"audio/voices/{language}/kids/samples/{sample_filename_with_timestamp}"
                else:
                    firebase_path = f"audio/voices/{language}/samples/{sample_filename_with_timestamp}"
                
                # Upload base64 sample data directly to Firebase
                sample_url = firebase_service.upload_base64_audio(request.generated_sample, firebase_path)
                if sample_url:
                    firebase_urls['samples'].append(sample_url)
                    logger.info(f"✅ Pre-generated voice sample uploaded to Firebase: {sample_url}")
                else:
                    logger.error(f"❌ Failed to upload pre-generated voice sample to Firebase")
            else:
                # Upload sample from RunPod directory
                sample_url = firebase_service.upload_runpod_voice_sample(
                    voice_id, sample_filename, language, is_kids_voice
                )
                if sample_url:
                    firebase_urls['samples'].append(sample_url)
                    logger.info(f"✅ Voice sample uploaded to Firebase: {sample_url}")
        
        # Upload voice profile to Firebase
        if profile_filename and profile_path:
            # Use the profile_base64 data from RunPod response instead of file access
            profile_base64 = output.get('profile_base64')
            if profile_base64:
                # Build Firebase path based on language and kids voice
                if is_kids_voice:
                    firebase_path = f"audio/voices/{language}/kids/profiles/{voice_id}_{profile_filename}"
                else:
                    firebase_path = f"audio/voices/{language}/profiles/{voice_id}_{profile_filename}"
                
                # Upload base64 profile data directly to Firebase
                profile_url = firebase_service.upload_base64_profile(profile_base64, firebase_path)
                if profile_url:
                    firebase_urls['profiles'].append(profile_url)
                    logger.info(f"✅ Voice profile uploaded to Firebase: {profile_url}")
                else:
                    logger.error(f"❌ Failed to upload voice profile to Firebase")
            else:
                logger.warning(f"⚠️ No profile_base64 data in RunPod response")
        else:
            if request.generated_sample:
                logger.info(f"ℹ️ No voice profile available when using pre-generated sample")
            else:
                logger.warning(f"⚠️ No profile_filename or profile_path available")
        
        # Clean up RunPod files after successful upload (only if we called RunPod)
        if sample_filename and not request.generated_sample:
            firebase_service.cleanup_runpod_file(f"/voice_samples/{sample_filename}")
        
        # Prepare response with organized Firebase URLs
        response_data = {
            "status": "success",
            "voice_id": voice_id,
            "voice_name": request.title,
            "audio_base64": output.get('audio_base64'),
            "profile_base64": output.get('profile_base64'),
            "metadata": {
                **output.get('metadata', {}),
                "firebase_urls": firebase_urls,
                "shared_access": True,
                "uploaded_at": datetime.now().isoformat(),
                "language": language,
                "is_kids_voice": is_kids_voice,
                "user_id": user_id,
                "visibility": request.visibility,
                "title": request.title,
                "recordings_count": len(request.voices),
                "optimization_used": bool(request.generated_sample),
                "runpod_calls_saved": 1 if request.generated_sample else 0
            }
        }
        
        logger.info(f"✅ Voice clone completed with organized storage: {voice_id}")
        return VoiceCloneResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating voice clone: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save", response_model=SuccessResponse)
async def save_voice_locally(
    voice_id: str = Form(...),
    voice_name: str = Form(...),
    audio_file: UploadFile = File(...),
    profile_file: Optional[UploadFile] = File(None),
    template_message: str = Form("")
):
    """Save voice files - DEPRECATED: Use Firebase endpoints instead"""
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use voice cloning with Firebase storage instead")

@router.get("/{voice_id}/sample")
async def get_voice_sample(voice_id: str):
    """Get voice sample file - DEPRECATED: Use Firebase endpoints instead"""
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use /{voice_id}/sample/firebase instead")

@router.get("/{voice_id}/sample/base64")
async def get_voice_sample_base64(voice_id: str):
    """Get voice sample as base64 - DEPRECATED: Use Firebase endpoints instead"""
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use /{voice_id}/sample/firebase instead")

@router.get("/{voice_id}/profile")
async def get_voice_profile(voice_id: str):
    """Get voice profile file - DEPRECATED: Use Firebase endpoints instead"""
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use /{voice_id}/firebase-urls instead")

@router.get("/{voice_id}/firebase-urls")
async def get_voice_firebase_urls(voice_id: str, language: str = "en", is_kids_voice: bool = False):
    """Get all Firebase URLs for a voice (shared access)"""
    try:
        firebase_urls = firebase_service.list_voice_files(voice_id, language, is_kids_voice)
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "language": language,
            "is_kids_voice": is_kids_voice,
            "firebase_urls": firebase_urls,
            "shared_access": True
        }
        
    except Exception as e:
        logger.error(f"Error getting Firebase URLs for voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{voice_id}/sample/firebase")
async def get_voice_sample_firebase(voice_id: str, language: str = "en", is_kids_voice: bool = False):
    """Get voice sample from Firebase (shared access)"""
    try:
        # Get the latest sample URL from Firebase
        firebase_urls = firebase_service.list_voice_files(voice_id, language, is_kids_voice)
        sample_urls = firebase_urls.get("samples", [])
        
        if not sample_urls:
            raise HTTPException(status_code=404, detail=f"No samples found for voice: {voice_id}")
        
        # Return the most recent sample URL
        latest_sample_url = sample_urls[-1]  # Assuming sorted by creation time
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "language": language,
            "is_kids_voice": is_kids_voice,
            "sample_url": latest_sample_url,
            "shared_access": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Firebase sample for voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-language/{language}")
async def list_voices_by_language(language: str, is_kids_voice: bool = False):
    """List all voices for a specific language and type"""
    try:
        voices = firebase_service.list_voices_by_language(language, is_kids_voice)
        
        return {
            "status": "success",
            "language": language,
            "is_kids_voice": is_kids_voice,
            "voices": voices,
            "total_voices": len(voices),
            "shared_access": True
        }
        
    except Exception as e:
        logger.error(f"Error listing voices for language {language}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/languages")
async def list_available_languages():
    """List all available languages in the voice library"""
    try:
        if not firebase_service.is_connected():
            raise HTTPException(status_code=500, detail="Firebase not connected")
        
        # This would require listing all directories in the bucket
        # For now, return common languages
        languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"}
        ]
        
        return {
            "status": "success",
            "languages": languages,
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Error listing languages: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
import json
import logging
from datetime import datetime

from ..models.schemas import (
    TTSGeneration, TTSGenerationsResponse, TTSGenerateResponse,
    TTSSaveRequest, TTSSaveResponse, ErrorResponse, SuccessResponse
)
from ..services.firebase import FirebaseService
from ..services.runpod_client import RunPodClient
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts", tags=["tts"])

# Initialize services
firebase_service = FirebaseService(
    credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
    bucket_name=settings.get_firebase_bucket_name()
)

runpod_client = RunPodClient(
    api_key=settings.RUNPOD_API_KEY or "",
    voice_endpoint_id=settings.RUNPOD_ENDPOINT_ID or "",
    tts_endpoint_id=settings.TTS_ENDPOINT_ID or ""
)

@router.get("/generations", response_model=TTSGenerationsResponse)
async def list_tts_generations():
    """Get TTS generations library with Firebase integration"""
    try:
        generations: List[TTSGeneration] = []
        
        if not settings.TTS_GENERATED_DIR.exists():
            return TTSGenerationsResponse(
                status="success",
                total_generations=0,
                generations=[]
            )
        
        tts_files = list(settings.TTS_GENERATED_DIR.glob("TTS_*.wav"))
        
        for tts_file in tts_files:
            try:
                filename_parts = tts_file.stem.split('_')
                if len(filename_parts) >= 4:
                    voice_id = '_'.join(filename_parts[1:-1])
                    timestamp = filename_parts[-1]
                    display_name = voice_id.replace("voice_", "").replace("_", " ").title()
                    
                    # Try to get Firebase URL
                    firebase_url = firebase_service.get_public_url(f"audio/stories/en/user/{tts_file.name}")
                    
                    generation = TTSGeneration(
                        file_id=tts_file.stem,
                        voice_id=voice_id,
                        voice_name=display_name,
                        file_path=str(tts_file),
                        created_date=tts_file.stat().st_mtime,
                        timestamp=timestamp,
                        file_size=tts_file.stat().st_size,
                        firebase_url=firebase_url
                    )
                    generations.append(generation)
                    
            except Exception as e:
                logger.error(f"Error parsing TTS file {tts_file}: {e}")
                continue
        
        generations.sort(key=lambda x: x.created_date or 0, reverse=True)
        
        return TTSGenerationsResponse(
            status="success",
            total_generations=len(generations),
            generations=generations
        )
        
    except Exception as e:
        logger.error(f"Error listing TTS generations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=TTSGenerateResponse)
async def generate_tts(
    voice_id: str = Form(...),
    text: str = Form(...),
    responseFormat: str = Form("base64"),
    language: str = Form("en"),
    is_kids_voice: bool = Form(False),
    story_type: str = Form("user")
):
    """Generate TTS using RunPod with organized stories storage"""
    try:
        if not voice_id or not text:
            raise HTTPException(status_code=400, detail="voice_id and text are required")
        
        if not runpod_client.is_configured():
            raise HTTPException(status_code=500, detail="RunPod configuration missing")
        
        # Validate story_type
        if story_type not in ["user", "app"]:
            raise HTTPException(status_code=400, detail="story_type must be 'user' or 'app'")
        
        logger.info(f"🎤 Generating TTS for voice: {voice_id}")
        logger.info(f"📝 Text length: {len(text)} characters")
        logger.info(f"🌍 Language: {language}")
        logger.info(f"👶 Kids Voice: {is_kids_voice}")
        logger.info(f"📚 Story Type: {story_type}")
        
        # Prepare RunPod request
        runpod_request = {
            "voice_id": voice_id,
            "text": text,
            "responseFormat": responseFormat
        }
        
        # Submit to RunPod
        job = await runpod_client.generate_tts(runpod_request)
        if not job:
            raise HTTPException(status_code=500, detail="Failed to submit TTS job to RunPod")
        
        # Wait for completion
        result = await runpod_client.wait_for_job_completion(job['id'])
        if not result or result.get('status') != 'COMPLETED':
            error_msg = result.get('error', 'Unknown error') if result else 'Job failed'
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {error_msg}")
        
        output = result.get('output', {})
        if output.get('status') != 'success':
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {output.get('message', 'Unknown error')}")
        
        # Extract file information from RunPod response
        tts_file = output.get('metadata', {}).get('tts_file')
        if not tts_file:
            raise HTTPException(status_code=500, detail="Invalid response from RunPod")
        
        # Extract filename from RunPod path
        tts_filename = Path(tts_file).name
        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Upload to Firebase for shared access
        firebase_urls = {}
        
        # Upload TTS generation to stories directory
        if tts_filename:
            tts_url = firebase_service.upload_runpod_tts_generation(
                generation_id, tts_filename, language, is_kids_voice, story_type
            )
            if tts_url:
                firebase_urls['story_url'] = tts_url
                logger.info(f"✅ TTS story uploaded to Firebase: {tts_url}")
        
        # Clean up RunPod file after successful upload
        if tts_filename:
            firebase_service.cleanup_runpod_file(f"/voice_samples/{tts_filename}")
        
        # Prepare response with shared access URLs
        response_data = {
            "status": "success",
            "generation_id": generation_id,
            "voice_id": voice_id,
            "text": text,
            "audio_base64": output.get('audio_base64'),
            "metadata": {
                **output.get('metadata', {}),
                "firebase_urls": firebase_urls,
                "shared_access": True,
                "uploaded_at": datetime.now().isoformat(),
                "generation_id": generation_id,
                "language": language,
                "is_kids_voice": is_kids_voice,
                "story_type": story_type
            }
        }
        
        logger.info(f"✅ TTS story generation completed with organized storage: {generation_id}")
        return TTSGenerateResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save", response_model=TTSSaveResponse)
async def save_tts_generation(
    voice_id: str = Form(...),
    audio_base64: str = Form(...),
    text_input: str = Form(...),
    generation_time: float = Form(...),
    sample_rate: int = Form(...),
    audio_shape: str = Form(...)  # JSON string of audio shape
):
    """Save TTS generation locally and to Firebase"""
    try:
        if not voice_id or not audio_base64:
            raise HTTPException(status_code=400, detail="voice_id and audio_base64 are required")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
        
        # Save TTS file locally
        tts_filename = settings.TTS_GENERATED_DIR / f"TTS_{voice_id}_{timestamp}.wav"
        with open(tts_filename, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"✅ Saved TTS file: {tts_filename}")
        
        # Upload to Firebase
        firebase_url = None
        if firebase_service.is_connected():
            firebase_url = firebase_service.upload_file(
                tts_filename,
                f"audio/stories/en/user/{tts_filename.name}"
            )
            if firebase_url:
                logger.info(f"✅ Uploaded TTS to Firebase: {firebase_url}")
        
        # Parse audio shape
        try:
            audio_shape_list = json.loads(audio_shape)
        except:
            audio_shape_list = []
        
        return TTSSaveResponse(
            status="success",
            message="TTS generation saved successfully",
            voice_id=voice_id,
            local_file=str(tts_filename),
            firebase_url=firebase_url,
            file_size_bytes=len(audio_data),
            file_size_mb=len(audio_data) / 1024 / 1024,
            metadata={
                "voice_id": voice_id,
                "voice_name": voice_id.replace('voice_', ''),
                "text_input": text_input[:500] + "..." if len(text_input) > 500 else text_input,
                "generation_time": generation_time,
                "sample_rate": sample_rate,
                "audio_shape": audio_shape_list,
                "timestamp": timestamp
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving TTS generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generations/{file_id}/audio")
async def get_tts_audio(file_id: str):
    """Get TTS audio file"""
    try:
        tts_file = settings.TTS_GENERATED_DIR / f"{file_id}.wav"
        if not tts_file.exists():
            raise HTTPException(status_code=404, detail=f"TTS file not found: {file_id}")
        
        return FileResponse(tts_file, media_type="audio/wav")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting TTS audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generations/{generation_id}/firebase")
async def get_tts_firebase_url(generation_id: str):
    """Get TTS generation from Firebase (shared access)"""
    try:
        # Get the TTS URL from Firebase
        tts_url = firebase_service.get_public_url(f"audio/stories/en/user/{generation_id}")
        
        if not tts_url:
            raise HTTPException(status_code=404, detail=f"TTS generation not found: {generation_id}")
        
        return {
            "status": "success",
            "generation_id": generation_id,
            "tts_url": tts_url,
            "shared_access": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Firebase TTS for generation {generation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage/usage")
async def get_storage_usage():
    """Get Firebase storage usage statistics"""
    try:
        usage_stats = firebase_service.get_storage_usage()
        
        return {
            "status": "success",
            "storage_usage": usage_stats,
            "shared_access": True
        }
        
    except Exception as e:
        logger.error(f"Error getting storage usage: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/stories/languages")
async def list_story_languages():
    """List all available languages for stories"""
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
            {"code": "zh", "name": "Chinese"},
            {"code": "da", "name": "Danish"},
            {"code": "sv", "name": "Swedish"},
            {"code": "no", "name": "Norwegian"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"}
        ]
        
        return {
            "status": "success",
            "languages": languages,
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Error listing story languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stories/{language}")
async def list_stories_by_language(language: str, story_type: str = "user"):
    """List all stories for a specific language and type"""
    try:
        if story_type not in ["user", "app"]:
            raise HTTPException(status_code=400, detail="story_type must be 'user' or 'app'")
        
        stories = firebase_service.list_stories_by_language(language, story_type)
        
        return {
            "status": "success",
            "language": language,
            "story_type": story_type,
            "stories": stories,
            "total_stories": len(stories),
            "shared_access": True
        }
        
    except Exception as e:
        logger.error(f"Error listing stories for language {language}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stories/{language}/{story_type}")
async def list_stories_by_language_and_type(language: str, story_type: str):
    """List all stories for a specific language and type (alternative endpoint)"""
    return await list_stories_by_language(language, story_type)

@router.get("/stories/{language}/{story_type}/{generation_id}/audio")
async def get_story_audio(language: str, story_type: str, generation_id: str):
    """Get story audio from Firebase (shared access)"""
    try:
        if story_type not in ["user", "app"]:
            raise HTTPException(status_code=400, detail="story_type must be 'user' or 'app'")
        
        # Get the story URL from Firebase
        story_url = firebase_service.get_story_audio_url(generation_id, language, story_type)
        
        if not story_url:
            raise HTTPException(status_code=404, detail=f"Story not found: {generation_id}")
        
        return {
            "status": "success",
            "language": language,
            "story_type": story_type,
            "generation_id": generation_id,
            "story_url": story_url,
            "shared_access": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting story audio for generation {generation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
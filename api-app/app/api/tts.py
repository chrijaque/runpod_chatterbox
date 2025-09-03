from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
import logging
from ..services.runpod_client import RunPodClient
from ..services.firebase import FirebaseService
from ..models.schemas import (
    TTSGenerateRequest,
    TTSGenerateResponse,
    TTSGeneration,
    TTSErrorCallbackRequest,
    TTSErrorCallbackResponse,
    TTSSuccessCallbackRequest,
    TTSSuccessCallbackResponse,
)
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
# Import the get_firebase_service function
from .voices import get_firebase_service

# Get Firebase service using the proper initialization
firebase_service = get_firebase_service()

@router.post("/error-callback", response_model=TTSErrorCallbackResponse)
async def tts_error_callback(request: TTSErrorCallbackRequest):
    """
    Handle TTS generation error callbacks from RunPod service.
    Updates story status in Firestore and triggers frontend notifications.
    """
    try:
        logger.info(f"❌ TTS Error callback received for story: {request.story_id}")
        logger.info(f"🔍 Error details: {request.error}")
        logger.info(f"🔍 User ID: {request.user_id}")
        logger.info(f"🔍 Voice ID: {request.voice_id}")
        logger.info(f"🔍 Job ID: {request.job_id}")
        
        # Update story status in Firestore
        try:
            # Import Firestore client
            import firebase_admin
            from firebase_admin import firestore
            
            # Get Firestore client
            db = firestore.client()
            
            # Update the story document with error information
            story_ref = db.collection('stories').document(request.story_id)
            
            update_data = {
                'audioStatus': 'failed',
                'audioError': request.error,
                'audioErrorDetails': request.error_details,
                'audioJobId': request.job_id,
                'updatedAt': firestore.SERVER_TIMESTAMP
            }
            
            # Add metadata if provided
            if request.metadata:
                update_data['audioErrorMetadata'] = request.metadata
            
            # Update the document
            story_ref.update(update_data)
            
            logger.info(f"✅ Story {request.story_id} updated with error status in Firestore")
            
            return TTSErrorCallbackResponse(success=True)
            
        except Exception as firestore_error:
            logger.error(f"❌ Failed to update Firestore for story {request.story_id}: {firestore_error}")
            return TTSErrorCallbackResponse(
                success=False, 
                error=f"Failed to update Firestore: {str(firestore_error)}"
            )
            
    except Exception as e:
        logger.error(f"❌ TTS Error callback processing failed: {str(e)}")
        return TTSErrorCallbackResponse(
            success=False, 
            error=f"Internal server error: {str(e)}"
        )

@router.post("/callback", response_model=TTSSuccessCallbackResponse)
async def tts_success_callback(request: TTSSuccessCallbackRequest):
    """
    Handle TTS success callbacks from RunPod worker. Update Firestore story document
    with the generated audio URL/path and mark audioStatus as 'ready'.
    """
    try:
        logger.info(f"✅ TTS Success callback received for story: {request.story_id}")
        logger.info(f"🔍 User ID: {request.user_id}")
        logger.info(f"🔍 Voice ID: {request.voice_id}")
        logger.info(f"🔍 Audio URL: {request.audio_url}")
        logger.info(f"🔍 Storage path: {request.storage_path}")

        # Update story document in Firestore
        try:
            import firebase_admin
            from firebase_admin import firestore

            db = firestore.client()
            story_ref = db.collection("stories").document(request.story_id)

            update_data = {
                "audioStatus": "ready",
                "audioUrl": request.audio_url,
                "audioStoragePath": request.storage_path,
                "voiceId": request.voice_id,
                "voiceName": request.voice_name,
                "language": request.language,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }

            # Merge in metadata if present
            if request.metadata:
                update_data["audioMetadata"] = request.metadata

            story_ref.update(update_data)

            logger.info(f"✅ Story {request.story_id} updated with audio info in Firestore")
            return TTSSuccessCallbackResponse(success=True)
        except Exception as firestore_error:
            logger.error(f"❌ Failed to update Firestore for story {request.story_id}: {firestore_error}")
            return TTSSuccessCallbackResponse(success=False, error=f"Failed to update Firestore: {str(firestore_error)}")
    except Exception as e:
        logger.error(f"❌ TTS success callback processing failed: {str(e)}")
        return TTSSuccessCallbackResponse(success=False, error=f"Internal server error: {str(e)}")

@router.post("/generate", response_model=TTSGenerateResponse, dependencies=[Depends(verify_hmac)])
async def generate_tts(request: TTSGenerateRequest, job_id: str | None = None, http_req: Request | None = None):
    """
    Generate TTS using a voice profile.
    """
    try:
        logger.info(f"📖 TTS generation request received for voice: {request.voice_id}")
        logger.info(f"📊 Request details: language={request.language}, story_type={request.story_type}, kids_voice={request.is_kids_voice}")
        try:
            b64_len = len(request.profile_base64 or "")
        except Exception:
            b64_len = 0
        logger.info(f"🔑 Profile base64 length: {b64_len}")
        logger.info(f"📁 Profile path: {getattr(request, 'profile_path', None) or ''}")
        logger.info(f"📝 Text preview: {request.text[:50]}...")
        # Validate that one of profile_base64 or profile_path is provided
        if not (request.profile_base64 or getattr(request, 'profile_path', None)):
            raise HTTPException(status_code=400, detail="Either profile_base64 or profile_path must be provided")
        
        # Queue disabled: call RunPod directly (synchronous)

        # Fallback: Call RunPod for TTS generation synchronously
        logger.info("📞 Calling RunPod client generate_tts_with_context...")
        logger.info(f"🔧 RunPod client config - TTS Endpoint ID: {runpod_client.tts_endpoint_id}")
        logger.info(f"🔧 RunPod client config - Base URL: {runpod_client.base_url}")
        
        try:
            # Default callback_url if none provided
            default_cb = None
            try:
                if settings.PUBLIC_API_BASE_URL:
                    default_cb = settings.PUBLIC_API_BASE_URL.rstrip("/") + "/api/tts/callback"
                elif http_req is not None and hasattr(http_req, "base_url"):
                    default_cb = str(http_req.base_url).rstrip("/") + "/api/tts/callback"
            except Exception:
                default_cb = None

            result = runpod_client.generate_tts_with_context(
                voice_id=request.voice_id,
                text=request.text,
                profile_base64=request.profile_base64 or "",
                language=request.language,
                story_type=request.story_type,
                is_kids_voice=request.is_kids_voice,
                model_type=request.model_type,
                user_id=request.user_id,
                story_id=request.story_id,
                profile_path=request.profile_path,
                callback_url=(request.callback_url or default_cb),
                story_name=getattr(request, 'story_name', None),
                output_basename=getattr(request, 'output_basename', None),
                voice_name=getattr(request, 'voice_name', None),
            )
            logger.info(f"✅ RunPod call completed successfully")
        except Exception as e:
            logger.error(f"❌ RunPod call failed with exception: {str(e)}")
            logger.error(f"❌ Exception type: {type(e)}")
            raise
        logger.info(f"✅ RunPod response received: {type(result)}")
        logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict):
            logger.info(f"🔍 Response status: {result.get('status', 'No status')}")
            logger.info(f"🔍 Response message: {result.get('message', 'No message')}")
        
        # Return immediately when job is queued (like voice cloning)
        if result.get("status") in ["IN_QUEUE", "IN_PROGRESS"] and result.get("id"):
            job_id = result.get("id")
            logger.info(f"⏳ TTS job queued with ID: {job_id}, returning immediately...")
            
            # Return quickly after dispatching job; UI will listen to Firestore updates via callback
            return TTSGenerateResponse(
                status="queued",
                job_id=job_id,
                metadata={"message": "TTS generation job queued successfully"}
            )
        elif result.get("status") == "success":
            logger.info("✅ TTS generation completed successfully")
            
            # Extract audio path from RunPod response
            audio_path = result.get("audio_path")
            metadata = result.get("metadata", {})
            
            logger.info(f"🎵 Audio path: {audio_path}")
            
            return TTSGenerateResponse(
                status="success",
                audio_path=audio_path,
                metadata=metadata
            )
        else:
            error_message = result.get("error", result.get("message", "Unknown error occurred"))
            logger.error(f"❌ TTS generation failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
    except Exception as e:
        logger.error(f"❌ TTS generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@router.get("/generations", response_model=List[TTSGeneration])
async def list_tts_generations():
    """
    List all TTS generations.
    """
    try:
        logger.info("📚 Listing TTS generations from Firebase...")
        
        # Get TTS generations from Firebase
        generations = firebase_service.list_stories_by_language("en", "user")
        
        # Convert to TTSGeneration objects
        generation_list = []
        for generation in generations:
            tts_generation = TTSGeneration(
                generation_id=generation.get("generation_id", ""),
                voice_id=generation.get("voice_id", ""),
                voice_name=generation.get("voice_name"),
                text_input=generation.get("text_input"),
                audio_file=generation.get("audio_file"),
                created_date=generation.get("created_date"),
                language=generation.get("language"),
                story_type=generation.get("story_type")
            )
            generation_list.append(tts_generation)
        
        logger.info(f"✅ Found {len(generation_list)} TTS generations")
        return generation_list
        
    except Exception as e:
        logger.error(f"❌ Error listing TTS generations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list TTS generations: {str(e)}")

@router.get("/test-files")
async def test_firebase_files():
    """
    Test endpoint to list all files in Firebase stories directory
    """
    try:
        logger.info("🔍 Testing Firebase files listing...")
        
        # Test listing all files in stories directory
        files = firebase_service.test_list_all_files("audio/stories/en/user/")
        
        logger.info(f"📁 Found files: {files}")
        
        return {
            "status": "success",
            "files": files,
            "message": f"Found {len(files)} files in Firebase"
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing Firebase files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test Firebase files: {str(e)}")

@router.get("/stories/{language}")
async def list_stories_by_language(language: str, story_type: str = "user"):
    """
    List TTS stories by language and story type.
    """
    try:
        logger.info(f"📚 Listing TTS stories for language: {language}, story_type: {story_type}")
        
        # Get stories from Firebase
        stories = firebase_service.list_stories_by_language(language, story_type)
        
        logger.info(f"✅ Found {len(stories)} stories for {language}/{story_type}")
        
        return {
            "status": "success",
            "stories": stories,
            "language": language,
            "story_type": story_type,
            "total": len(stories)
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing stories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list stories: {str(e)}")

@router.get("/stories/{language}/{story_type}/{file_id}/audio")
async def get_story_audio(language: str, story_type: str, file_id: str):
    """
    Get audio URL for a specific story.
    """
    try:
        logger.info(f"🎵 Getting audio URL for story: {language}/{story_type}/{file_id}")
        
        # The file_id contains voice_id_timestamp, so we need to construct the full filename
        # Pattern: voice_christianmp3test2_20250728_075206
        # We need: TTS_voice_christianmp3test2_20250728_075206.wav
        
        # Construct the Firebase path
        firebase_path = f"audio/stories/{language}/{story_type}/TTS_{file_id}.wav"
        
        logger.info(f"🔍 Looking for file: {firebase_path}")
        
        # Get the public URL from Firebase
        audio_url = firebase_service.get_public_url(firebase_path)
        
        if audio_url:
            logger.info(f"✅ Audio URL: {audio_url}")
            return {
                "status": "success",
                "audio_url": audio_url,
                "file_id": file_id,
                "language": language,
                "story_type": story_type
            }
        else:
            logger.error(f"❌ Audio file not found: {firebase_path}")
            raise HTTPException(status_code=404, detail="Audio file not found")
        
    except Exception as e:
        logger.error(f"❌ Error getting story audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get story audio: {str(e)}") 
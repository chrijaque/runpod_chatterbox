from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
import logging
from ..services.runpod_client import RunPodClient
from ..models.schemas import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMSuccessCallbackRequest,
    LLMSuccessCallbackResponse,
    LLMErrorCallbackRequest,
    LLMErrorCallbackResponse,
)
from ..config import settings
from ..middleware.security import verify_hmac

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RunPod client with LLM endpoint
runpod_client = RunPodClient(
    api_key=settings.RUNPOD_API_KEY,
    voice_endpoint_id=settings.VC_CB_ENDPOINT_ID,
    tts_endpoint_id=settings.TTS_CB_ENDPOINT_ID,
    llm_endpoint_id=settings.LLM_CB_ENDPOINT_ID
)

@router.post("/generate-story", response_model=LLMGenerateResponse, dependencies=[Depends(verify_hmac)])
async def generate_story_llm(request: LLMGenerateRequest, http_req: Request):
    """
    Generate story content using RunPod LLM endpoint.
    """
    try:
        logger.info(f"📖 LLM story generation request received for story: {request.story_id}")
        logger.info(f"📊 Request details: language={request.language}, genre={request.genre}, age_range={request.age_range}")
        
        # Log message counts based on workflow type
        is_two_step = request.workflow_type == "two-step" or (request.outline_messages and request.story_messages)
        if is_two_step:
            logger.info(f"📝 Two-step workflow: outline_messages={len(request.outline_messages) if request.outline_messages else 0}, story_messages={len(request.story_messages) if request.story_messages else 0}")
        else:
            logger.info(f"📝 Single-step workflow: messages count={len(request.messages) if request.messages else 0}")
        
        if not settings.LLM_CB_ENDPOINT_ID:
            raise HTTPException(status_code=500, detail="LLM endpoint not configured")
        
        # Default callback_url if none provided
        if settings.PUBLIC_API_BASE_URL:
            default_cb = settings.PUBLIC_API_BASE_URL.rstrip("/") + "/api/llm/callback"
        else:
            default_cb = "https://runpod-chatterbox.fly.dev/api/llm/callback"
            logger.warning("⚠️ PUBLIC_API_BASE_URL not set, using hardcoded fallback URL")
        
        result = runpod_client.generate_llm_completion(
            messages=request.messages or [],  # Use empty list if None (for two-step workflow)
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            language=request.language,
            genre=request.genre,
            age_range=request.age_range,
            user_id=request.user_id,
            story_id=request.story_id,
            callback_url=(request.callback_url or default_cb),
            workflow_type=request.workflow_type,
            outline_messages=request.outline_messages,
            story_messages=request.story_messages,
        )
        
        logger.info(f"✅ RunPod LLM call completed")
        logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Check for error status first
        if result.get("status") == "error":
            error_message = result.get("error", result.get("message", "Unknown error occurred"))
            logger.error(f"❌ LLM generation failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
        
        # Return immediately when job is queued
        elif result.get("status") in ["IN_QUEUE", "IN_PROGRESS"] and result.get("id"):
            job_id = result.get("id")
            logger.info(f"⏳ LLM job queued with ID: {job_id}, returning immediately...")
            
            return LLMGenerateResponse(
                status="queued",
                job_id=job_id,
                metadata={"message": "LLM generation job queued successfully"}
            )
        elif result.get("status") == "success":
            logger.info("✅ LLM generation completed successfully")
            
            content = result.get("content")
            metadata = result.get("metadata", {})
            
            logger.info(f"📝 Generated content length: {len(content) if content else 0}")
            
            return LLMGenerateResponse(
                status="success",
                content=content,
                metadata=metadata
            )
        else:
            error_message = result.get("error", result.get("message", "Unknown error occurred"))
            logger.error(f"❌ LLM generation failed: {error_message}")
            raise HTTPException(status_code=500, detail=error_message)
            
    except Exception as e:
        logger.error(f"❌ LLM generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

@router.post("/callback", response_model=LLMSuccessCallbackResponse)
async def llm_success_callback(request: LLMSuccessCallbackRequest):
    """
    Handle LLM success callbacks from RunPod worker. Update Firestore story document
    with the generated content and mark status as 'ready'.
    Note: Handler already saves to Firestore, but this callback ensures updates are applied.
    """
    try:
        logger.info(f"✅ LLM Success callback received for story: {request.story_id}")
        logger.info(f"🔍 User ID: {request.user_id}")
        logger.info(f"📝 Content length: {len(request.content) if request.content else 0}")

        # Update story document in Firestore
        try:
            import firebase_admin
            from firebase_admin import firestore

            db = firestore.client()
            
            # Check if this is a default story
            default_story_ref = db.collection("defaultStories").document(request.story_id)
            default_snap = default_story_ref.get()
            is_default_story = default_snap.exists
            
            if is_default_story:
                # Update defaultStories collection
                update_data = {
                    "content": request.content,
                    "generationStatus": "ready",
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                }
                
                # Merge in metadata if present
                if request.metadata:
                    update_data["llmMetadata"] = request.metadata
                
                default_story_ref.update(update_data)
                logger.info(f"✅ Default story {request.story_id} updated with LLM content in Firestore")
            else:
                # Update stories collection
                story_ref = db.collection("stories").document(request.story_id)
                update_data = {
                    "content": request.content,
                    "status": "ready",
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                }
                
                # Merge in metadata if present
                if request.metadata:
                    update_data["llmMetadata"] = request.metadata
                
                story_ref.update(update_data)
                logger.info(f"✅ Story {request.story_id} updated with LLM content in Firestore")

            return LLMSuccessCallbackResponse(success=True)
        except Exception as firestore_error:
            logger.error(f"❌ Failed to update Firestore for story {request.story_id}: {firestore_error}")
            return LLMSuccessCallbackResponse(success=False, error=f"Failed to update Firestore: {str(firestore_error)}")
    except Exception as e:
        logger.error(f"❌ LLM success callback processing failed: {str(e)}")
        return LLMSuccessCallbackResponse(success=False, error=f"Internal server error: {str(e)}")

@router.post("/error-callback", response_model=LLMErrorCallbackResponse)
async def llm_error_callback(request: LLMErrorCallbackRequest):
    """
    Handle LLM generation error callbacks from RunPod service.
    Updates story status in Firestore and triggers frontend notifications.
    """
    try:
        logger.info(f"❌ LLM Error callback received for story: {request.story_id}")
        logger.info(f"🔍 Error details: {request.error}")
        logger.info(f"🔍 User ID: {request.user_id}")
        logger.info(f"🔍 Job ID: {request.job_id}")
        
        # Update story status in Firestore
        try:
            import firebase_admin
            from firebase_admin import firestore
            
            db = firestore.client()
            
            # Check if this is a default story
            default_story_ref = db.collection("defaultStories").document(request.story_id)
            default_snap = default_story_ref.get()
            is_default_story = default_snap.exists
            
            if is_default_story:
                # Update defaultStories collection
                update_data = {
                    'generationStatus': 'failed',
                    'failureReason': request.error,
                    'llmErrorDetails': request.error_details,
                    'llmJobId': request.job_id,
                    'updatedAt': firestore.SERVER_TIMESTAMP
                }
                
                # Add metadata if provided
                if request.metadata:
                    update_data['llmErrorMetadata'] = request.metadata
                
                default_story_ref.update(update_data)
                logger.info(f"✅ Default story {request.story_id} updated with error status in Firestore")
            else:
                # Update stories collection
                story_ref = db.collection('stories').document(request.story_id)
                update_data = {
                    'status': 'failed',
                    'failureReason': request.error,
                    'llmErrorDetails': request.error_details,
                    'llmJobId': request.job_id,
                    'updatedAt': firestore.SERVER_TIMESTAMP
                }
                
                # Add metadata if provided
                if request.metadata:
                    update_data['llmErrorMetadata'] = request.metadata
                
                story_ref.update(update_data)
                logger.info(f"✅ Story {request.story_id} updated with error status in Firestore")
            
            return LLMErrorCallbackResponse(success=True)
            
        except Exception as firestore_error:
            logger.error(f"❌ Failed to update Firestore for story {request.story_id}: {firestore_error}")
            return LLMErrorCallbackResponse(
                success=False, 
                error=f"Failed to update Firestore: {str(firestore_error)}"
            )
            
    except Exception as e:
        logger.error(f"❌ LLM Error callback processing failed: {str(e)}")
        return LLMErrorCallbackResponse(
            success=False, 
            error=f"Internal server error: {str(e)}"
        )


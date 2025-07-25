from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import os
import logging
from datetime import datetime

from ..models.schemas import HealthResponse, DebugResponse
from ..services.firebase import FirebaseService
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Initialize Firebase service for health checks
firebase_service = FirebaseService(
    credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
    bucket_name=settings.get_firebase_bucket_name()
)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        firebase_connected = firebase_service.is_connected()
        
        return HealthResponse(
            status="healthy",
            service="voice-library-api",
            firebase_connected=firebase_connected,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/debug/directories", response_model=DebugResponse)
async def debug_directories():
    """Debug endpoint to check Firebase storage status"""
    try:
        firebase_connected = firebase_service.is_connected()
        
        # Get Firebase storage usage
        storage_usage = {}
        if firebase_connected:
            storage_usage = firebase_service.get_storage_usage()
        
        return DebugResponse(
            status="success",
            directories={},  # No local directories needed
            current_working_directory=os.getcwd(),
            timestamp=datetime.now().isoformat(),
            firebase_connected=firebase_connected,
            firebase_storage_usage=storage_usage
        )
        
    except Exception as e:
        logger.error(f"Debug directories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
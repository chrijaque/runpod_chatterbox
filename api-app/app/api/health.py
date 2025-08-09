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

# Import the get_firebase_service function
from .voices import get_firebase_service

# Get Firebase service using the proper initialization (may be None if not configured)
firebase_service = get_firebase_service()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Re-resolve service in case env changed after startup
        service = firebase_service or get_firebase_service()
        firebase_connected = service.is_connected() if service else False
        
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
        service = firebase_service or get_firebase_service()
        firebase_connected = service.is_connected() if service else False
        
        # Get Firebase storage usage
        storage_usage = {}
        if firebase_connected and service:
            storage_usage = service.get_storage_usage()
        
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
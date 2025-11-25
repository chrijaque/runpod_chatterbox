#!/usr/bin/env python3
import uvicorn
import logging
from pathlib import Path
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server"""
    logger.info("🎵 Voice Library API Server Starting...")
    
    # Log configuration
    logger.info(f"🔥 R2: {'Enabled' if settings.FIREBASE_STORAGE_ENABLED else 'Disabled'}")
    logger.info(f"🌐 API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"📚 API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    
    # Validate Firebase configuration
    if settings.FIREBASE_STORAGE_ENABLED and not settings.validate_firebase_config():
        logger.warning("⚠️ Firebase configuration invalid - check firebase_creds.json and FIREBASE_STORAGE_BUCKET")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main() 
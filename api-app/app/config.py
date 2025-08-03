import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the api-app directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class Settings:
    # Firebase Configuration
    FIREBASE_CREDENTIALS: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET")
    
    # Local Firebase credentials file for library display
    FIREBASE_LOCAL_CREDS_FILE: str = os.getenv("FIREBASE_LOCAL_CREDS_FILE", "../firebase_local_only.json")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000", 
        "http://localhost:5001", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5001"
    ]
    
    # Storage Configuration
    LOCAL_STORAGE_ENABLED: bool = os.getenv("LOCAL_STORAGE_ENABLED", "False").lower() == "true"
    FIREBASE_STORAGE_ENABLED: bool = os.getenv("FIREBASE_STORAGE_ENABLED", "True").lower() == "true"
    
    # File Size Limits
    MAX_AUDIO_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_PROFILE_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # RunPod Configuration - ChatterboxTTS
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    VC_CB_ENDPOINT_ID: Optional[str] = os.getenv("VC_CB_ENDPOINT_ID")
    TTS_CB_ENDPOINT_ID: Optional[str] = os.getenv("TTS_CB_ENDPOINT_ID")
    
    # RunPod Configuration - ChatterboxTTS Only

    @classmethod
    def validate_firebase_config(cls) -> bool:
        """Validate Firebase configuration"""
        if not cls.FIREBASE_STORAGE_ENABLED:
            return True
        if not cls.FIREBASE_CREDENTIALS:
            return False
        if not cls.FIREBASE_STORAGE_BUCKET:
            return False
        return True

    @classmethod
    def get_firebase_bucket_name(cls) -> str:
        """Get Firebase bucket name without gs:// prefix"""
        if cls.FIREBASE_STORAGE_BUCKET.startswith("gs://"):
            return cls.FIREBASE_STORAGE_BUCKET[5:]
        return cls.FIREBASE_STORAGE_BUCKET

    @classmethod
    def validate_runpod_config(cls) -> bool:
        """Validate RunPod configuration"""
        return all([
            cls.RUNPOD_API_KEY,
            cls.VC_CB_ENDPOINT_ID,
            cls.TTS_CB_ENDPOINT_ID
        ])

    @classmethod
    def get_missing_config(cls) -> list:
        """Get list of missing configuration items"""
        missing = []
        
        if cls.FIREBASE_STORAGE_ENABLED and not cls.validate_firebase_config():
            missing.append("Firebase configuration (check RUNPOD_SECRET_Firebase and FIREBASE_STORAGE_BUCKET)")
        
        if not cls.validate_runpod_config():
            if not cls.RUNPOD_API_KEY:
                missing.append("RUNPOD_API_KEY")
            if not cls.VC_CB_ENDPOINT_ID:
                missing.append("VC_CB_ENDPOINT_ID")
            if not cls.TTS_CB_ENDPOINT_ID:
                missing.append("TTS_CB_ENDPOINT_ID")
        
        return missing

settings = Settings()

# Debug: Log configuration on startup
import logging
logger = logging.getLogger(__name__)

logger.info("🔍 ===== CONFIGURATION DEBUG =====")
logger.info(f"📋 RUNPOD_API_KEY: {'SET' if settings.RUNPOD_API_KEY else 'NOT SET'}")
logger.info(f"📋 VC_CB_ENDPOINT_ID: {settings.VC_CB_ENDPOINT_ID}")
logger.info(f"📋 TTS_CB_ENDPOINT_ID: {settings.TTS_CB_ENDPOINT_ID}")
logger.info(f"📋 FIREBASE_STORAGE_BUCKET: {settings.FIREBASE_STORAGE_BUCKET}")
logger.info(f"📋 FIREBASE_CREDENTIALS: {'SET' if settings.FIREBASE_CREDENTIALS else 'NOT SET'}")
if settings.FIREBASE_CREDENTIALS:
    logger.info(f"📋 FIREBASE_CREDENTIALS_LENGTH: {len(settings.FIREBASE_CREDENTIALS)} characters")
    logger.info(f"📋 FIREBASE_CREDENTIALS_PREVIEW: {settings.FIREBASE_CREDENTIALS[:200]}...")
else:
    logger.warning("⚠️ RUNPOD_SECRET_Firebase environment variable is not set!")
    logger.warning("⚠️ Firebase functionality will not work without proper credentials")
logger.info(f"📋 RunPod config valid: {settings.validate_runpod_config()}")
logger.info(f"📋 Firebase config valid: {settings.validate_firebase_config()}")
logger.info("🔍 ===== END CONFIGURATION DEBUG =====") 
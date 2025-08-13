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
    # Note: RUNPOD_SECRET_Firebase is used by RunPod handlers for uploading
    # firebase_local_only.json is used by API app for reading library data
    FIREBASE_CREDENTIALS: Optional[str] = os.getenv("RUNPOD_SECRET_Firebase")  # For RunPod uploads
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET")
    
    # Local Firebase credentials file for library display (API app only)
    # Use absolute path to project root
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    FIREBASE_LOCAL_CREDS_FILE: str = os.getenv("FIREBASE_LOCAL_CREDS_FILE", str(project_root / "firebase_local_only.json"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Configuration
    # Default origins; can be overridden by ALLOW_ORIGINS (comma-separated)
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "https://daezend.com",
        "https://www.daezend.com"
    ]
    ALLOW_ORIGINS: Optional[str] = os.getenv("ALLOW_ORIGINS")
    
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

    # Redis / Queueing
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    REDIS_NAMESPACE: str = os.getenv("REDIS_NAMESPACE", "runpod")
    RUNPOD_MAX_CONCURRENCY: int = int(os.getenv("RUNPOD_MAX_CONCURRENCY", "1"))
    RUNPOD_MAX_CONCURRENCY_VC: int = int(os.getenv("RUNPOD_MAX_CONCURRENCY_VC", os.getenv("RUNPOD_MAX_CONCURRENCY", "1")))
    RUNPOD_MAX_CONCURRENCY_TTS: int = int(os.getenv("RUNPOD_MAX_CONCURRENCY_TTS", os.getenv("RUNPOD_MAX_CONCURRENCY", "1")))
    REDIS_USE_TLS: bool = os.getenv("REDIS_USE_TLS", "true").lower() == "true"
    REDIS_STREAM_NAME: str = os.getenv("REDIS_STREAM_NAME", "runpod:jobs")
    REDIS_STREAM_NAME_VC: str = os.getenv("REDIS_STREAM_NAME_VC", "runpod:jobs:vc")
    REDIS_STREAM_NAME_TTS: str = os.getenv("REDIS_STREAM_NAME_TTS", "runpod:jobs:tts")
    REDIS_CONSUMER_GROUP: str = os.getenv("REDIS_CONSUMER_GROUP", "runpod-consumers")
    REDIS_CONSUMER_NAME: str = os.getenv("REDIS_CONSUMER_NAME", "worker-1")
    REDIS_DLP_STREAM: str = os.getenv("REDIS_DLP_STREAM", "runpod:dlq")
    REDIS_MAX_RETRIES: int = int(os.getenv("REDIS_MAX_RETRIES", "3"))
    REDIS_RETRY_BASE_DELAY_SECONDS: int = int(os.getenv("REDIS_RETRY_BASE_DELAY_SECONDS", "5"))
    REDIS_JOB_TTL_SECONDS: int = int(os.getenv("REDIS_JOB_TTL_SECONDS", "604800"))  # 7 days
    FIREBASE_WEBHOOK_SECRET: Optional[str] = os.getenv("FIREBASE_WEBHOOK_SECRET")
    
    # Security / Auth
    DAEZEND_API_SHARED_SECRET: Optional[str] = os.getenv("DAEZEND_API_SHARED_SECRET")
    SECURITY_ENABLE_HMAC: bool = os.getenv("SECURITY_ENABLE_HMAC", "false").lower() == "true"
    SECURITY_ENABLE_FIREBASE_AUTH: bool = os.getenv("SECURITY_ENABLE_FIREBASE_AUTH", "false").lower() == "true"
    SECURITY_ENABLE_APP_CHECK: bool = os.getenv("SECURITY_ENABLE_APP_CHECK", "false").lower() == "true"
    HMAC_MAX_SKEW_SECONDS: int = int(os.getenv("HMAC_MAX_SKEW_SECONDS", "300"))
    IDEMPOTENCY_TTL_SECONDS: int = int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "86400"))
    
    # Rate limiting
    RATE_LIMIT_DEFAULT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_DEFAULT_PER_MINUTE", "60"))
    RATE_LIMIT_CLONE_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_CLONE_PER_MINUTE", "5"))
    RATE_LIMIT_TTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_TTS_PER_MINUTE", "10"))
    
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
    logger.info(f"📋 FIREBASE_CREDENTIALS: Loaded successfully")
else:
    logger.warning("⚠️ RUNPOD_SECRET_Firebase environment variable is not set!")
    logger.warning("⚠️ Firebase functionality will not work without proper credentials")
logger.info(f"📋 RunPod config valid: {settings.validate_runpod_config()}")
logger.info(f"📋 Firebase config valid: {settings.validate_firebase_config()}")
if settings.ALLOW_ORIGINS:
    try:
        parsed = [o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o.strip()]
        if parsed:
            settings.CORS_ORIGINS = parsed
    except Exception:
        pass
logger.info(f"📋 CORS origins: {settings.CORS_ORIGINS}")
logger.info(f"📋 Security (HMAC): {'ENABLED' if settings.SECURITY_ENABLE_HMAC else 'disabled'}")
logger.info(f"📋 Security (Firebase Auth): {'ENABLED' if settings.SECURITY_ENABLE_FIREBASE_AUTH else 'disabled'}")
logger.info(f"📋 Security (App Check): {'ENABLED' if settings.SECURITY_ENABLE_APP_CHECK else 'disabled'}")
logger.info("🔍 ===== END CONFIGURATION DEBUG =====") 
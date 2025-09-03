import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the api-app directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class Settings:
    # RunPod Configuration
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    VC_CB_ENDPOINT_ID: str = os.getenv("VC_CB_ENDPOINT_ID", "")
    TTS_CB_ENDPOINT_ID: str = os.getenv("TTS_CB_ENDPOINT_ID", "")
    RUNPOD_MAX_CONCURRENCY_VC: int = int(os.getenv("RUNPOD_MAX_CONCURRENCY_VC", "2"))
    RUNPOD_MAX_CONCURRENCY_TTS: int = int(os.getenv("RUNPOD_MAX_CONCURRENCY_TTS", "2"))

    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "")
    FIREBASE_CREDENTIALS: Optional[str] = os.getenv("RUNPOD_SECRET_Firebase")  # For RunPod uploads
    
    # Local Firebase credentials file for library display (API app only)
    # Use absolute path to project root
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    FIREBASE_LOCAL_CREDS_FILE: str = os.getenv("FIREBASE_LOCAL_CREDS_FILE", str(project_root / "firebase_local_only.json"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PUBLIC_API_BASE_URL: Optional[str] = os.getenv("PUBLIC_API_BASE_URL")
    
    # CORS Configuration
    # Default origins; can be overridden by ALLOW_ORIGINS (comma-separated)
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "https://daezend.com",
        "https://www.daezend.com",
        "https://daezend.app",
        "https://www.daezend.app"
    ]
    ALLOW_ORIGINS: Optional[str] = os.getenv("ALLOW_ORIGINS")
    
    # Storage Configuration
    LOCAL_STORAGE_ENABLED: bool = os.getenv("LOCAL_STORAGE_ENABLED", "False").lower() == "true"
    FIREBASE_STORAGE_ENABLED: bool = os.getenv("FIREBASE_STORAGE_ENABLED", "True").lower() == "true"
    
    # File Size Limits
    MAX_AUDIO_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_PROFILE_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Security Configuration
    SECURITY_ENABLE_HMAC: bool = os.getenv("SECURITY_ENABLE_HMAC", "True").lower() == "true"
    SECURITY_ENABLE_FIREBASE_AUTH: bool = os.getenv("SECURITY_ENABLE_FIREBASE_AUTH", "False").lower() == "true"
    SECURITY_ENABLE_APP_CHECK: bool = os.getenv("SECURITY_ENABLE_APP_CHECK", "False").lower() == "true"
    DAEZEND_API_SHARED_SECRET: Optional[str] = os.getenv("DAEZEND_API_SHARED_SECRET")
    HMAC_MAX_SKEW_SECONDS: int = int(os.getenv("HMAC_MAX_SKEW_SECONDS", "300"))  # 5 minutes
    IDEMPOTENCY_TTL_SECONDS: int = int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "3600"))  # 1 hour
    
    def get_firebase_bucket_name(self) -> str:
        """Get Firebase bucket name with fallback logic."""
        if self.FIREBASE_STORAGE_BUCKET:
            return self.FIREBASE_STORAGE_BUCKET
        
        # Fallback to environment variable
        fallback = os.getenv("FIREBASE_STORAGE_BUCKET")
        if fallback:
            return fallback
        
        # Final fallback
        return "daezend-audio-storage"

# Create settings instance
settings = Settings()
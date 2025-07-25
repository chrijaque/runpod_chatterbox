import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Firebase Configuration
    FIREBASE_CREDENTIALS_FILE: str = "firebase_creds.json"
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "godnathistorie-a25fa.appspot.com")
    
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
    
    # RunPod Configuration
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    RUNPOD_ENDPOINT_ID: Optional[str] = os.getenv("RUNPOD_ENDPOINT_ID")
    TTS_ENDPOINT_ID: Optional[str] = os.getenv("TTS_ENDPOINT_ID")

    @classmethod
    def validate_firebase_config(cls) -> bool:
        """Validate Firebase configuration"""
        if not cls.FIREBASE_STORAGE_ENABLED:
            return True
        if not Path(cls.FIREBASE_CREDENTIALS_FILE).exists():
            return False
        if cls.FIREBASE_STORAGE_BUCKET == "your-project-id.appspot.com":
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
            cls.RUNPOD_ENDPOINT_ID,
            cls.TTS_ENDPOINT_ID
        ])

    @classmethod
    def get_missing_config(cls) -> list:
        """Get list of missing configuration items"""
        missing = []
        
        if cls.FIREBASE_STORAGE_ENABLED and not cls.validate_firebase_config():
            missing.append("Firebase configuration (check firebase_creds.json and FIREBASE_STORAGE_BUCKET)")
        
        if not cls.validate_runpod_config():
            if not cls.RUNPOD_API_KEY:
                missing.append("RUNPOD_API_KEY")
            if not cls.RUNPOD_ENDPOINT_ID:
                missing.append("RUNPOD_ENDPOINT_ID")
            if not cls.TTS_ENDPOINT_ID:
                missing.append("TTS_ENDPOINT_ID")
        
        return missing

settings = Settings() 
from pydantic import BaseModel
from typing import Optional, Dict, Any

class VoiceCloneRequest(BaseModel):
    user_id: str
    name: str
    audio_data: Optional[str] = None  # Base64 encoded audio
    audio_path: Optional[str] = None  # Firebase Storage path (audio/voices/.../recorded/...)
    audio_format: str = "wav"
    language: str = "en"
    is_kids_voice: bool = False
    model_type: str = "chatterbox"  # Model selection (chatterbox only)
    profile_id: Optional[str] = None

class VoiceCloneResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    voice_id: Optional[str] = None
    profile_path: Optional[str] = None
    recorded_audio_path: Optional[str] = None  # New: path to recorded audio
    sample_audio_path: Optional[str] = None    # Renamed from audio_path
    generation_time: Optional[float] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TTSGenerateRequest(BaseModel):
    user_id: str
    voice_id: str
    text: str
    profile_base64: Optional[str] = None
    profile_path: Optional[str] = None
    story_id: str
    language: str = "en"
    story_type: str = "user"
    is_kids_voice: bool = False
    model_type: str = "chatterbox"  # Model selection (chatterbox only)
    callback_url: Optional[str] = None

class TTSGenerateResponse(BaseModel):
    status: str
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    sample_file: Optional[str] = None
    embedding_file: Optional[str] = None
    created_date: Optional[int] = None
    language: Optional[str] = None
    is_kids_voice: Optional[bool] = None

class TTSGeneration(BaseModel):
    generation_id: str
    voice_id: str
    voice_name: Optional[str] = None
    text_input: Optional[str] = None
    audio_file: Optional[str] = None
    created_date: Optional[int] = None
    language: Optional[str] = None
    story_type: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: Optional[str] = None
    timestamp: Optional[str] = None
    service: Optional[str] = None
    firebase_connected: Optional[bool] = None

class DebugResponse(BaseModel):
    status: str
    message: Optional[str] = None
    directories: Optional[Dict[str, Any]] = None
    current_working_directory: Optional[str] = None
    timestamp: Optional[str] = None
    firebase_connected: Optional[bool] = None
    firebase_storage_usage: Optional[Dict[str, Any]] = None
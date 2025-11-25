from pydantic import BaseModel
from typing import Optional, Dict, Any

class VoiceCloneRequest(BaseModel):
    user_id: str
    name: str
    audio_data: Optional[str] = None  # Base64 encoded audio
    audio_path: Optional[str] = None  # R2 path (audio/voices/.../recorded/...)
    audio_format: str = "wav"
    language: str = "en"
    is_kids_voice: bool = False
    model_type: str = "chatterbox"  # Model selection (chatterbox only)
    profile_id: Optional[str] = None
    # NEW: Pass-through identifiers and naming hints from the app
    voice_id: Optional[str] = None
    profile_filename: Optional[str] = None
    sample_filename: Optional[str] = None
    output_basename: Optional[str] = None
    callback_url: Optional[str] = None
    # Geo/bucket routing (optional)
    bucket_name: Optional[str] = None
    country_code: Optional[str] = None

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
    # Naming hints (optional)
    story_name: Optional[str] = None
    output_basename: Optional[str] = None
    output_filename: Optional[str] = None
    voice_name: Optional[str] = None  # Voice name for metadata and Firebase uploads
    # Geo/bucket routing (optional)
    bucket_name: Optional[str] = None
    country_code: Optional[str] = None  # e.g., 'AU'

class TTSGenerateResponse(BaseModel):
    status: str
    job_id: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TTSErrorCallbackRequest(BaseModel):
    story_id: str
    error: str
    user_id: Optional[str] = None
    voice_id: Optional[str] = None
    error_details: Optional[str] = None
    job_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TTSErrorCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class TTSSuccessCallbackRequest(BaseModel):
    story_id: str
    user_id: Optional[str] = None
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    audio_url: Optional[str] = None
    storage_path: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TTSSuccessCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class VoiceCloneSuccessCallbackRequest(BaseModel):
    status: Optional[str] = None
    user_id: Optional[str] = None
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    language: Optional[str] = None
    profile_path: Optional[str] = None
    sample_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VoiceCloneSuccessCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class VoiceCloneErrorCallbackRequest(BaseModel):
    status: Optional[str] = None
    user_id: Optional[str] = None
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    language: Optional[str] = None
    error: str
    metadata: Optional[Dict[str, Any]] = None

class VoiceCloneErrorCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None

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
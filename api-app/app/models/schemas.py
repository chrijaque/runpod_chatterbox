from pydantic import BaseModel
from typing import Optional, Dict, Any

class VoiceCloneRequest(BaseModel):
    name: str
    audio_data: str  # Base64 encoded audio
    audio_format: str = "wav"
    language: str = "en"
    is_kids_voice: bool = False

class VoiceCloneResponse(BaseModel):
    status: str
    profile_path: Optional[str] = None
    recorded_audio_path: Optional[str] = None  # New: path to recorded audio
    sample_audio_path: Optional[str] = None    # Renamed from audio_path
    metadata: Optional[Dict[str, Any]] = None

class TTSGenerateRequest(BaseModel):
    voice_id: str
    text: str
    profile_base64: str
    language: str = "en"
    story_type: str = "user"
    is_kids_voice: bool = False

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
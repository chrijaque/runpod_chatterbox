from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Voice Management Models
class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    sample_file: Optional[str] = None
    profile_file: Optional[str] = None
    created_date: Optional[float] = None
    has_profile: bool = False
    has_metadata: bool = False
    firebase_url: Optional[str] = None
    language: Optional[str] = None
    is_kids_voice: Optional[bool] = None
    user_id: Optional[str] = None
    visibility: Optional[str] = None

class VoiceSaveRequest(BaseModel):
    voice_id: str
    voice_name: str
    audio_file: str  # Base64 encoded audio
    profile_file: Optional[str] = None  # Base64 encoded profile
    template_message: str = ""

class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning from main app"""
    title: str = Field(..., description="Voice name/title")
    voices: List[str] = Field(..., description="List of base64 encoded WAV audio blobs")
    visibility: str = Field(default="private", description="Voice visibility (private/public)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VoiceMetadata(BaseModel):
    """Voice metadata structure"""
    language: str = Field(default="en", description="Voice language code")
    isKidsVoice: bool = Field(default=False, description="Whether this is a kids voice")
    userId: str = Field(..., description="User ID who created the voice")
    createdAt: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    title: str = Field(..., description="Voice title/name")
    visibility: str = Field(default="private", description="Voice visibility")

class VoiceCloneResponse(BaseModel):
    """Response model for voice cloning"""
    status: str
    voice_id: str
    voice_name: str
    audio_base64: Optional[str] = None
    profile_base64: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# TTS Models
class TTSGeneration(BaseModel):
    file_id: str
    voice_id: str
    voice_name: str
    file_path: Optional[str] = None
    created_date: Optional[float] = None
    timestamp: Optional[str] = None
    file_size: Optional[int] = None
    firebase_url: Optional[str] = None

class TTSGenerateRequest(BaseModel):
    """Request model for TTS generation"""
    voice_id: str
    text: str
    responseFormat: str = "base64"
    language: str = "en"
    is_kids_voice: bool = False
    story_type: str = "user"  # "user" or "app"

class TTSGenerateResponse(BaseModel):
    """Response model for TTS generation"""
    status: str
    generation_id: str
    voice_id: str
    text: str
    audio_base64: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TTSSaveRequest(BaseModel):
    voice_id: str
    audio_base64: str
    text_input: str
    generation_time: float
    sample_rate: int
    audio_shape: str  # JSON string of audio shape

class TTSSaveResponse(BaseModel):
    status: str
    message: str
    file_path: Optional[str] = None
    firebase_url: Optional[str] = None

class StoryInfo(BaseModel):
    """Story information model"""
    generation_id: str
    audio_files: List[str] = []
    created_at: Optional[str] = None

class StoriesResponse(BaseModel):
    """Response model for stories listing"""
    status: str
    language: str
    story_type: str
    stories: List[StoryInfo]
    total_stories: int
    shared_access: bool

class StoryAudioResponse(BaseModel):
    """Response model for story audio"""
    status: str
    language: str
    story_type: str
    generation_id: str
    story_url: str
    shared_access: bool

class StoryLanguagesResponse(BaseModel):
    """Response model for story languages"""
    status: str
    languages: List[Dict[str, str]]
    total_languages: int

# API Response Models
class VoiceLibraryResponse(BaseModel):
    status: str
    voices: List[VoiceInfo]
    total_voices: int

class TTSGenerationsResponse(BaseModel):
    status: str
    total_generations: int
    generations: List[TTSGeneration]

class HealthResponse(BaseModel):
    status: str
    service: str
    firebase_connected: bool
    timestamp: str

class DebugResponse(BaseModel):
    status: str
    firebase_connected: bool
    directories: Dict[str, Any]
    current_working_directory: str
    timestamp: str
    firebase_storage_usage: Optional[Dict[str, Any]] = None

# Error Models
class ErrorResponse(BaseModel):
    status: str
    error: str
    message: str

# Success Models
class SuccessResponse(BaseModel):
    status: str
    message: str 
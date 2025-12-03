from pydantic import BaseModel, model_validator
from typing import Optional, Dict, Any, List

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
    # Deprecated: Geo/bucket routing - no longer used (all storage is R2)
    bucket_name: Optional[str] = None  # Deprecated - ignored
    country_code: Optional[str] = None  # Deprecated - ignored

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

# LLM Generation Schemas
class LLMGenerateRequest(BaseModel):
    user_id: str
    story_id: str
    messages: Optional[List[Dict[str, str]]] = None  # List of message dicts with 'role' and 'content' (required for single-step)
    temperature: float = 0.7
    max_tokens: int = 6000
    language: Optional[str] = None
    genre: Optional[str] = None
    age_range: Optional[str] = None
    callback_url: Optional[str] = None
    # Two-step workflow support
    workflow_type: Optional[str] = None  # 'two-step' or 'single-step'
    outline_messages: Optional[List[Dict[str, str]]] = None
    story_messages: Optional[List[Dict[str, str]]] = None
    outline_max_tokens: Optional[int] = 5000  # Max tokens for outline/beat generation (two-step workflow)
    expansion_max_tokens: Optional[int] = 4000  # Max tokens for Step 3 expansion (if story < 8500 chars)
    finetune_max_tokens: Optional[int] = 4000  # Max tokens for Step 4 finetuning (rewrite duplicated dialogue/descriptions)
    mode: Optional[str] = "sensual"  # NSFW mode for tone (soft, sensual, hardcore, extreme)
    
    @model_validator(mode='after')
    def validate_messages(self):
        """Validate that either messages OR (outline_messages AND story_messages) are provided."""
        is_two_step = self.workflow_type == "two-step" or (self.outline_messages and self.story_messages)
        
        if is_two_step:
            if not self.outline_messages or not self.story_messages:
                raise ValueError("For two-step workflow, both outline_messages and story_messages are required")
        else:
            if not self.messages:
                raise ValueError("messages is required for single-step workflow")
        
        return self

class LLMGenerateResponse(BaseModel):
    status: str  # 'queued', 'success', 'error'
    job_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMSuccessCallbackRequest(BaseModel):
    story_id: str
    user_id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None

class LLMSuccessCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None

class LLMErrorCallbackRequest(BaseModel):
    story_id: str
    user_id: Optional[str] = None
    error: str
    error_details: Optional[str] = None
    job_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMErrorCallbackResponse(BaseModel):
    success: bool
    error: Optional[str] = None
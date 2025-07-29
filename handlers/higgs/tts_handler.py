import runpod
import time
import os
import tempfile
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from google.cloud import storage
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Higgs Audio components
try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    HIGGS_AVAILABLE = True
    logger.info("✅ Successfully imported Higgs Audio components")
except ImportError as e:
    HIGGS_AVAILABLE = False
    logger.error(f"❌ Could not import Higgs Audio components: {e}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# Model paths for Higgs Audio (from network volume)
MODEL_PATH = "/runpod-volume/higgs_audio_generation"
AUDIO_TOKENIZER_PATH = "/runpod-volume/higgs_audio_tokenizer"
HUBERT_PATH = "/runpod-volume/hubert_base"

# Set cache directory to use network volume
import os
os.environ["HF_HOME"] = "/runpod-volume"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume"

# Local directory paths
TTS_GENERATED_DIR = Path("/tts_generated")
TEMP_VOICE_DIR = Path("/temp_voice")

# Pre-load Higgs Audio models at module level (avoids runtime initialization)
logger.info("🔧 Pre-loading Higgs Audio models for TTS...")
model = None
serve_engine = None

try:
    if HIGGS_AVAILABLE:
        logger.info("🔍 Attempting to pre-load Higgs Audio serve engine for TTS...")
        logger.info(f"   - Model path: {MODEL_PATH}")
        logger.info(f"   - Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
        logger.info(f"   - Device: cuda")
        
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device="cuda"
        )
        
        model = serve_engine  # For compatibility with existing code
        logger.info("✅ Higgs Audio models pre-loaded successfully for TTS")
        logger.info(f"✅ Serve engine type: {type(serve_engine)}")
    else:
        logger.error("❌ Higgs Audio components not available for pre-loading")
        serve_engine = None
        model = None
        
except Exception as e:
    logger.error(f"❌ Failed to pre-load Higgs Audio models for TTS: {e}")
    logger.error(f"❌ Error type: {type(e)}")
    import traceback
    logger.error(f"❌ Full pre-load traceback: {traceback.format_exc()}")
    serve_engine = None
    model = None

def initialize_firebase():
    """Initialize Firebase Storage client"""
    global storage_client, bucket
    
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        # Get Firebase credentials from environment
        firebase_creds = os.getenv('RUNPOD_SECRET_Firebase')
        bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')
        
        if not firebase_creds or not bucket_name:
            logger.error("❌ Firebase credentials or bucket name not found in environment")
            return False
        
        # Parse Firebase credentials
        import json
        creds_dict = json.loads(firebase_creds)
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        
        # Initialize storage client
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        
        logger.info(f"✅ Firebase initialized successfully")
        logger.info(f"✅ Connected to bucket: {bucket_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase: {e}")
        return False

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """Upload data to Firebase Storage"""
    global bucket
    
    if not bucket:
        if not initialize_firebase():
            return None
    
    try:
        blob = bucket.blob(destination_blob_name)
        
        # Set metadata if provided
        if metadata:
            blob.metadata = metadata
        
        # Upload the data
        blob.upload_from_string(data, content_type=content_type)
        
        # Make the blob publicly accessible
        blob.make_public()
        
        logger.info(f"✅ Uploaded to Firebase: {destination_blob_name}")
        return blob.public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload to Firebase: {e}")
        return None

def initialize_model():
    """Model is pre-loaded, no initialization needed"""
    global model, serve_engine
    
    logger.info("🔍 Checking pre-loaded Higgs Audio model for TTS...")
    
    if model is not None and serve_engine is not None:
        logger.info("✅ Model already pre-loaded")
        return model
    else:
        logger.error("❌ Model not pre-loaded")
        raise RuntimeError("Higgs Audio model not available - pre-loading failed")

def load_voice_profile(profile_base64: str) -> Optional[np.ndarray]:
    """Load voice profile from base64 data"""
    try:
        # Decode base64 data
        profile_data = base64.b64decode(profile_base64)
        
        # Convert to numpy array
        voice_profile = np.frombuffer(profile_data, dtype=np.float32)
        
        logger.info(f"✅ Voice profile loaded: shape={voice_profile.shape}")
        return voice_profile
        
    except Exception as e:
        logger.error(f"❌ Failed to load voice profile: {e}")
        return None

def chunk_text(text: str, chunk_method: str = "word", chunk_max_word_num: int = 200) -> List[str]:
    """Split text into chunks for long-form TTS"""
    
    if chunk_method == "word":
        # Simple word-based chunking
        words = text.split()
        chunks = []
        current_chunk = []
        word_count = 0
        
        for word in words:
            current_chunk.append(word)
            word_count += 1
            
            if word_count >= chunk_max_word_num:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    elif chunk_method == "speaker":
        # Speaker-based chunking for dialogue
        # Split by speaker tags like [SPEAKER0], [SPEAKER1]
        import re
        speaker_pattern = r'\[SPEAKER\d+\]'
        parts = re.split(speaker_pattern, text)
        speakers = re.findall(speaker_pattern, text)
        
        chunks = []
        for i, (speaker, part) in enumerate(zip(speakers, parts[1:])):  # Skip first empty part
            if part.strip():
                chunks.append(f"{speaker} {part.strip()}")
        
        return chunks
    
    else:
        # Default: return as single chunk
        return [text]

def generate_tts_chunk(text: str, voice_profile: np.ndarray, temperature: float = 0.3) -> Optional[bytes]:
    """Generate TTS for a single chunk"""
    global serve_engine
    
    if not serve_engine:
        logger.error("❌ Serve engine not pre-loaded")
        return None
    
    try:
        logger.info(f"🎵 Generating TTS chunk: {len(text)} characters")
        
        # Create system prompt
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        # Create messages for TTS generation
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=text)
        ]
        
        # Generate audio
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        # Convert audio to bytes
        import torch
        audio_tensor = torch.from_numpy(response.audio)
        
        # Save to temporary file and read as bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            import torchaudio
            torchaudio.save(temp_file.name, audio_tensor.unsqueeze(0), response.sampling_rate)
            
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
        
        logger.info(f"✅ TTS chunk generated: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"❌ Failed to generate TTS chunk: {e}")
        return None

def stitch_audio_chunks(chunks: List[bytes]) -> bytes:
    """Stitch multiple audio chunks into a single audio file"""
    try:
        from pydub import AudioSegment
        import io
        
        if not chunks:
            return b""
        
        # Convert first chunk to AudioSegment
        audio = AudioSegment.from_wav(io.BytesIO(chunks[0]))
        
        # Add silence between chunks (200ms)
        silence = AudioSegment.silent(duration=200)
        
        # Stitch remaining chunks
        for chunk_bytes in chunks[1:]:
            chunk_audio = AudioSegment.from_wav(io.BytesIO(chunk_bytes))
            audio = audio + silence + chunk_audio
        
        # Export as MP3
        output = io.BytesIO()
        audio.export(output, format="mp3", bitrate="96k")
        output.seek(0)
        
        return output.read()
        
    except Exception as e:
        logger.error(f"❌ Failed to stitch audio chunks: {e}")
        return b""

def generate_tts_story(voice_id: str, text: str, profile_base64: str, language: str = "en", 
                       story_type: str = "user", is_kids_voice: bool = False) -> Optional[Dict]:
    """Generate TTS story using Higgs Audio"""
    global serve_engine
    
    logger.info("📖 ===== HIGGS AUDIO TTS STORY GENERATION =====")
    logger.info(f"🔍 Parameters: voice_id={voice_id}, language={language}, story_type={story_type}")
    
    try:
        # Check if model is pre-loaded
        if not serve_engine:
            logger.error("❌ Serve engine not pre-loaded")
            return {"status": "error", "message": "Higgs Audio model not available"}
        
        # Load voice profile
        voice_profile = load_voice_profile(profile_base64)
        if voice_profile is None:
            return {"status": "error", "message": "Failed to load voice profile"}
        
        start_time = time.time()
        
        # Determine chunking method based on text length
        if len(text) > 1000:
            # Long-form TTS with chunking
            logger.info("📦 Using long-form TTS with chunking")
            chunks = chunk_text(text, chunk_method="word", chunk_max_word_num=200)
            logger.info(f"📦 Split into {len(chunks)} chunks")
            
            # Generate audio for each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🔄 Generating chunk {i+1}/{len(chunks)}")
                chunk_audio = generate_tts_chunk(chunk, voice_profile, temperature=0.3)
                if chunk_audio:
                    audio_chunks.append(chunk_audio)
                else:
                    logger.error(f"❌ Failed to generate chunk {i+1}")
                    return {"status": "error", "message": f"Failed to generate chunk {i+1}"}
            
            # Stitch chunks together
            logger.info("🔗 Stitching audio chunks...")
            final_audio = stitch_audio_chunks(audio_chunks)
            
        else:
            # Single-shot TTS
            logger.info("🎵 Using single-shot TTS")
            final_audio = generate_tts_chunk(text, voice_profile, temperature=0.3)
            if not final_audio:
                return {"status": "error", "message": "Failed to generate TTS"}
        
        generation_time = time.time() - start_time
        logger.info(f"✅ TTS generation completed in {generation_time:.2f}s")
        
        # Upload to Firebase
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = f"TTS_{voice_id}_{timestamp}.mp3"
        tts_path = f"audio/stories/{language}/{story_type}/{tts_filename}"
        
        tts_metadata = {
            "voice_id": voice_id,
            "voice_name": voice_id.replace('voice_', ''),
            "created_date": str(int(start_time)),
            "language": language,
            "story_type": story_type,
            "is_kids_voice": str(is_kids_voice),
            "file_type": "tts_story",
            "model": "higgs_audio_v2",
            "format": "96k_mp3",
            "text_length": str(len(text)),
            "generation_time": str(generation_time),
            "timestamp": timestamp
        }
        
        tts_url = upload_to_firebase(
            final_audio,
            tts_path,
            "audio/mpeg",
            tts_metadata
        )
        
        logger.info("✅ TTS story uploaded to Firebase successfully")
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "audio_path": tts_path,
            "audio_url": tts_url,
            "generation_time": generation_time,
            "model": "higgs_audio_v2",
            "metadata": tts_metadata
        }
        
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"TTS generation failed: {str(e)}"}

def handler(event):
    """Handle TTS generation requests using Higgs Audio"""
    global model, serve_engine
    
    logger.info("🚀 ===== HIGGS AUDIO TTS HANDLER =====")
    logger.info(f"📥 Received event: {type(event)}")
    
    input_data = event.get('input', {})
    logger.info(f"📥 Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
    
    # Extract parameters
    voice_id = input_data.get('voice_id')
    text = input_data.get('text')
    profile_base64 = input_data.get('profile_base64')
    language = input_data.get('language', 'en')
    story_type = input_data.get('story_type', 'user')
    is_kids_voice = input_data.get('is_kids_voice', False)
    
    if not voice_id or not text or not profile_base64:
        logger.error("❌ Missing required parameters: voice_id, text, and profile_base64")
        return {"status": "error", "message": "voice_id, text, and profile_base64 are required"}
    
    try:
        logger.info(f"📋 TTS request: voice_id={voice_id}, text_length={len(text)}")
        
        # Generate TTS story
        result = generate_tts_story(
            voice_id=voice_id,
            text=text,
            profile_base64=profile_base64,
            language=language,
            story_type=story_type,
            is_kids_voice=is_kids_voice
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"TTS generation failed: {str(e)}"} 
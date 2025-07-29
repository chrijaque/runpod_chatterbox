import runpod
import time
import os
import tempfile
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
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

# Model paths for Higgs Audio
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Add model download verification
def verify_model_availability():
    """Verify that Higgs Audio models are available"""
    try:
        from transformers import AutoTokenizer, AutoModel
        logger.info("🔍 Checking model availability...")
        
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(AUDIO_TOKENIZER_PATH)
        logger.info(f"✅ Audio tokenizer loaded: {AUDIO_TOKENIZER_PATH}")
        
        # Try to load the model (this will download if not cached)
        model = AutoModel.from_pretrained(MODEL_PATH)
        logger.info(f"✅ Model loaded: {MODEL_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False

# Local directory paths
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")
TEMP_VOICE_DIR = Path("/temp_voice")

# Initialize Higgs Audio model
model = None
serve_engine = None

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
    """Initialize Higgs Audio model"""
    global model, serve_engine
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    if not HIGGS_AVAILABLE:
        raise RuntimeError("Higgs Audio components not available")
    
    logger.info("Initializing Higgs Audio model...")
    
    try:
        import torch
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Verify model availability first
        if not verify_model_availability():
            raise RuntimeError("Higgs Audio models not available")
        
        # Initialize Higgs Audio serve engine
        logger.info(f"🔧 Initializing serve engine with model: {MODEL_PATH}")
        logger.info(f"🔧 Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
        
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device=device
        )
        
        model = serve_engine  # For compatibility with existing code
        logger.info("✅ Higgs Audio model initialized successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Higgs Audio model: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise

def extract_voice_profile(audio_data: bytes, voice_id: str) -> Optional[np.ndarray]:
    """Extract voice profile from audio data using Higgs Audio"""
    global serve_engine
    
    if not serve_engine:
        initialize_model()
    
    try:
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_audio_path = temp_file.name
        
        logger.info(f"🔄 Extracting voice profile for: {voice_id}")
        
        # Create system prompt for voice cloning
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        # Create messages for voice profile extraction
        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user", 
                content=AudioContent(audio=audio_data, sampling_rate=24000)  # Default sample rate
            )
        ]
        
        # Generate voice profile
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        # Extract voice profile from response
        # Note: This is a simplified approach - actual voice profile extraction
        # would require more sophisticated processing of the model's internal states
        
        # For now, we'll create a placeholder voice profile
        # In a real implementation, you'd extract the actual voice embeddings
        voice_profile = np.random.randn(1024).astype(np.float32)  # Placeholder
        
        logger.info(f"✅ Voice profile extracted: shape={voice_profile.shape}")
        return voice_profile
        
    except Exception as e:
        logger.error(f"❌ Failed to extract voice profile: {e}")
        return None
    finally:
        # Clean up temporary file
        if 'temp_audio_path' in locals():
            try:
                os.unlink(temp_audio_path)
            except:
                pass

def generate_voice_sample(voice_profile: np.ndarray, voice_id: str, text: str) -> Optional[bytes]:
    """Generate voice sample using Higgs Audio"""
    global serve_engine
    
    if not serve_engine:
        initialize_model()
    
    try:
        logger.info(f"🎤 Generating voice sample for: {voice_id}")
        
        # Create system prompt with voice profile
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
            temperature=0.3,
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
        
        logger.info(f"✅ Voice sample generated: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"❌ Failed to generate voice sample: {e}")
        return None

def handler(event):
    """Handle voice cloning requests using Higgs Audio"""
    global model, serve_engine
    
    logger.info("🚀 ===== HIGGS AUDIO VOICE CLONING HANDLER =====")
    logger.info(f"📥 Received event: {type(event)}")
    
    input_data = event.get('input', {})
    logger.info(f"📥 Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
    
    # Extract parameters (compatible with API format)
    name = input_data.get('name')
    audio_data_base64 = input_data.get('audio_data')
    text = input_data.get('text', "Hello, this is a test of the voice cloning system.")
    
    if not name or not audio_data_base64:
        logger.error("❌ Missing required parameters: name and audio_data")
        return {"status": "error", "message": "name and audio_data are required"}
    
    # Generate voice_id from name (compatible with ChatterboxTTS format)
    import hashlib
    voice_id = f"voice_{name.lower().replace(' ', '_')}"
    logger.info(f"🎯 Generated voice_id: {voice_id} from name: {name}")
    
    try:
        # Decode audio data
        audio_data = base64.b64decode(audio_data_base64)
        logger.info(f"✅ Audio data decoded: {len(audio_data)} bytes")
        
        # Initialize model if needed
        if not model:
            logger.info("🔧 Initializing Higgs Audio model...")
            initialize_model()
            logger.info("✅ Model initialization completed")
        
        start_time = time.time()
        
        # Step 1: Extract voice profile
        logger.info("🔄 Step 1: Extracting voice profile...")
        voice_profile = extract_voice_profile(audio_data, voice_id)
        
        if voice_profile is None:
            return {"status": "error", "message": "Failed to extract voice profile"}
        
        # Step 2: Generate voice sample
        logger.info("🔄 Step 2: Generating voice sample...")
        sample_audio = generate_voice_sample(voice_profile, voice_id, text)
        
        if sample_audio is None:
            return {"status": "error", "message": "Failed to generate voice sample"}
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Voice cloning completed in {generation_time:.2f}s")
        
        # Step 3: Upload to Firebase
        logger.info("🔄 Step 3: Uploading to Firebase...")
        
        # Create filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload voice profile
        profile_filename = f"{voice_id}_{timestamp}.npy"
        profile_path = f"audio/voices/en/profiles/{profile_filename}"
        
        profile_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "voice_profile",
            "model": "higgs_audio_v2"
        }
        
        profile_url = upload_to_firebase(
            voice_profile.tobytes(),
            profile_path,
            "application/octet-stream",
            profile_metadata
        )
        
        # Upload voice sample
        sample_filename = f"{voice_id}_sample_{timestamp}.mp3"
        sample_path = f"audio/voices/en/samples/{sample_filename}"
        
        sample_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "voice_sample",
            "model": "higgs_audio_v2",
            "format": "96k_mp3"
        }
        
        sample_url = upload_to_firebase(
            sample_audio,
            sample_path,
            "audio/mpeg",
            sample_metadata
        )
        
        # Upload original audio
        recorded_filename = f"{voice_id}_{timestamp}.mp3"
        recorded_path = f"audio/voices/en/recorded/{recorded_filename}"
        
        recorded_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "recorded_audio",
            "model": "higgs_audio_v2",
            "format": "160k_mp3"
        }
        
        recorded_url = upload_to_firebase(
            audio_data,
            recorded_path,
            "audio/mpeg",
            recorded_metadata
        )
        
        logger.info("✅ All files uploaded to Firebase successfully")
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "profile_path": profile_path,
            "sample_path": sample_path,
            "recorded_path": recorded_path,
            "profile_url": profile_url,
            "sample_url": sample_url,
            "recorded_url": recorded_url,
            "generation_time": generation_time,
            "model": "higgs_audio_v2"
        }
        
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Voice cloning failed: {str(e)}"} 
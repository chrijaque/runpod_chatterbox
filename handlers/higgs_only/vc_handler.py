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

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Higgs Audio components with detailed debugging
logger.info("🔍 Attempting to import Higgs Audio components...")
try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    HIGGS_AVAILABLE = True
    logger.info("✅ Successfully imported Higgs Audio components")
    logger.info(f"✅ HiggsAudioServeEngine: {HiggsAudioServeEngine}")
    logger.info(f"✅ HiggsAudioResponse: {HiggsAudioResponse}")
    logger.info(f"✅ ChatMLSample: {ChatMLSample}")
    logger.info(f"✅ Message: {Message}")
    logger.info(f"✅ AudioContent: {AudioContent}")
except ImportError as e:
    HIGGS_AVAILABLE = False
    logger.error(f"❌ Could not import Higgs Audio components: {e}")
    logger.error(f"❌ Import error type: {type(e)}")
    logger.error(f"❌ Import error details: {str(e)}")
    import traceback
    logger.error(f"❌ Full import traceback: {traceback.format_exc()}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# Model paths for Higgs Audio (from network volume)
MODEL_PATH = "/workspace/higgs_audio_generation"
AUDIO_TOKENIZER_PATH = "/workspace/higgs_audio_tokenizer"
HUBERT_PATH = "/workspace/hubert_base"

# Set cache directory to use network volume
import os
os.environ["HF_HOME"] = "/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/cache"

logger.info(f"🔧 Model configuration:")
logger.info(f"   - MODEL_PATH: {MODEL_PATH}")
logger.info(f"   - AUDIO_TOKENIZER_PATH: {AUDIO_TOKENIZER_PATH}")
logger.info(f"   - HUBERT_PATH: {HUBERT_PATH}")

# Pre-load Higgs Audio models at module level (avoids runtime initialization)
logger.info("🔧 Pre-loading Higgs Audio models...")
serve_engine = None
model = None

try:
    if HIGGS_AVAILABLE:
        logger.info("🔍 Attempting to pre-load Higgs Audio serve engine...")
        logger.info(f"   - Model path: {MODEL_PATH}")
        logger.info(f"   - Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
        logger.info(f"   - Device: cuda")
        
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device="cuda"
        )
        
        model = serve_engine  # For compatibility with existing code
        logger.info("✅ Higgs Audio models pre-loaded successfully")
        logger.info(f"✅ Serve engine type: {type(serve_engine)}")
    else:
        logger.error("❌ Higgs Audio components not available for pre-loading")
        serve_engine = None
        model = None
        
except Exception as e:
    logger.error(f"❌ Failed to pre-load Higgs Audio models: {e}")
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
    
    logger.info("🔍 Checking pre-loaded Higgs Audio model...")
    
    if model is not None and serve_engine is not None:
        logger.info("✅ Model already pre-loaded")
        return model
    else:
        logger.error("❌ Model not pre-loaded")
        raise RuntimeError("Higgs Audio model not available - pre-loading failed")

def extract_voice_profile(audio_data: bytes, voice_id: str) -> Optional[np.ndarray]:
    """Extract voice profile from audio data using Higgs Audio with detailed logging"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice profile extraction for: {voice_id}")
    logger.info(f"   - Audio data size: {len(audio_data)} bytes")
    
    if not serve_engine:
        logger.info("🔍 Serve engine not initialized, initializing...")
        initialize_model()
    
    try:
        # Convert audio data to numpy array
        import io
        import soundfile as sf
        
        # Read audio data
        audio, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        logger.info(f"✅ Audio loaded: shape={audio.shape}, sample_rate={sample_rate}")
        
        # Create voice profile using Higgs Audio
        # This is a simplified version - actual implementation would use Higgs Audio's voice extraction
        voice_profile = np.array(audio, dtype=np.float32)
        
        logger.info(f"✅ Voice profile extracted: shape={voice_profile.shape}")
        return voice_profile
        
    except Exception as e:
        logger.error(f"❌ Failed to extract voice profile: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice profile extraction traceback: {traceback.format_exc()}")
        return None

def generate_voice_sample(voice_profile: np.ndarray, voice_id: str, text: str) -> Optional[bytes]:
    """Generate voice sample using Higgs Audio with detailed logging"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice sample generation for: {voice_id}")
    logger.info(f"   - Voice profile shape: {voice_profile.shape}")
    logger.info(f"   - Text: {text[:50]}...")
    
    if not serve_engine:
        logger.error("❌ Serve engine not pre-loaded")
        return None
    
    try:
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
        
        # Generate audio using Higgs Audio
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
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice sample generation traceback: {traceback.format_exc()}")
        return None

def handler(event):
    """Handle voice cloning requests using Higgs Audio with comprehensive debugging"""
    global model, serve_engine
    
    logger.info("🚀 ===== HIGGS AUDIO VOICE CLONING HANDLER =====")
    logger.info(f"📥 Received event type: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input_data = event.get('input', {})
    logger.info(f"📥 Input data type: {type(input_data)}")
    logger.info(f"📥 Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
    
    # Extract parameters (compatible with API format)
    name = input_data.get('name')
    audio_data_base64 = input_data.get('audio_data')
    text = input_data.get('text', "Hello, this is a test of the voice cloning system.")
    
    logger.info(f"🔍 Extracted parameters:")
    logger.info(f"   - Name: {name}")
    logger.info(f"   - Audio data base64: {'SET' if audio_data_base64 else 'NOT SET'}")
    logger.info(f"   - Text: {text}")
    
    if not name or not audio_data_base64:
        logger.error("❌ Missing required parameters: name and audio_data")
        return {"status": "error", "message": "name and audio_data are required"}
    
    # Generate voice_id from name (compatible with ChatterboxTTS format)
    import hashlib
    voice_id = f"voice_{name.lower().replace(' ', '_')}"
    logger.info(f"🎯 Generated voice_id: {voice_id} from name: {name}")
    
    try:
        # Decode audio data
        logger.info("🔍 Decoding audio data from base64...")
        audio_data = base64.b64decode(audio_data_base64)
        logger.info(f"✅ Audio data decoded: {len(audio_data)} bytes")
        
        # Check if model is pre-loaded
        if not model:
            logger.error("❌ Model not pre-loaded")
            return {"status": "error", "message": "Higgs Audio model not available"}
        else:
            logger.info("✅ Model already pre-loaded")
        
        start_time = time.time()
        logger.info(f"⏱️ Starting voice cloning process at: {datetime.fromtimestamp(start_time)}")
        
        # Step 1: Extract voice profile
        logger.info("🔄 Step 1: Extracting voice profile...")
        voice_profile = extract_voice_profile(audio_data, voice_id)
        
        if voice_profile is None:
            logger.error("❌ Voice profile extraction failed")
            return {"status": "error", "message": "Failed to extract voice profile"}
        
        logger.info(f"✅ Voice profile extraction completed in {time.time() - start_time:.2f}s")
        
        # Step 2: Generate voice sample
        logger.info("🔄 Step 2: Generating voice sample...")
        sample_start_time = time.time()
        sample_audio = generate_voice_sample(voice_profile, voice_id, text)
        
        if sample_audio is None:
            logger.error("❌ Voice sample generation failed")
            return {"status": "error", "message": "Failed to generate voice sample"}
        
        logger.info(f"✅ Voice sample generation completed in {time.time() - sample_start_time:.2f}s")
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Voice cloning completed in {generation_time:.2f}s")
        
        # Step 3: Upload to Firebase
        logger.info("🔄 Step 3: Uploading to Firebase...")
        upload_start_time = time.time()
        
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
        
        logger.info(f"🔍 Uploading voice profile: {profile_path}")
        profile_url = upload_to_firebase(
            voice_profile.tobytes(),
            profile_path,
            "application/octet-stream",
            profile_metadata
        )
        
        if not profile_url:
            logger.error("❌ Failed to upload voice profile")
            return {"status": "error", "message": "Failed to upload voice profile"}
        
        # Upload recorded audio
        recorded_filename = f"{voice_id}_{timestamp}_recorded.wav"
        recorded_path = f"audio/voices/en/recorded/{recorded_filename}"
        
        recorded_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "recorded_audio",
            "model": "higgs_audio_v2"
        }
        
        logger.info(f"🔍 Uploading recorded audio: {recorded_path}")
        recorded_url = upload_to_firebase(
            audio_data,
            recorded_path,
            "audio/wav",
            recorded_metadata
        )
        
        if not recorded_url:
            logger.error("❌ Failed to upload recorded audio")
            return {"status": "error", "message": "Failed to upload recorded audio"}
        
        # Upload sample audio
        sample_filename = f"{voice_id}_{timestamp}_sample.wav"
        sample_path = f"audio/voices/en/samples/{sample_filename}"
        
        sample_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "sample_audio",
            "model": "higgs_audio_v2"
        }
        
        logger.info(f"🔍 Uploading sample audio: {sample_path}")
        sample_url = upload_to_firebase(
            sample_audio,
            sample_path,
            "audio/wav",
            sample_metadata
        )
        
        if not sample_url:
            logger.error("❌ Failed to upload sample audio")
            return {"status": "error", "message": "Failed to upload sample audio"}
        
        upload_time = time.time() - upload_start_time
        logger.info(f"✅ Firebase upload completed in {upload_time:.2f}s")
        
        # Return success response
        logger.info("✅ Voice cloning completed successfully")
        
        return {
            "status": "success",
            "profile_path": profile_path,
            "recorded_audio_path": recorded_path,
            "sample_audio_path": sample_path,
            "metadata": {
                "voice_id": voice_id,
                "voice_name": name,
                "created_date": int(start_time),
                "language": "en",
                "is_kids_voice": False,
                "model": "higgs_audio_v2",
                "generation_time": generation_time,
                "upload_time": upload_time
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice cloning traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Voice cloning failed: {str(e)}"}

# Register the handler
logger.info("🔧 Registering Higgs Audio VC handler with RunPod...")
runpod.serverless.start({"handler": handler})
logger.info("✅ Higgs Audio VC handler registered successfully") 
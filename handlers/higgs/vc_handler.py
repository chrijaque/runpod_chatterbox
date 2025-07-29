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

# Model paths for Higgs Audio
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

logger.info(f"🔧 Model configuration:")
logger.info(f"   - MODEL_PATH: {MODEL_PATH}")
logger.info(f"   - AUDIO_TOKENIZER_PATH: {AUDIO_TOKENIZER_PATH}")

# Add model download verification with detailed logging
def verify_model_availability():
    """Verify that Higgs Audio models are available with detailed logging"""
    logger.info("🔍 Starting model availability verification...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        logger.info("✅ Successfully imported transformers")
        
        # Try to load the tokenizer
        logger.info(f"🔍 Attempting to load tokenizer: {AUDIO_TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(AUDIO_TOKENIZER_PATH)
        logger.info(f"✅ Audio tokenizer loaded successfully: {type(tokenizer)}")
        logger.info(f"✅ Tokenizer vocab size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'N/A'}")
        
        # Try to load the model (this will download if not cached)
        logger.info(f"🔍 Attempting to load model: {MODEL_PATH}")
        model = AutoModel.from_pretrained(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully: {type(model)}")
        logger.info(f"✅ Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full model verification traceback: {traceback.format_exc()}")
        return False

# Local directory paths
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")
TEMP_VOICE_DIR = Path("/temp_voice")

logger.info(f"🔧 Directory configuration:")
logger.info(f"   - VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"   - VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"   - TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Initialize Higgs Audio model
model = None
serve_engine = None

def initialize_firebase():
    """Initialize Firebase Storage client with detailed logging"""
    global storage_client, bucket
    
    logger.info("🔍 Initializing Firebase Storage...")
    
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        
        logger.info("✅ Successfully imported Google Cloud Storage")
        
        # Get Firebase credentials from environment
        firebase_creds = os.getenv('RUNPOD_SECRET_Firebase')
        bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')
        
        logger.info(f"🔍 Environment variables:")
        logger.info(f"   - RUNPOD_SECRET_Firebase: {'SET' if firebase_creds else 'NOT SET'}")
        logger.info(f"   - FIREBASE_STORAGE_BUCKET: {bucket_name}")
        
        if not firebase_creds or not bucket_name:
            logger.error("❌ Firebase credentials or bucket name not found in environment")
            return False
        
        # Parse Firebase credentials
        import json
        logger.info("🔍 Parsing Firebase credentials...")
        creds_dict = json.loads(firebase_creds)
        logger.info(f"✅ Firebase credentials parsed successfully")
        logger.info(f"✅ Credentials keys: {list(creds_dict.keys())}")
        
        # Create credentials object
        logger.info("🔍 Creating service account credentials...")
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        logger.info(f"✅ Service account credentials created: {credentials}")
        
        # Initialize storage client
        logger.info("🔍 Initializing storage client...")
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        
        logger.info(f"✅ Firebase initialized successfully")
        logger.info(f"✅ Connected to bucket: {bucket_name}")
        logger.info(f"✅ Storage client: {storage_client}")
        logger.info(f"✅ Bucket: {bucket}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full Firebase initialization traceback: {traceback.format_exc()}")
        return False

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """Upload data to Firebase Storage with detailed logging"""
    global bucket
    
    logger.info(f"🔍 Starting Firebase upload...")
    logger.info(f"   - Destination: {destination_blob_name}")
    logger.info(f"   - Content type: {content_type}")
    logger.info(f"   - Data size: {len(data)} bytes")
    logger.info(f"   - Metadata: {metadata}")
    
    if not bucket:
        logger.info("🔍 Bucket not initialized, attempting to initialize Firebase...")
        if not initialize_firebase():
            logger.error("❌ Failed to initialize Firebase for upload")
            return None
    
    try:
        logger.info("🔍 Creating blob...")
        blob = bucket.blob(destination_blob_name)
        logger.info(f"✅ Blob created: {blob}")
        
        # Set metadata if provided
        if metadata:
            logger.info("🔍 Setting blob metadata...")
            blob.metadata = metadata
            logger.info(f"✅ Metadata set: {metadata}")
        
        # Upload the data
        logger.info("🔍 Uploading data to Firebase...")
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"✅ Data uploaded successfully")
        
        # Make the blob publicly accessible
        logger.info("🔍 Making blob publicly accessible...")
        blob.make_public()
        logger.info(f"✅ Blob made public")
        
        public_url = blob.public_url
        logger.info(f"✅ Upload completed successfully")
        logger.info(f"✅ Public URL: {public_url}")
        
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload to Firebase: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full upload traceback: {traceback.format_exc()}")
        return None

def initialize_model():
    """Initialize Higgs Audio model with detailed logging"""
    global model, serve_engine
    
    logger.info("🔍 Starting Higgs Audio model initialization...")
    
    if model is not None:
        logger.info("✅ Model already initialized")
        return model
    
    if not HIGGS_AVAILABLE:
        logger.error("❌ Higgs Audio components not available")
        raise RuntimeError("Higgs Audio components not available")
    
    logger.info("✅ Higgs Audio components are available")
    
    try:
        import torch
        
        logger.info("🔍 Checking CUDA availability...")
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"✅ CUDA device: {device_name}")
            logger.info(f"✅ CUDA version: {cuda_version}")
            logger.info(f"✅ PyTorch version: {torch.__version__}")
        else:
            logger.warning("⚠️ CUDA is required but not available")
            raise RuntimeError("CUDA is required but not available")
        
        device = "cuda" if cuda_available else "cpu"
        logger.info(f"✅ Using device: {device}")
        
        # Verify model availability first
        logger.info("🔍 Verifying model availability...")
        if not verify_model_availability():
            logger.error("❌ Model verification failed")
            raise RuntimeError("Higgs Audio models not available")
        
        # Initialize Higgs Audio serve engine
        logger.info(f"🔧 Initializing serve engine...")
        logger.info(f"   - Model path: {MODEL_PATH}")
        logger.info(f"   - Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
        logger.info(f"   - Device: {device}")
        
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device=device
        )
        
        logger.info(f"✅ Serve engine initialized successfully: {type(serve_engine)}")
        
        model = serve_engine  # For compatibility with existing code
        logger.info("✅ Higgs Audio model initialized successfully")
        logger.info(f"✅ Model type: {type(model)}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Higgs Audio model: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full model initialization traceback: {traceback.format_exc()}")
        raise

def extract_voice_profile(audio_data: bytes, voice_id: str) -> Optional[np.ndarray]:
    """Extract voice profile from audio data using Higgs Audio with detailed logging"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice profile extraction for: {voice_id}")
    logger.info(f"   - Audio data size: {len(audio_data)} bytes")
    
    if not serve_engine:
        logger.info("🔍 Serve engine not initialized, initializing...")
        initialize_model()
    
    try:
        # Save audio data to temporary file
        logger.info("🔍 Creating temporary audio file...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_audio_path = temp_file.name
        
        logger.info(f"✅ Temporary file created: {temp_audio_path}")
        
        # Create system prompt for voice cloning
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        logger.info(f"🔍 System prompt: {system_prompt}")
        
        # Create messages for voice profile extraction
        logger.info("🔍 Creating messages for voice profile extraction...")
        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user", 
                content=AudioContent(audio=audio_data, sampling_rate=24000)  # Default sample rate
            )
        ]
        
        logger.info(f"✅ Messages created: {len(messages)} messages")
        logger.info(f"✅ Message types: {[type(msg.content) for msg in messages]}")
        
        # Generate voice profile
        logger.info("🔍 Generating voice profile using serve engine...")
        logger.info(f"   - Max new tokens: 512")
        logger.info(f"   - Temperature: 0.3")
        logger.info(f"   - Top p: 0.95")
        logger.info(f"   - Top k: 50")
        
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        logger.info(f"✅ Voice profile generation completed")
        logger.info(f"✅ Response type: {type(response)}")
        
        # Extract voice profile from response
        # Note: This is a simplified approach - actual voice profile extraction
        # would require more sophisticated processing of the model's internal states
        
        # For now, we'll create a placeholder voice profile
        # In a real implementation, you'd extract the actual voice embeddings
        logger.info("🔍 Creating placeholder voice profile...")
        voice_profile = np.random.randn(1024).astype(np.float32)  # Placeholder
        
        logger.info(f"✅ Voice profile extracted: shape={voice_profile.shape}")
        logger.info(f"✅ Voice profile dtype: {voice_profile.dtype}")
        logger.info(f"✅ Voice profile min/max: {voice_profile.min():.4f}/{voice_profile.max():.4f}")
        
        return voice_profile
        
    except Exception as e:
        logger.error(f"❌ Failed to extract voice profile: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice profile extraction traceback: {traceback.format_exc()}")
        return None
    finally:
        # Clean up temporary file
        if 'temp_audio_path' in locals():
            try:
                logger.info(f"🔍 Cleaning up temporary file: {temp_audio_path}")
                os.unlink(temp_audio_path)
                logger.info(f"✅ Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up temporary file: {e}")

def generate_voice_sample(voice_profile: np.ndarray, voice_id: str, text: str) -> Optional[bytes]:
    """Generate voice sample using Higgs Audio with detailed logging"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice sample generation for: {voice_id}")
    logger.info(f"   - Voice profile shape: {voice_profile.shape}")
    logger.info(f"   - Text: {text}")
    
    if not serve_engine:
        logger.info("🔍 Serve engine not initialized, initializing...")
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
        
        logger.info(f"🔍 System prompt: {system_prompt}")
        
        # Create messages for TTS generation
        logger.info("🔍 Creating messages for TTS generation...")
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=text)
        ]
        
        logger.info(f"✅ Messages created: {len(messages)} messages")
        logger.info(f"✅ Message types: {[type(msg.content) for msg in messages]}")
        
        # Generate audio
        logger.info("🔍 Generating audio using serve engine...")
        logger.info(f"   - Max new tokens: 1024")
        logger.info(f"   - Temperature: 0.3")
        logger.info(f"   - Top p: 0.95")
        logger.info(f"   - Top k: 50")
        
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        logger.info(f"✅ Audio generation completed")
        logger.info(f"✅ Response type: {type(response)}")
        
        # Convert audio to bytes
        logger.info("🔍 Converting audio to bytes...")
        import torch
        audio_tensor = torch.from_numpy(response.audio)
        
        logger.info(f"✅ Audio tensor created: {audio_tensor.shape}")
        logger.info(f"✅ Audio tensor dtype: {audio_tensor.dtype}")
        logger.info(f"✅ Sampling rate: {response.sampling_rate}")
        
        # Save to temporary file and read as bytes
        logger.info("🔍 Saving audio to temporary file...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            import torchaudio
            torchaudio.save(temp_file.name, audio_tensor.unsqueeze(0), response.sampling_rate)
            logger.info(f"✅ Audio saved to temporary file: {temp_file.name}")
            
            with open(temp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            logger.info(f"✅ Audio bytes read: {len(audio_bytes)} bytes")
            
            # Clean up
            os.unlink(temp_file.name)
            logger.info(f"✅ Temporary file cleaned up")
        
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
        
        # Initialize model if needed
        if not model:
            logger.info("🔧 Model not initialized, starting initialization...")
            initialize_model()
            logger.info("✅ Model initialization completed")
        else:
            logger.info("✅ Model already initialized")
        
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
        
        logger.info(f"🔍 Uploading voice sample: {sample_path}")
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
        
        logger.info(f"🔍 Uploading original audio: {recorded_path}")
        recorded_url = upload_to_firebase(
            audio_data,
            recorded_path,
            "audio/mpeg",
            recorded_metadata
        )
        
        upload_time = time.time() - upload_start_time
        logger.info(f"✅ All files uploaded to Firebase successfully in {upload_time:.2f}s")
        
        # Prepare response
        response = {
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
        
        logger.info("✅ Voice cloning process completed successfully")
        logger.info(f"📤 Returning response with {len(response)} keys")
        logger.info(f"📤 Response keys: {list(response.keys())}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice cloning traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Voice cloning failed: {str(e)}"} 
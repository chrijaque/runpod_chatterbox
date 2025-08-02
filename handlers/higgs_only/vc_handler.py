import runpod
import time
import os
import tempfile
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
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

# Set cache directory to use network volume
import os
os.environ["HF_HOME"] = "/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/cache"

# Configure PyTorch CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Cache for audio tokenizer to avoid reloading
audio_tokenizer_cache = None

def initialize_cuda_device():
    """Initialize and clean up CUDA device"""
    try:
        import torch
        import gc
        
        logger.info("🔍 Initializing CUDA device...")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("❌ CUDA is not available")
            return False
        
        # Clear GPU memory
        logger.info("🔍 Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check device availability
        device_count = torch.cuda.device_count()
        logger.info(f"🔍 CUDA devices available: {device_count}")
        
        if device_count > 0:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            
            logger.info(f"🔍 Current CUDA device: {current_device}")
            logger.info(f"🔍 Device name: {device_name}")
            logger.info(f"🔍 Memory allocated: {memory_allocated:.2f} GB")
            logger.info(f"🔍 Memory reserved: {memory_reserved:.2f} GB")
            
            # Set device explicitly
            torch.cuda.set_device(current_device)
            logger.info(f"✅ CUDA device {current_device} set successfully")
            
            return True
        else:
            logger.error("❌ No CUDA devices available")
            return False
            
    except Exception as e:
        logger.error(f"❌ CUDA initialization failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full CUDA initialization traceback: {traceback.format_exc()}")
        return False

# Model paths for Higgs Audio (from network volume)
MODEL_PATH = "/runpod-volume/higgs_audio_generation"
AUDIO_TOKENIZER_PATH = "/runpod-volume/higgs_audio_tokenizer"
HUBERT_PATH = "/runpod-volume/hubert_base"

# MP3 conversion functions (from ChatterboxTTS implementation)
def tensor_to_mp3_bytes(audio_tensor, sample_rate, bitrate="96k"):
    """
    Convert audio tensor directly to MP3 bytes.
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :param bitrate: MP3 bitrate (e.g., "96k", "128k", "160k")
    :return: MP3 bytes
    """
    try:
        from pydub import AudioSegment
        # Convert tensor to AudioSegment
        audio_segment = tensor_to_audiosegment(audio_tensor, sample_rate)
        # Export to MP3 bytes
        mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
        # Read the bytes from the file object
        mp3_bytes = mp3_file.read()
        return mp3_bytes
    except ImportError:
        logger.warning("pydub not available, falling back to WAV")
        return tensor_to_wav_bytes(audio_tensor, sample_rate)
    except Exception as e:
        logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
        return tensor_to_wav_bytes(audio_tensor, sample_rate)

def tensor_to_audiosegment(audio_tensor, sample_rate):
    """
    Convert PyTorch audio tensor to pydub AudioSegment.
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :return: pydub AudioSegment
    """
    from pydub import AudioSegment
    
    # Convert tensor to numpy array
    if audio_tensor.dim() == 2:
        # Stereo: (channels, samples)
        audio_np = audio_tensor.numpy()
    else:
        # Mono: (samples,) -> (1, samples)
        audio_np = audio_tensor.unsqueeze(0).numpy()
    
    # Convert to int16 for pydub
    audio_np = (audio_np * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_np.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=audio_np.shape[0]
    )
    
    return audio_segment

def tensor_to_wav_bytes(audio_tensor, sample_rate):
    """
    Convert audio tensor to WAV bytes (fallback).
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :return: WAV bytes
    """
    # Save to temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(temp_wav.name, audio_tensor, sample_rate)
    
    # Read WAV bytes
    with open(temp_wav.name, 'rb') as f:
        wav_bytes = f.read()
    
    # Clean up temp file
    os.unlink(temp_wav.name)
    
    return wav_bytes

def convert_audio_file_to_mp3(input_path, output_path, bitrate="160k"):
    """
    Convert audio file to MP3 with specified bitrate.
    
    :param input_path: Path to input audio file
    :param output_path: Path to output MP3 file
    :param bitrate: MP3 bitrate
    """
    try:
        from pydub import AudioSegment
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        # Export as MP3
        audio.export(output_path, format="mp3", bitrate=bitrate)
        logger.info(f"✅ Converted {input_path} to MP3: {output_path}")
    except ImportError:
        raise ImportError("pydub is required for audio conversion")
    except Exception as e:
        logger.error(f"❌ Failed to convert {input_path} to MP3: {e}")
        raise

logger.info(f"🔧 Model configuration:")
logger.info(f"   - MODEL_PATH: {MODEL_PATH}")
logger.info(f"   - AUDIO_TOKENIZER_PATH: {AUDIO_TOKENIZER_PATH}")
logger.info(f"   - HUBERT_PATH: {HUBERT_PATH}")

# Debug: Check Network Volume contents
logger.info("🔍 Checking Network Volume contents...")
runpod_volume = "/runpod-volume"
if os.path.exists(runpod_volume):
    logger.info(f"✅ {runpod_volume} exists")
    try:
        contents = os.listdir(runpod_volume)
        logger.info(f"📁 Contents of {runpod_volume}:")
        for item in contents:
            logger.info(f"   - {item}")
    except Exception as e:
        logger.error(f"❌ Error listing {runpod_volume}: {e}")
else:
    logger.error(f"❌ {runpod_volume} does not exist")

# Debug: Check model directory contents
logger.info("📂 Checking model directory contents:")
for model_path in [MODEL_PATH, AUDIO_TOKENIZER_PATH, HUBERT_PATH]:
    logger.info(f"🔍 Checking: {model_path}")
    if os.path.exists(model_path):
        logger.info(f"✅ {model_path} exists")
        try:
            files = os.listdir(model_path)
            logger.info(f"📁 Contents of {model_path}:")
            for f in files[:10]:  # Show first 10 files
                logger.info(f"   - {f}")
            if len(files) > 10:
                logger.info(f"   ... and {len(files) - 10} more files")
            
            # Check for key files
            config_path = os.path.join(model_path, "config.json")
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            logger.info(f"📄 Checking if config.json exists: {os.path.exists(config_path)}")
            logger.info(f"📄 Checking if model index exists: {os.path.exists(index_path)}")
        except Exception as e:
            logger.error(f"❌ Error reading {model_path}: {e}")
    else:
        logger.error(f"❌ {model_path} does not exist")

# Pre-load Higgs Audio models at module level (avoids runtime initialization)
logger.info("🔧 Pre-loading Higgs Audio models...")
serve_engine = None
model = None

# Step 1: Initialize CUDA device
logger.info("🔍 Step 1: Initializing CUDA device...")
if not initialize_cuda_device():
    logger.error("❌ CUDA device initialization failed")
    serve_engine = None
    model = None
else:
    # Step 2: Try to load models with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if HIGGS_AVAILABLE:
                logger.info(f"🔍 Step 2: Attempt {attempt + 1}/{max_retries} to pre-load Higgs Audio serve engine...")
                logger.info(f"   - Model path: {MODEL_PATH}")
                logger.info(f"   - Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
                logger.info(f"   - Device: cuda")
                
                serve_engine = HiggsAudioServeEngine(
                    model_name_or_path=MODEL_PATH,
                    audio_tokenizer_name_or_path=AUDIO_TOKENIZER_PATH,
                    device="cuda"
                )
                
                model = serve_engine  # For compatibility with existing code
                logger.info("✅ Higgs Audio models pre-loaded successfully")
                logger.info(f"✅ Serve engine type: {type(serve_engine)}")
                break  # Success, exit retry loop
            else:
                logger.error("❌ Higgs Audio components not available for pre-loading")
                serve_engine = None
                model = None
                break
                
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries - 1:
                logger.warning(f"⚠️ CUDA error on attempt {attempt + 1}, retrying in 5 seconds...")
                import time
                time.sleep(5)  # Wait 5 seconds before retry
                
                # Clear memory before retry
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                logger.error(f"❌ Failed to pre-load Higgs Audio models: {e}")
                logger.error(f"❌ Error type: {type(e)}")
                import traceback
                logger.error(f"❌ Full pre-load traceback: {traceback.format_exc()}")
                serve_engine = None
                model = None
                break
        except Exception as e:
            logger.error(f"❌ Unexpected error during model loading: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full pre-load traceback: {traceback.format_exc()}")
            serve_engine = None
            model = None
            break

def initialize_firebase():
    """Initialize Firebase storage client using google-cloud-storage"""
    global storage_client, bucket
    
    try:
        from google.cloud import storage
        
        # Debug: Check environment variables
        logger.info("🔍 Checking Firebase environment variables...")
        firebase_secret = os.getenv('RUNPOD_SECRET_Firebase')
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"🔍 RUNPOD_SECRET_Firebase exists: {firebase_secret is not None}")
        logger.info(f"🔍 GOOGLE_APPLICATION_CREDENTIALS exists: {google_creds is not None}")
        
        # Check if we're in RunPod and have the secret
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        
        if firebase_secret_path:
            if firebase_secret_path.startswith('{'):
                # It's JSON content, create a temporary file
                logger.info("✅ Using RunPod Firebase secret as JSON content")
                import tempfile
                import json
                
                # Validate JSON first
                try:
                    creds_data = json.loads(firebase_secret_path)
                    logger.info(f"✅ Valid JSON with project_id: {creds_data.get('project_id', 'unknown')}")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in RUNPOD_SECRET_Firebase: {e}")
                    raise
                
                # Create temporary file with the JSON content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(creds_data, tmp_file)
                    tmp_path = tmp_file.name
                
                logger.info(f"✅ Created temporary credentials file: {tmp_path}")
                storage_client = storage.Client.from_service_account_json(tmp_path)
                
            elif os.path.exists(firebase_secret_path):
                # It's a file path
                logger.info(f"✅ Using RunPod Firebase secret file: {firebase_secret_path}")
                storage_client = storage.Client.from_service_account_json(firebase_secret_path)
            else:
                logger.warning(f"⚠️ RUNPOD_SECRET_Firebase exists but is not JSON content or valid file path")
                # Fallback to GOOGLE_APPLICATION_CREDENTIALS
                logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
                storage_client = storage.Client()
        else:
            # No RunPod secret, fallback to GOOGLE_APPLICATION_CREDENTIALS
            logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
            storage_client = storage.Client()
        
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase storage client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase storage: {e}")
        return False

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """
    Upload data directly to Firebase Storage with metadata
    
    :param data: Binary data to upload
    :param destination_blob_name: Destination path in Firebase
    :param content_type: MIME type of the file
    :param metadata: Optional metadata to store with the file
    :return: Public URL or None if failed
    """
    global bucket # Ensure bucket is accessible
    if bucket is None:
        logger.info("🔍 Bucket is None, initializing Firebase...")
        if not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot upload")
            return None
    
    try:
        logger.info(f"🔍 Creating blob: {destination_blob_name}")
        blob = bucket.blob(destination_blob_name)
        logger.info(f"🔍 Uploading {len(data)} bytes...")
        
        # Set metadata if provided
        if metadata:
            blob.metadata = metadata
            logger.info(f"🔍 Set metadata: {metadata}")
        
        # Set content type
        blob.content_type = content_type
        logger.info(f"🔍 Set content type: {content_type}")
        
        # Upload the data
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"🔍 Upload completed, making public...")
        
        # Make the blob publicly accessible
        blob.make_public()
        
        public_url = blob.public_url
        logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Firebase upload failed: {e}")
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

def ensure_model_loaded():
    """Ensure model is loaded, try runtime loading if pre-loading failed"""
    global serve_engine, model
    
    if serve_engine is not None and model is not None:
        logger.info("✅ Model already loaded")
        return True
    
    logger.warning("⚠️ Model not pre-loaded, attempting runtime loading...")
    
    # Step 1: Initialize CUDA device
    logger.info("🔍 Step 1: Initializing CUDA device for runtime loading...")
    if not initialize_cuda_device():
        logger.error("❌ CUDA device initialization failed for runtime loading")
        return False
    
    # Step 2: Try to load models with retry logic
    max_retries = 2  # Fewer retries for runtime loading
    for attempt in range(max_retries):
        try:
            if HIGGS_AVAILABLE:
                logger.info(f"🔍 Step 2: Runtime loading attempt {attempt + 1}/{max_retries}...")
                logger.info(f"   - Model path: {MODEL_PATH}")
                logger.info(f"   - Audio tokenizer path: {AUDIO_TOKENIZER_PATH}")
                logger.info(f"   - Device: cuda")
                
                serve_engine = HiggsAudioServeEngine(
                    model_name_or_path=MODEL_PATH,
                    audio_tokenizer_name_or_path=AUDIO_TOKENIZER_PATH,
                    device="cuda"
                )
                
                model = serve_engine  # For compatibility with existing code
                logger.info("✅ Higgs Audio models loaded successfully at runtime")
                logger.info(f"✅ Serve engine type: {type(serve_engine)}")
                return True
            else:
                logger.error("❌ Higgs Audio components not available for runtime loading")
                return False
                
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries - 1:
                logger.warning(f"⚠️ CUDA error on runtime attempt {attempt + 1}, retrying in 3 seconds...")
                import time
                time.sleep(3)  # Shorter wait for runtime loading
                
                # Clear memory before retry
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                logger.error(f"❌ Failed to load Higgs Audio models at runtime: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during runtime model loading: {e}")
            return False
    
    return False

def extract_voice_profile(audio_data: bytes, voice_id: str) -> Optional[np.ndarray]:
    """Extract voice profile embedding using Higgs Audio tokenizer with HuBERT cleanup"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice profile extraction for: {voice_id}")
    logger.info(f"   - Audio data size: {len(audio_data)} bytes")
    
    try:
        # 1. Save audio data to temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_file.write(audio_data)
        temp_audio_file.close()
        
        logger.info(f"✅ Created temporary audio file: {temp_audio_file.name}")
        
        # 2. Load Higgs Audio tokenizer
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        logger.info("🔍 Loading Higgs Audio tokenizer...")
        tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cuda")
        logger.info("✅ Higgs Audio tokenizer loaded successfully")
        
        # 3. Extract voice profile embedding
        logger.info("🔍 Extracting voice profile embedding...")
        voice_profile_tokens = tokenizer.encode(temp_audio_file.name)
        
        # 4. Convert to numpy array
        voice_profile_np = voice_profile_tokens.squeeze(0).cpu().numpy()
        
        # 5. Clean up temp file
        os.unlink(temp_audio_file.name)
        logger.info(f"✅ Cleaned up temporary file: {temp_audio_file.name}")
        
        logger.info(f"✅ Voice profile extracted: shape={voice_profile_np.shape}")
        logger.info(f"✅ Voice profile size: {voice_profile_np.nbytes / 1024 / 1024:.2f} MB")
        
        # 6. HuBERT cleanup after voice profile extraction
        logger.info("🔍 HuBERT cleanup: Unloading HuBERT model after voice profile extraction...")
        import torch
        import gc
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check memory after cleanup
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
        
        logger.info(f"✅ After HuBERT cleanup:")
        logger.info(f"   - Allocated: {memory_allocated:.2f} GB")
        logger.info(f"   - Reserved: {memory_reserved:.2f} GB")
        logger.info(f"   - Free: {memory_free:.2f} GB")
        
        return voice_profile_np
        
    except Exception as e:
        logger.error(f"❌ Failed to extract voice profile: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full voice profile extraction traceback: {traceback.format_exc()}")
        return None

def generate_voice_sample(voice_profile: np.ndarray, voice_id: str, text: str) -> Optional[bytes]:
    """Generate voice sample using voice profile embedding with correct API"""
    global serve_engine
    
    logger.info(f"🔍 Starting voice sample generation for: {voice_id}")
    logger.info(f"   - Voice profile shape: {voice_profile.shape}")
    logger.info(f"   - Text: {text[:50]}...")
    
    if not serve_engine:
        logger.error("❌ Serve engine not pre-loaded")
        return None
    
    try:
        # Step 1: Load audio tokenizer for decoding voice profile (with caching)
        global audio_tokenizer_cache
        if audio_tokenizer_cache is None:
            logger.info("🔍 Loading audio tokenizer for voice profile decoding...")
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            audio_tokenizer_cache = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device="cuda")
            logger.info("✅ Audio tokenizer loaded and cached successfully")
        else:
            logger.info("✅ Using cached audio tokenizer")
        
        audio_tokenizer = audio_tokenizer_cache
        
        # Step 2: Convert voice profile back to audio
        logger.info("🔍 Converting voice profile back to audio...")
        import torch
        voice_profile_tensor = torch.from_numpy(voice_profile).unsqueeze(0).to("cuda")
        logger.info(f"✅ Voice profile converted to tensor: shape={voice_profile_tensor.shape}")
        
        decoded_audio = audio_tokenizer.decode(voice_profile_tensor)
        logger.info(f"✅ Voice profile decoded to audio: shape={decoded_audio.shape}")
        
        # Step 3: Convert to AudioContent
        logger.info("🔍 Converting decoded audio to AudioContent...")
        audio_int16 = (decoded_audio[0, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_content = AudioContent(raw_audio=audio_base64, audio_url="placeholder")
        logger.info(f"✅ AudioContent created: {len(audio_base64)} characters")
        
        # Step 4: Create ChatMLSample with voice profile
        logger.info("🔍 Creating ChatMLSample with voice profile...")
        messages = [
            Message(role="user", content=text),
            Message(role="assistant", content=audio_content)  # Voice profile as audio
        ]
        chat_ml_sample = ChatMLSample(messages=messages)
        logger.info(f"✅ ChatMLSample created with {len(messages)} messages")
        
        # Step 5: Generate with correct API
        logger.info("🔍 Generating audio with voice profile embedding...")
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        # Step 6: Convert response to MP3 bytes
        logger.info("🔍 Converting response to MP3 bytes...")
        audio_tensor = torch.from_numpy(response.audio)
        logger.info(f"✅ Audio generated: shape={audio_tensor.shape}, sample_rate={response.sampling_rate}")
        
        audio_bytes = tensor_to_mp3_bytes(audio_tensor, response.sampling_rate, "96k")
        logger.info(f"✅ Voice sample generated: {len(audio_bytes)} bytes (96k MP3)")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"❌ Manual approach failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full manual approach traceback: {traceback.format_exc()}")
        
        # Try fallback method
        logger.info("🔄 Attempting fallback method...")
        fallback_result = generate_with_voice_profile_fallback(voice_profile, voice_id, text)
        if fallback_result is not None:
            logger.info("✅ Fallback method succeeded")
            return fallback_result
        else:
            logger.error("❌ Both manual and fallback methods failed")
            return None

def validate_voice_profile(voice_profile: np.ndarray) -> bool:
    """Validate voice profile embedding format"""
    try:
        # Check shape (should be 2D: num_codebooks x sequence_length)
        if len(voice_profile.shape) != 2:
            logger.error(f"❌ Invalid voice profile shape: {voice_profile.shape}")
            return False
        
        # Check codebook values (should be integers 0-1023)
        if not np.all((voice_profile >= 0) & (voice_profile < 1024)):
            logger.error(f"❌ Invalid codebook values in voice profile")
            return False
        
        # Check reasonable size (0.01-10 MB) - allow smaller profiles for short audio
        size_mb = voice_profile.nbytes / 1024 / 1024
        if size_mb < 0.01 or size_mb > 10:
            logger.error(f"❌ Voice profile size too large/small: {size_mb:.2f} MB")
            return False
        
        logger.info(f"✅ Voice profile validation passed: shape={voice_profile.shape}, size={size_mb:.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice profile validation failed: {e}")
        return False

def generate_with_voice_profile_fallback(voice_profile: np.ndarray, voice_id: str, text: str) -> Optional[bytes]:
    """Fallback method using VoiceProfileTTSGenerator if manual approach fails"""
    logger.info(f"🔄 Attempting fallback method with VoiceProfileTTSGenerator for: {voice_id}")
    
    try:
        from tts import VoiceProfileTTSGenerator
        
        # Initialize TTS generator
        logger.info("🔍 Initializing VoiceProfileTTSGenerator...")
        tts_generator = VoiceProfileTTSGenerator(
            model_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer", 
            device="cuda"
        )
        logger.info("✅ VoiceProfileTTSGenerator initialized successfully")
        
        # Generate with voice profile
        logger.info("🔍 Generating TTS with voice profile...")
        response = tts_generator.generate_tts_with_voice_profile(
            text=text,
            voice_profile=voice_profile,
            temperature=0.3,
            max_new_tokens=1024
        )
        logger.info("✅ TTS generation completed successfully")
        
        # Convert to MP3 bytes
        import torch
        audio_tensor = torch.from_numpy(response.audio)
        audio_bytes = tensor_to_mp3_bytes(audio_tensor, response.sampling_rate, "96k")
        logger.info(f"✅ Fallback voice sample generated: {len(audio_bytes)} bytes (96k MP3)")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"❌ Fallback method also failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full fallback traceback: {traceback.format_exc()}")
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
    
    # Extract parameters
    name = input_data.get('name')
    audio_data_base64 = input_data.get('audio_data')
    audio_format = input_data.get('audio_format', 'wav')
    response_format = input_data.get('responseFormat', 'base64')
    language = input_data.get('language', 'en')
    is_kids_voice = input_data.get('is_kids_voice', False)
    model_type = input_data.get('model_type', 'higgs')
    text = f"Hello, this is the voice clone of {name}. This voice is used to narrate whimsical stories and fairytales."
    
    logger.info("🔍 Extracted parameters:")
    logger.info(f"   - Name: {name}")
    logger.info(f"   - Audio data base64: {'SET' if audio_data_base64 else 'NOT SET'}")
    logger.info(f"   - Audio format: {audio_format}")
    logger.info(f"   - Response format: {response_format}")
    logger.info(f"   - Language: {language}")
    logger.info(f"   - Is kids voice: {is_kids_voice}")
    logger.info(f"   - Model type: {model_type}")
    logger.info(f"   - Text: {text}")
    
    # Debug audio data received from API
    logger.info(f"🔍 Audio data details received from API:")
    logger.info(f"   - Has audio data: {bool(audio_data_base64)}")
    logger.info(f"   - Audio data length: {len(audio_data_base64) if audio_data_base64 else 0}")
    logger.info(f"   - Audio format: {audio_format}")
    logger.info(f"   - Audio data preview: {audio_data_base64[:200] + '...' if audio_data_base64 and len(audio_data_base64) > 200 else audio_data_base64}")
    logger.info(f"   - Audio data end: {audio_data_base64[-100:] if audio_data_base64 and len(audio_data_base64) > 100 else audio_data_base64}")
    
    # Validate audio data before processing
    if not audio_data_base64 or len(audio_data_base64) < 1000:
        logger.error(f"❌ Invalid audio data received from API:")
        logger.error(f"   - Has audio data: {bool(audio_data_base64)}")
        logger.error(f"   - Audio data length: {len(audio_data_base64) if audio_data_base64 else 0}")
        logger.error(f"   - Minimum expected: 1000")
        return {"status": "error", "message": "Invalid audio data - audio file too small or empty"}
    
    # Generate voice ID
    voice_id = f"voice_{name.lower().replace(' ', '_')}"
    logger.info(f"🎯 Generated voice_id: {voice_id} from name: {name}")
    
    try:
        # Decode audio data
        logger.info("🔍 Decoding audio data from base64...")
        audio_data = base64.b64decode(audio_data_base64)
        logger.info(f"✅ Audio data decoded: {len(audio_data)} bytes")
        
        # Debug decoded audio data
        logger.info(f"🔍 Decoded audio data details:")
        logger.info(f"   - Decoded data length: {len(audio_data)} bytes")
        logger.info(f"   - First 100 bytes: {audio_data[:100]}")
        logger.info(f"   - Last 100 bytes: {audio_data[-100:]}")
        
        # Check if model is loaded (with runtime loading fallback)
        if not ensure_model_loaded():
            logger.error("❌ Model loading failed")
            return {"status": "error", "message": "Higgs Audio model not available"}
        else:
            logger.info("✅ Model loaded successfully")
        
        start_time = time.time()
        logger.info(f"⏱️ Starting voice cloning process at: {datetime.fromtimestamp(start_time)}")
        
        # Step 1: Extract voice profile embedding (with HuBERT cleanup)
        logger.info("🔄 Step 1: Extracting voice profile embedding with HuBERT cleanup...")
        
        # Check memory before voice profile extraction
        import torch
        memory_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"🔍 Memory before voice profile extraction: {memory_before:.2f} GB")
        
        voice_profile = extract_voice_profile(audio_data, voice_id)
        
        if voice_profile is None:
            logger.error("❌ Voice profile extraction failed")
            return {"status": "error", "message": "Failed to extract voice profile"}
        
        # Validate voice profile embedding
        if not validate_voice_profile(voice_profile):
            logger.error("❌ Voice profile validation failed")
            return {"status": "error", "message": "Invalid voice profile format"}
        
        logger.info(f"✅ Voice profile embedding extraction completed in {time.time() - start_time:.2f}s")
        
        # Check memory after voice profile extraction
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_freed = memory_before - memory_after
        logger.info(f"🔍 Memory after voice profile extraction: {memory_after:.2f} GB")
        logger.info(f"🔍 Memory freed by HuBERT cleanup: {memory_freed:.2f} GB")
        
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
            "file_type": "voice_profile_embedding",  # Updated type
            "model": "higgs_audio_v2",
            "embedding_shape": voice_profile.shape,  # Add embedding info
            "embedding_size_mb": voice_profile.nbytes / 1024 / 1024,
            "num_codebooks": voice_profile.shape[0],
            "sequence_length": voice_profile.shape[1]
        }
        
        logger.info(f"🔍 Uploading voice profile embedding: {profile_path}")
        logger.info(f"   - Shape: {voice_profile.shape}")
        logger.info(f"   - Size: {voice_profile.nbytes / 1024 / 1024:.2f} MB")
        profile_url = upload_to_firebase(
            voice_profile.tobytes(),
            profile_path,
            "application/octet-stream",
            profile_metadata
        )
        
        if not profile_url:
            logger.error("❌ Failed to upload voice profile")
            return {"status": "error", "message": "Failed to upload voice profile"}
        
        # Upload recorded audio (160 kbps MP3)
        recorded_filename = f"{voice_id}_{timestamp}_recorded.mp3"
        recorded_path = f"audio/voices/en/recorded/{recorded_filename}"
        
        # Convert recorded audio to MP3
        temp_recorded_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_recorded_wav.write(audio_data)
        temp_recorded_wav.close()
        
        temp_recorded_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        convert_audio_file_to_mp3(temp_recorded_wav.name, temp_recorded_mp3.name, "160k")
        
        with open(temp_recorded_mp3.name, 'rb') as f:
            recorded_mp3_data = f.read()
        
        # Clean up temp files
        os.unlink(temp_recorded_wav.name)
        os.unlink(temp_recorded_mp3.name)
        
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
        
        logger.info(f"🔍 Uploading recorded audio: {recorded_path}")
        recorded_url = upload_to_firebase(
            recorded_mp3_data,
            recorded_path,
            "audio/mpeg",
            recorded_metadata
        )
        
        if not recorded_url:
            logger.error("❌ Failed to upload recorded audio")
            return {"status": "error", "message": "Failed to upload recorded audio"}
        
        # Upload sample audio (96 kbps MP3)
        sample_filename = f"{voice_id}_{timestamp}_sample.mp3"
        sample_path = f"audio/voices/en/samples/{sample_filename}"
        
        sample_metadata = {
            "voice_id": voice_id,
            "voice_name": name,
            "created_date": str(int(start_time)),
            "language": "en",
            "is_kids_voice": "False",
            "file_type": "sample_audio",
            "model": "higgs_audio_v2",
            "format": "96k_mp3"
        }
        
        logger.info(f"🔍 Uploading sample audio: {sample_path}")
        sample_url = upload_to_firebase(
            sample_audio,
            sample_path,
            "audio/mpeg",
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
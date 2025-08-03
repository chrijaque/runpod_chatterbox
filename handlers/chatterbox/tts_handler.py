import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
import torch
import logging
import hashlib
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from datetime import datetime, timedelta
from google.cloud import storage
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the handler from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    FORKED_HANDLER_AVAILABLE = True
    logger.info("✅ Successfully imported ChatterboxVC from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Could not import ChatterboxVC from forked repository: {e}")

# Check for NLTK availability
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    NLTK_AVAILABLE = True
    logger.info("✅ NLTK available for sentence tokenization")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

# Pre-load ChatterboxTTS model at module level (avoids re-initialization)
logger.info("🔧 Pre-loading ChatterboxTTS model...")
try:
    model = ChatterboxTTS.from_pretrained(device='cuda')
    logger.info("✅ ChatterboxTTS model pre-loaded successfully")
    
    # Initialize the forked repository handler if available
    if FORKED_HANDLER_AVAILABLE:
        logger.info("🔧 Pre-loading ChatterboxVC...")
        try:
            forked_handler = ChatterboxVC(
                s3gen=model.s3gen,
                device=model.device
            )
            logger.info("✅ ChatterboxVC pre-loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to pre-load ChatterboxVC: {e}")
            forked_handler = None
    else:
        logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")
        forked_handler = None
        
except Exception as e:
    logger.error(f"❌ Failed to pre-load ChatterboxTTS model: {e}")
    model = None
    forked_handler = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")  # For voice clone samples
TTS_GENERATED_DIR = Path("/tts_generated")  # For TTS story generation
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Check if directories exist
logger.info(f"Directory existence check:")
logger.info(f"  VOICE_PROFILES_DIR exists: {VOICE_PROFILES_DIR.exists()}")
logger.info(f"  VOICE_SAMPLES_DIR exists: {VOICE_SAMPLES_DIR.exists()}")
logger.info(f"  TTS_GENERATED_DIR exists: {TTS_GENERATED_DIR.exists()}")
logger.info(f"  TEMP_VOICE_DIR exists: {TEMP_VOICE_DIR.exists()}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# -------------------------------------------------------------------
# 🎵 MP3 Conversion Utilities
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 🐞  Firebase / GCS credential debug helper
# -------------------------------------------------------------------
def _debug_gcs_creds():
    """Minimal Firebase credential check - removed extensive debugging since voice cloning is working"""
    import os
    logger.info("🔍 Firebase credentials check")
    
    # Check if RunPod secret is available
    firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
    if firebase_secret_path:
        if firebase_secret_path.startswith('{'):
            logger.info("✅ Using RunPod Firebase secret (JSON content)")
        else:
            logger.info("✅ Using RunPod Firebase secret (file path)")
    else:
        logger.warning("⚠️ No RunPod Firebase secret found")

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    # Firebase initialization
    
    try:
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
    global model, forked_handler
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    logger.info("Initializing S3Token2Wav model...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    try:
        # Minimal initialization - focus on core functionality
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS model initialized")

        # Initialize the forked repository handler if available
        if FORKED_HANDLER_AVAILABLE:
            logger.info("🔧 Initializing ChatterboxVC...")
            try:
                forked_handler = ChatterboxVC(
                    s3gen=model.s3gen,
                    device=model.device
                )
                logger.info("✅ ChatterboxVC initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChatterboxVC: {e}")
                forked_handler = None
        else:
            logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")

        # Additional model introspection logs
        import inspect
        logger.info(f"📦 Model class: {model.__class__}")
        logger.info(f"📁 Model module: {model.__class__.__module__}")
        logger.info(f"📂 Loaded model from file: {inspect.getfile(model.__class__)}")
        logger.info(f"🧠 Model dir(): {dir(model)}")
        logger.info(f"🔎 Has method load_voice_profile: {hasattr(model, 'load_voice_profile')}")

        # List all methods that contain 'voice' or 'profile'
        voice_methods = [method for method in dir(model) if 'voice' in method.lower() or 'profile' in method.lower()]
        logger.info(f"🔍 Voice/Profile related methods: {voice_methods}")

        # Fast-fail check for required method
        assert hasattr(model, 'load_voice_profile'), "🚨 Loaded model is missing `load_voice_profile`. Wrong class?"

        # Verify s3gen module source
        logger.info("🔍 ===== S3GEN VERIFICATION =====")
        if hasattr(model, "s3gen"):
            logger.info(f"📂 s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 s3gen class: {model.s3gen.__class__}")
            logger.info(f"📂 s3gen class file: {model.s3gen.__class__.__module__}")
            
            # Check s3gen module file path
            try:
                s3gen_file = inspect.getfile(model.s3gen.__class__)
                logger.info(f"📂 s3gen file path: {s3gen_file}")
                logger.info(f"📂 s3gen file exists: {os.path.exists(s3gen_file)}")
                
                # Check if it's from the forked repository
                if "chatterbox_embed" in s3gen_file:
                    logger.info("✅ s3gen is from forked repository")
                else:
                    logger.warning("⚠️ s3gen is NOT from forked repository")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not determine s3gen file path: {e}")
        else:
            logger.error("❌ Model doesn't have s3gen attribute")
            
        # Check for generate_long_text method
        logger.info(f"🔎 Has method generate_long_text: {hasattr(model, 'generate_long_text')}")
        if hasattr(model, 'generate_long_text'):
            logger.info("✅ Model has generate_long_text method - can use built-in chunking/stitching")
        else:
            logger.warning("⚠️ Model doesn't have generate_long_text method - will use fallback")

        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TTS_GENERATED_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def generate_voice_sample(voice_id, text, profile_base64, language, is_kids_voice, temp_profile_path, start_time):
    """Generate voice sample using saved voice profile"""
    global model, forked_handler
    
    logger.info("🎵 ===== VOICE SAMPLE GENERATION =====")
    logger.info(f"🔍 Parameters received:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  language: {language}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
    
    # Create output filename for voice sample (MP3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = VOICE_SAMPLES_DIR / language
    if is_kids_voice:
        local_dir = local_dir / "kids"
    
    logger.info(f"🔍 Local directory path: {local_dir}")
    
    # Create directory if it doesn't exist
    logger.info(f"🔍 Creating directory: {local_dir}")
    logger.info(f"🔍 Directory parent exists: {local_dir.parent.exists()}")
    logger.info(f"🔍 Directory parent: {local_dir.parent}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🔍 Directory created/exists: {local_dir.exists()}")
    logger.info(f"🔍 Directory absolute path: {local_dir.absolute()}")
    
    # Test write permissions
    try:
        test_file = local_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"🔍 Write permissions: OK")
    except Exception as e:
        logger.error(f"🔍 Write permissions: FAILED - {e}")
    
    sample_filename = local_dir / f"{voice_id}_sample_{timestamp}.mp3"  # Changed to .mp3
    logger.info(f"🎯 Voice sample local path: {sample_filename}")
    logger.info(f"🔍 Full absolute path: {sample_filename.absolute()}")
    
    try:
        logger.info("🔄 Starting voice sample processing...")
        
        # Use the model's built-in long text generation method
        logger.info(f"🎵 Using voice profile: {temp_profile_path}")
        
        if hasattr(model, 'generate_long_text'):
            logger.info("✅ Using model's built-in generate_long_text method")
            audio_tensor, sample_rate, metadata = model.generate_long_text(
                text=text,
                voice_profile_path=str(temp_profile_path),
                output_path=str(sample_filename),
                max_chars=500,      # Maximum characters per chunk
                pause_ms=150,        # Pause between chunks in milliseconds
                temperature=0.8,     # Generation temperature
                exaggeration=0.5,    # Voice exaggeration
                cfg_weight=0.5       # CFG weight
            )
            logger.info(f"✅ Generated using built-in long text method")
            logger.info(f"🔍 Metadata: {metadata}")
        else:
            logger.warning("⚠️ Model doesn't have generate_long_text, using fallback")
            # Fallback to simple generation
            audio_tensor = model.generate(
                text=text,
                voice_profile_path=str(temp_profile_path),
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Voice sample generated in {generation_time:.2f}s")
        logger.info(f"🔍 Audio tensor shape: {audio_tensor.shape}")
        
        # Convert to MP3 bytes
        mp3_bytes = tensor_to_mp3_bytes(audio_tensor, model.sr, "96k")
        logger.info(f"🎵 Converted to MP3: {len(mp3_bytes)} bytes")
        
        # Save MP3 file
        with open(sample_filename, 'wb') as f:
            f.write(mp3_bytes)
        
        logger.info(f"🔍 Final output path: {sample_filename}")
        logger.info(f"🔍 Output file exists: {sample_filename.exists()}")
        logger.info(f"🔍 File size: {sample_filename.stat().st_size if sample_filename.exists() else 'N/A'} bytes")
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Failed to generate voice sample after {generation_time:.2f}s")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error message: {str(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed to generate voice sample: {e}"}
    
    # Upload voice sample to Firebase
    if is_kids_voice:
        sample_firebase_path = f"audio/voices/{language}/kids/samples/{voice_id}_sample_{timestamp}.mp3"
    else:
        sample_firebase_path = f"audio/voices/{language}/samples/{voice_id}_sample_{timestamp}.mp3"
    
    logger.info(f"🎯 Voice sample Firebase path: {sample_firebase_path}")
    
    try:
        logger.info(f"🔍 About to read file: {sample_filename}")
        logger.info(f"🔍 File exists before reading: {sample_filename.exists()}")
        logger.info(f"🔍 File absolute path: {sample_filename.absolute()}")
        
        with open(sample_filename, 'rb') as f:
            sample_mp3_bytes = f.read()
        
        logger.info(f"🔍 Read {len(sample_mp3_bytes)} bytes from file")
        
        # Store metadata with the voice sample file
        sample_metadata = {
            'voice_id': voice_id,
            'file_type': 'voice_sample',
            'language': language,
            'is_kids_voice': str(is_kids_voice),
            'format': '96k_mp3',
            'timestamp': timestamp,
            'created_date': datetime.now().isoformat(),
            'model': 'chatterbox_tts',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'public, max-age=3600'
        }
        
        sample_uploaded = upload_to_firebase(
            sample_mp3_bytes,
            sample_firebase_path,
            "audio/mpeg",
            sample_metadata
        )
        
        if sample_uploaded:
            logger.info(f"🎵 Voice sample uploaded: {sample_firebase_path}")
            return {
                "status": "success",
                "sample_path": sample_firebase_path,
                "sample_url": sample_uploaded,
                "generation_time": generation_time,
                "model": "chatterbox_tts"
            }
        else:
            logger.error(f"❌ Failed to upload voice sample to Firebase")
            return {"status": "error", "message": "Failed to upload voice sample to Firebase"}
            
    except Exception as e:
        logger.error(f"❌ Failed to read/upload voice sample: {e}")
        return {"status": "error", "message": f"Failed to read/upload voice sample: {e}"}

def generate_tts_story(voice_id, text, profile_base64, language, story_type, is_kids_voice, temp_profile_path, start_time):
    """Generate TTS story using built-in long text generation"""
    global model, forked_handler
    
    logger.info("📖 ===== TTS STORY GENERATION =====")
    logger.info(f"🔍 Parameters received:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  language: {language}")
    logger.info(f"  story_type: {story_type}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
    
    # Create output filename for TTS story (MP3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = TTS_GENERATED_DIR / language / story_type
    logger.info(f"🔍 Local directory path: {local_dir}")
    
    # Create directory if it doesn't exist
    logger.info(f"🔍 Creating directory: {local_dir}")
    logger.info(f"🔍 Directory parent exists: {local_dir.parent.exists()}")
    logger.info(f"🔍 Directory parent: {local_dir.parent}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🔍 Directory created/exists: {local_dir.exists()}")
    logger.info(f"🔍 Directory absolute path: {local_dir.absolute()}")
    
    # Test write permissions
    try:
        test_file = local_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"🔍 Write permissions: OK")
    except Exception as e:
        logger.error(f"🔍 Write permissions: FAILED - {e}")
    
    tts_filename = local_dir / f"TTS_{voice_id}_{timestamp}.mp3"  # Changed to .mp3
    logger.info(f"🎯 TTS story local path: {tts_filename}")
    logger.info(f"🔍 Full absolute path: {tts_filename.absolute()}")
    
    try:
        logger.info("🔄 Starting TTS story processing...")
        
        # Use the model's built-in long text generation
        if hasattr(model, 'generate_long_text'):
            logger.info("✅ Using model's built-in generate_long_text method")
            audio_tensor, sample_rate, metadata = model.generate_long_text(
                text=text,
                voice_profile_path=str(temp_profile_path),
                output_path=str(tts_filename),
                max_chars=500,      # Maximum characters per chunk
                pause_ms=150,        # Pause between chunks in milliseconds
                temperature=0.8,     # Generation temperature
                exaggeration=0.5,    # Voice exaggeration
                cfg_weight=0.5       # CFG weight
            )
            logger.info(f"✅ TTS story generated using built-in method")
            logger.info(f"🔍 Metadata: {metadata}")
        else:
            logger.warning("⚠️ Model doesn't have generate_long_text, using fallback")
            # Fallback to simple generation
            audio_tensor = model.generate(
                text=text,
                voice_profile_path=str(temp_profile_path),
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5
            )
            # Convert to MP3 and save
            mp3_bytes = tensor_to_mp3_bytes(audio_tensor, model.sr, "96k")
            with open(tts_filename, 'wb') as f:
                f.write(mp3_bytes)
            metadata = {"chunk_count": 1, "duration_sec": audio_tensor.shape[-1] / model.sr}
        
        generation_time = time.time() - start_time
        logger.info(f"✅ TTS story generated in {generation_time:.2f}s")
        logger.info(f"🔍 Final output path: {tts_filename}")
        logger.info(f"🔍 Output file exists: {tts_filename.exists()}")
        logger.info(f"🔍 File size: {tts_filename.stat().st_size if tts_filename.exists() else 'N/A'} bytes")
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Failed to generate TTS story after {generation_time:.2f}s")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error message: {str(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Try with smaller chunks if it's a CUDA error
        if "CUDA error" in str(e) and len(text) > 300:
            logger.info("🔄 Retrying with smaller chunks due to CUDA error...")
            try:
                audio_tensor, sample_rate, metadata = model.generate_long_text(
                    text=text,
                    voice_profile_path=str(temp_profile_path),
                    output_path=str(tts_filename),
                    max_chars=300,       # Smaller chunks for better memory management
                    pause_ms=200,        # Longer pauses between chunks
                    temperature=0.7,     # Lower temperature for more consistent voice
                    exaggeration=0.6,    # Slightly more expressive
                    cfg_weight=0.4       # Lower CFG for more natural pacing
                )
                generation_time = time.time() - start_time
                logger.info(f"✅ TTS story generated with smaller chunks in {generation_time:.2f}s")
            except Exception as retry_error:
                logger.error(f"❌ Retry also failed: {retry_error}")
                return {"status": "error", "message": f"Failed to generate TTS story even with chunking: {retry_error}"}
        else:
            return {"status": "error", "message": f"Failed to generate TTS story: {e}"}
    
    # Upload TTS story to Firebase
    audio_path_firebase = f"audio/stories/{language}/{story_type}/TTS_{voice_id}_{timestamp}.mp3"  # Changed to .mp3
    
    logger.info(f"🎯 TTS story Firebase path: {audio_path_firebase}")
    
    try:
        logger.info(f"🔍 About to read file: {tts_filename}")
        logger.info(f"🔍 File exists before reading: {tts_filename.exists()}")
        logger.info(f"🔍 File absolute path: {tts_filename.absolute()}")
        
        with open(tts_filename, 'rb') as f:
            tts_mp3_bytes = f.read()
        
        logger.info(f"🔍 Read {len(tts_mp3_bytes)} bytes from file")
        
        # Store metadata with the TTS story file
        tts_metadata = {
            'voice_id': voice_id,
            'file_type': 'tts_story',
            'language': language,
            'story_type': story_type,
            'is_kids_voice': str(is_kids_voice),
            'format': '96k_mp3',
            'timestamp': timestamp,
            'created_date': datetime.now().isoformat(),
            'model': 'chatterbox_tts',
            'chunk_count': metadata.get('chunk_count', 1),
            'duration_sec': metadata.get('duration_sec', 0),
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'public, max-age=3600'
        }
        
        tts_uploaded = upload_to_firebase(
            tts_mp3_bytes,
            audio_path_firebase,
            "audio/mpeg",
            tts_metadata
        )
        
        if tts_uploaded:
            logger.info(f"📖 TTS story uploaded: {audio_path_firebase}")
            return {
                "status": "success",
                "tts_path": audio_path_firebase,
                "tts_url": tts_uploaded,
                "generation_time": generation_time,
                "model": "chatterbox_tts",
                "metadata": metadata
            }
        else:
            logger.error(f"❌ Failed to upload TTS story to Firebase")
            return {"status": "error", "message": "Failed to upload TTS story to Firebase"}
            
    except Exception as e:
        logger.error(f"❌ Failed to read/upload TTS story: {e}")
        return {"status": "error", "message": f"Failed to read/upload TTS story: {e}"}

def handler(event, responseFormat="base64"):
    """Handle TTS generation requests using saved voice embeddings"""
    global model, forked_handler
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "message": "Failed to initialize Firebase storage"}
    
    # Check if model is pre-loaded
    if model is None:
        logger.error("❌ ChatterboxTTS model not pre-loaded")
        return {"status": "error", "message": "ChatterboxTTS model not available"}
    
    logger.info("✅ Using pre-loaded ChatterboxTTS model")
    
    # Handle TTS generation request
    voice_id = event["input"].get("voice_id")
    text = event["input"].get("text")
    language = event["input"].get("language", "en")
    story_type = event["input"].get("story_type", "general")
    is_kids_voice = event["input"].get("is_kids_voice", False)
    
    if not voice_id or not text:
        return {"status": "error", "message": "Both voice_id and text are required"}

    logger.info(f"🎵 TTS request. Voice ID: {voice_id}")
    logger.info(f"📝 Text length: {len(text)} characters")
    logger.info(f"🌍 Language: {language}, Story type: {story_type}")
    logger.info(f"👶 Kids voice: {is_kids_voice}")
    
    start_time = time.time()
    
    try:
        # Load the voice profile
        profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
        logger.info(f"🔍 Looking for voice profile: {profile_path}")
        
        if not profile_path.exists():
            logger.error(f"❌ Voice profile not found: {profile_path}")
            return {"status": "error", "message": f"Voice profile not found for {voice_id}"}
        
        logger.info(f"✅ Voice profile found: {profile_path}")
        
        # Generate TTS story using built-in long text generation
        result = generate_tts_story(
            voice_id=voice_id,
            text=text,
            profile_base64=None,  # Not needed since we have the file
            language=language,
            story_type=story_type,
            is_kids_voice=is_kids_voice,
            temp_profile_path=profile_path,
            start_time=start_time
        )
        
        if result["status"] == "success":
            logger.info(f"✅ TTS generation completed successfully")
            return result
        else:
            logger.error(f"❌ TTS generation failed: {result.get('message', 'Unknown error')}")
            return result
            
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ TTS generation failed after {generation_time:.2f}s")
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"TTS generation failed: {e}"}

def handle_file_download(input):
    """Handle file download requests"""
    file_path = input.get("file_path")
    if not file_path:
        return {"status": "error", "message": "file_path is required"}
    
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return {
            "status": "success",
            "file_data": base64.b64encode(file_data).decode('utf-8'),
            "file_size": len(file_data)
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file: {e}"}

def list_available_files():
    """List all available files in the directories"""
    files = {}
    
    for directory_name, directory_path in [
        ("voice_profiles", VOICE_PROFILES_DIR),
        ("voice_samples", VOICE_SAMPLES_DIR),
        ("tts_generated", TTS_GENERATED_DIR),
        ("temp_voice", TEMP_VOICE_DIR)
    ]:
        if directory_path.exists():
            files[directory_name] = [f.name for f in directory_path.glob("*")]
        else:
            files[directory_name] = []
    
    return {"status": "success", "files": files}

if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    logger.info("✅ TTS Handler ready")
    runpod.serverless.start({'handler': handler }) 
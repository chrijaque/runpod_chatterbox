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
import numpy as np  # Added for MP3 conversion
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

model = None
forked_handler = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples") 
TEMP_VOICE_DIR = Path("/temp_voice")

# Create directories if they don't exist (RunPod deployment)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

logger.info(f"Using directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# Pre-load ChatterboxTTS model at module level (avoids re-initialization)
logger.info("🔧 Pre-loading ChatterboxTTS model...")
try:
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device='cuda')
    logger.info("✅ ChatterboxTTS model pre-loaded successfully")
    
    # Initialize the forked repository handler if available
    if FORKED_HANDLER_AVAILABLE:
        logger.info("🔧 Pre-loading ChatterboxVC...")
        try:
            from chatterbox.vc import ChatterboxVC
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
    
    # Attach the T3 text‑to‑token encoder to S3Gen
    if hasattr(model, "s3gen") and hasattr(model, "t3"):
        model.s3gen.text_encoder = model.t3
        logger.info("📌 Attached text_encoder to model.s3gen")
        
except Exception as e:
    logger.error(f"❌ Failed to pre-load ChatterboxTTS model: {e}")
    model = None
    forked_handler = None

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

def get_voice_id(name):
    """Generate a unique ID for a voice based on the name"""
    # Create a clean, filesystem-safe voice ID from the name
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name.lower().replace(' ', '_'))
    return f"voice_{clean_name}"

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")
            
def generate_template_message(name):
    """Generate the template message for the voice clone"""
    return f"Hello, this is the voice clone of {name}. This voice is used to narrate whimsical stories and fairytales."

def save_voice_profile(temp_voice_file, voice_id):
    """Save voice profile using forked repository's create_voice_clone method"""
    global model, forked_handler
    
    # Get final profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"💾 Saving voice profile using forked repository: {profile_path}")
    
    # Check if profile already exists
    if profile_path.exists():
        logger.info(f"✅ Voice profile already exists for {voice_id}")
        return profile_path
    
    try:
        # Use forked repository's create_voice_clone method
        if forked_handler is not None:
            logger.info(f"📁 Using forked repository ChatterboxVC.create_voice_clone method")
            result = forked_handler.create_voice_clone(
                audio_file_path=str(temp_voice_file),
                voice_id=voice_id,
                output_dir=str(VOICE_PROFILES_DIR)
            )
            
            if result["status"] == "success":
                logger.info(f"✅ Voice profile created using forked repository: {result.get('profile_path', profile_path)}")
                return profile_path
            else:
                logger.error(f"❌ Failed to create voice profile: {result.get('message', 'Unknown error')}")
                raise RuntimeError(f"Voice profile creation failed: {result.get('message', 'Unknown error')}")
        else:
            # Fallback to model method
            if hasattr(model, 'save_voice_profile'):
                model.save_voice_profile(str(temp_voice_file), str(profile_path))
                logger.info(f"✅ Voice profile saved using model method to: {profile_path}")
                return profile_path
            else:
                logger.error(f"❌ No voice profile creation method available")
                raise RuntimeError("No voice profile creation method available")
                
    except Exception as e:
        logger.error(f"Failed to save voice profile: {e}")
        raise

def load_voice_profile(voice_id):
    """Load existing voice profile using forked repository"""
    global model, forked_handler
    
    # Get profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"🔍 Loading voice profile from: {profile_path}")
    
    if not profile_path.exists():
        raise FileNotFoundError(f"No voice profile found for {voice_id}")
    
    try:
        # Use forked repository's set_voice_profile method
        if forked_handler is not None:
            logger.info(f"📁 Using forked repository ChatterboxVC.set_voice_profile method")
            forked_handler.set_voice_profile(str(profile_path))
            logger.info(f"✅ Voice profile loaded using forked repository from {profile_path}")
            return True  # Profile is now set in the forked handler
        else:
            # Fallback to model method
            if hasattr(model, 'load_voice_profile'):
                profile = model.load_voice_profile(str(profile_path))
                logger.info(f"✅ Loaded voice profile using model method from {profile_path}")
                return profile
            else:
                logger.warning(f"Enhanced profile loading not available, will use original audio file method for {voice_id}")
                return None
        
    except Exception as e:
        logger.error(f"Failed to load voice profile: {e}")
        raise

def handler(event, responseFormat="base64"):
    input = event['input']    
    
    # This handler is for voice cloning only
    return handle_voice_clone_request(input, responseFormat)

def handle_voice_clone_request(input, responseFormat):
    """Handle voice cloning requests"""
    global forked_handler
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "message": "Failed to initialize Firebase storage"}
    
    # Check if model is pre-loaded
    global model
    if model is None:
        logger.error("❌ ChatterboxTTS model not pre-loaded")
        return {"status": "error", "message": "ChatterboxTTS model not available"}
    
    logger.info("✅ Using pre-loaded ChatterboxTTS model")
    
    # Handle voice generation request only
    name = input.get('name')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio
    responseFormat = input.get('responseFormat', 'base64')  # Response format from frontend
    language = input.get('language', 'en')  # Language for storage organization
    is_kids_voice = input.get('is_kids_voice', False)  # Kids voice flag

    if not name or not audio_data:
        return {"status": "error", "message": "Both name and audio_data are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    logger.info(f"Language: {language}, Kids voice: {is_kids_voice}")
    
    # Generate the template message
    template_message = generate_template_message(name)
    logger.info(f"Generated template message: {template_message}")
    
    try:
        # Generate a unique voice ID based on the name
        voice_id = get_voice_id(name)
        logger.info(f"Generated voice ID: {voice_id}")
        
        # Save the uploaded audio to temp directory for embedding extraction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
        audio_bytes = base64.b64decode(audio_data)  # Still need this for now since FastAPI sends base64
        with open(temp_voice_file, 'wb') as f:
            f.write(audio_bytes)
        logger.info(f"Saved temporary voice file to {temp_voice_file}")

        # Step 1: Create voice profile (keep original quality)
        profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
        if profile_path.exists():
            logger.info(f"🎵 Loading existing profile: {voice_id}")
            profile = load_voice_profile(voice_id)
        else:
            logger.info(f"🎵 Creating new profile: {voice_id}")
            save_voice_profile(temp_voice_file, voice_id)
            profile = load_voice_profile(voice_id)
        
        # Step 2: Convert and save recorded audio (160 kbps MP3)
        recorded_audio_path = None
        try:
            # Convert original recording to 160 kbps MP3
            temp_mp3_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}_recorded.mp3"
            convert_audio_file_to_mp3(str(temp_voice_file), str(temp_mp3_file), "160k")
            
            # Upload recorded audio to Firebase
            with open(temp_mp3_file, 'rb') as f:
                recorded_mp3_bytes = f.read()
            
            if is_kids_voice:
                recorded_firebase_path = f"audio/voices/{language}/kids/recorded/{voice_id}_{timestamp}.mp3"
            else:
                recorded_firebase_path = f"audio/voices/{language}/recorded/{voice_id}_{timestamp}.mp3"
            
            # Store metadata with the recorded audio file
            recorded_metadata = {
                'voice_id': voice_id,
                'voice_name': name,
                'file_type': 'recorded_audio',
                'language': language,
                'is_kids_voice': str(is_kids_voice),
                'format': '160k_mp3',
                'timestamp': timestamp,
                'created_date': datetime.now().isoformat(),
                'model': 'chatterbox_tts',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600'
            }
            
            recorded_uploaded = upload_to_firebase(
                recorded_mp3_bytes,
                recorded_firebase_path,
                "audio/mpeg",
                recorded_metadata
            )
            if recorded_uploaded:
                recorded_audio_path = recorded_firebase_path
                logger.info(f"🎵 Recorded audio uploaded: {recorded_firebase_path}")
            
            # Clean up temp MP3 file
            os.unlink(temp_mp3_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert/upload recorded audio: {e}")
        
        # Track which generation method was used
        generation_method = "unknown"
        
        # Step 3: Generate voice sample using forked repository
        audio_tensor = None
        if forked_handler is not None:
            # Use forked repository's generate_voice_sample method
            logger.info("🔄 Using forked repository's generate_voice_sample method")
            generation_method = "forked_repository"
            
            try:
                # Use the forked repository's built-in voice sample generation
                result = forked_handler.generate_voice_sample(
                    voice_profile_path=str(profile_path),
                    text=template_message
                )
                
                if isinstance(result, tuple) and len(result) == 2:
                    audio_tensor, mp3_bytes = result
                    logger.info("✅ Used forked repository's generate_voice_sample method")
                else:
                    # Fallback to model method
                    audio_tensor = model.generate(
                        text=template_message,
                        voice_profile_path=str(profile_path),
                        temperature=0.8,
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )
                    generation_method = "fallback_model_generate"
                    logger.info("✅ Used fallback model.generate method")
                    
            except Exception as e:
                logger.warning(f"⚠️ Forked repository method failed: {e}")
                # Fallback to model method
                audio_tensor = model.generate(
                    text=template_message,
                    voice_profile_path=str(profile_path),
                    temperature=0.8,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                generation_method = "fallback_model_generate"
                logger.info("✅ Used fallback model.generate method")
        else:
            # Use model method directly
            logger.info("🔄 Using model method directly")
            audio_tensor = model.generate(
                text=template_message,
                voice_profile_path=str(profile_path),
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5
            )
            generation_method = "model_generate"

        # Convert audio tensor to MP3 bytes (96 kbps)
        sample_mp3_bytes = tensor_to_mp3_bytes(audio_tensor, model.sr, "96k")
        logger.info(f"🎵 Generated voice sample in MP3 format: {len(sample_mp3_bytes)} bytes")
        
        # Upload voice sample to Firebase
        sample_audio_path = None
        try:
            if is_kids_voice:
                sample_firebase_path = f"audio/voices/{language}/kids/samples/{voice_id}_sample_{timestamp}.mp3"
            else:
                sample_firebase_path = f"audio/voices/{language}/samples/{voice_id}_sample_{timestamp}.mp3"
            
            # Store metadata with the voice sample file
            sample_metadata = {
                'voice_id': voice_id,
                'voice_name': name,
                'file_type': 'voice_sample',
                'language': language,
                'is_kids_voice': str(is_kids_voice),
                'format': '96k_mp3',
                'timestamp': timestamp,
                'created_date': datetime.now().isoformat(),
                'generation_method': generation_method,
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
                sample_audio_path = sample_firebase_path
                logger.info(f"🎵 Voice sample uploaded: {sample_firebase_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload voice sample: {e}")

        # Clean up temporary voice file
        try:
            os.unlink(temp_voice_file)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if 'temp_voice_file' in locals():
            try:
                os.unlink(temp_voice_file)
            except:
                pass
        return {"status": "error", "message": str(e)}

    # Upload voice profile to Firebase
    profile_path_firebase = None
    if profile_path.exists():
        try:
            with open(profile_path, 'rb') as f:
                profile_data = f.read()
            
            if is_kids_voice:
                profile_firebase_path = f"audio/voices/{language}/kids/profiles/{voice_id}.npy"
            else:
                profile_firebase_path = f"audio/voices/{language}/profiles/{voice_id}.npy"
            
            # Store metadata with the voice profile file
            profile_metadata = {
                'voice_id': voice_id,
                'voice_name': name,
                'file_type': 'voice_profile',
                'language': language,
                'is_kids_voice': str(is_kids_voice),
                'format': 'npy',
                'created_date': datetime.now().isoformat(),
                'model': 'chatterbox_tts',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600'
            }
            
            profile_uploaded = upload_to_firebase(
                profile_data,
                profile_firebase_path,
                "application/octet-stream",
                profile_metadata
            )
            if profile_uploaded:
                profile_path_firebase = profile_firebase_path
                logger.info(f"📦 Profile uploaded: {profile_firebase_path}")
        except Exception as e:
            logger.error(f"Failed to upload profile file: {e}")
    else:
        logger.error(f"❌ Profile file does not exist: {profile_path}")

    # Log final summary (minimal)
    logger.info(f"🎯 Generation method: {generation_method}")
    logger.info(f"📦 Profile uploaded: {'YES' if profile_path_firebase else 'NO'}")
    logger.info(f"🎵 Recorded audio uploaded: {'YES' if recorded_audio_path else 'NO'}")
    logger.info(f"🎵 Voice sample uploaded: {'YES' if sample_audio_path else 'NO'}")
    
    # Return response
    response = {
        "status": "success",
        "voice_id": voice_id,
        "profile_path": profile_path_firebase,
        "sample_path": sample_audio_path,
        "recorded_path": recorded_audio_path,
        "profile_url": profile_path_firebase,  # Use path as URL for compatibility
        "sample_url": sample_audio_path,  # Use path as URL for compatibility
        "recorded_url": recorded_audio_path,  # Use path as URL for compatibility
        "generation_time": time.time() - start_time if 'start_time' in locals() else 0,
        "model": "chatterbox_tts",
        # Add metadata fields
        "sample_audio_path": sample_audio_path,
        "embedding_path": profile_path_firebase,
        "voice_name": name,
        "created_date": int(time.time()),
        # Keep original metadata for debugging
        "metadata": {
            "sample_rate": model.sr,
            "audio_shape": list(audio_tensor.shape) if 'audio_tensor' in locals() and audio_tensor is not None else None,
            "voice_id": voice_id,
            "voice_name": name,
            "profile_path_local": str(profile_path),
            "profile_exists": profile_path.exists(),
            "has_profile_support": hasattr(model, 'save_voice_profile') and hasattr(model, 'load_voice_profile'),
            "generation_method": generation_method,
            "template_message": template_message,
            "forked_handler_used": forked_handler is not None,
            "language": language,
            "is_kids_voice": is_kids_voice,
            "recorded_format": "160k_mp3",
            "sample_format": "96k_mp3"
        }
    }
    logger.info(f"📤 Voice clone completed successfully")
    return response 



if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

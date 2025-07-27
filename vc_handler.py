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

# -------------------------------------------------------------------
# 🐞  Firebase / GCS credential debug helper
# -------------------------------------------------------------------
def _debug_gcs_creds():
    import os, json, textwrap, pathlib, socket, ssl
    from google.auth import exceptions as gauth_exc
    logger.info("🔍 GCS-Debug | GOOGLE_APPLICATION_CREDENTIALS=%s",
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    logger.info("🔍 GCS-Debug | RUNPOD_SECRET envs: %s",
                [k for k in os.environ if k.startswith("RUNPOD_SECRET")])
    
    # Check RunPod Firebase secret specifically
    firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
    logger.info("🔍 GCS-Debug | RUNPOD_SECRET_Firebase=%s", firebase_secret_path)
    
    # Check if it's a file path or actual JSON content
    if firebase_secret_path:
        if firebase_secret_path.startswith('{'):
            logger.info("🔍 GCS-Debug | RUNPOD_SECRET_Firebase appears to be JSON content (starts with '{')")
            logger.info("🔍 GCS-Debug | JSON content preview: %s", firebase_secret_path[:200] + "..." if len(firebase_secret_path) > 200 else firebase_secret_path)
        elif os.path.exists(firebase_secret_path):
            logger.info("🔍 GCS-Debug | RUNPOD_SECRET_Firebase appears to be a file path (exists)")
        else:
            logger.info("🔍 GCS-Debug | RUNPOD_SECRET_Firebase is neither JSON content nor existing file path")
    else:
        logger.info("🔍 GCS-Debug | RUNPOD_SECRET_Firebase is None or empty")
    
    # 1) Does the expected file exist?
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/secrets/firebase.json")
    logger.info("🔍 GCS-Debug | Checking file %s", cred_path)
    p = pathlib.Path(cred_path)
    if p.exists():
        logger.info("✅ GCS-Debug | File exists (%d bytes)", p.stat().st_size)
        try:
            with p.open() as fp:
                first_line = fp.readline(256)
            logger.info("🔍 GCS-Debug | File starts with: %s",
                        textwrap.shorten(first_line.strip(), 120))
        except Exception as e:
            logger.warning("⚠️  GCS-Debug | Could not read file: %s", e)
    else:
        logger.error("❌ GCS-Debug | File NOT found on disk")

    # Check RunPod secret file if different and it's actually a file path
    if firebase_secret_path and firebase_secret_path != cred_path and not firebase_secret_path.startswith('{'):
        logger.info("🔍 GCS-Debug | Checking RunPod secret file %s", firebase_secret_path)
        p2 = pathlib.Path(firebase_secret_path)
        if p2.exists():
            logger.info("✅ GCS-Debug | RunPod secret file exists (%d bytes)", p2.stat().st_size)
            try:
                with p2.open() as fp:
                    first_line = fp.readline(256)
                logger.info("🔍 GCS-Debug | RunPod secret file starts with: %s",
                            textwrap.shorten(first_line.strip(), 120))
            except Exception as e:
                logger.warning("⚠️  GCS-Debug | Could not read RunPod secret file: %s", e)
        else:
            logger.error("❌ GCS-Debug | RunPod secret file NOT found on disk")
    elif firebase_secret_path and firebase_secret_path.startswith('{'):
        logger.info("🔍 GCS-Debug | RunPod secret is JSON content, skipping file existence check")

    # 2) Try manual credential load from RunPod secret
    if firebase_secret_path and os.path.exists(firebase_secret_path):
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(firebase_secret_path)
            logger.info("✅ GCS-Debug | Loaded RunPod creds for project_id=%s, client_email=%s",
                        creds.project_id, creds.service_account_email)
        except FileNotFoundError:
            logger.error("❌ GCS-Debug | FileNotFoundError for RunPod secret")
        except gauth_exc.DefaultCredentialsError as e:
            logger.error("❌ GCS-Debug | DefaultCredentialsError for RunPod secret: %s", e)
        except Exception as e:
            logger.error("❌ GCS-Debug | Unexpected error for RunPod secret: %s (%s)", e, type(e).__name__)

    # 3) Try manual credential load from fallback path
    try:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(cred_path)
        logger.info("✅ GCS-Debug | Loaded fallback creds for project_id=%s, client_email=%s",
                    creds.project_id, creds.service_account_email)
    except FileNotFoundError:
        logger.error("❌ GCS-Debug | FileNotFoundError for fallback path")
    except gauth_exc.DefaultCredentialsError as e:
        logger.error("❌ GCS-Debug | DefaultCredentialsError for fallback: %s", e)
    except Exception as e:
        logger.error("❌ GCS-Debug | Unexpected error for fallback: %s (%s)", e, type(e).__name__)

    # 4) Quick network check to storage.googleapis.com
    try:
        sock = socket.create_connection(("storage.googleapis.com", 443), timeout=3)
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(sock, server_hostname="storage.googleapis.com"):
            logger.info("✅ GCS-Debug | TLS handshake to storage.googleapis.com ok")
    except Exception as e:
        logger.warning("⚠️  GCS-Debug | Network to GCS failed: %s", e)

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    # Call debug helper first
    logger.info("🔍 ===== FIREBASE INITIALIZATION DEBUG =====")
    _debug_gcs_creds()
    logger.info("🔍 ===== END FIREBASE INITIALIZATION DEBUG =====")
    
    try:
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

def upload_to_firebase(data: bytes, dst: str, ctype: str):
    """Upload data to Firebase and return success status"""
    global bucket
    
    logger.info(f"🔍 Upload-Debug | Starting upload: {dst} ({len(data)} bytes, {ctype})")
    
    if bucket is None:
        logger.info("🔍 Upload-Debug | Bucket is None, initializing Firebase...")
        if not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot upload")
            return False
        logger.info("🔍 Upload-Debug | Firebase initialized, bucket: %s", bucket.name if bucket else "None")
    
    try:
        logger.info(f"🔍 Upload-Debug | Creating blob: {dst}")
        blob = bucket.blob(dst)
        logger.info(f"🔍 Upload-Debug | Blob created, uploading {len(data)} bytes...")
        
        blob.upload_from_string(data, content_type=ctype)
        logger.info(f"✅ Uploaded to Firebase: {dst}")
        
        # Verify upload
        try:
            blob.reload()
            logger.info(f"🔍 Upload-Debug | Upload verified: {blob.name} ({blob.size} bytes)")
        except Exception as verify_error:
            logger.warning(f"⚠️ Upload-Debug | Could not verify upload: {verify_error}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload to Firebase: {e}")
        logger.error(f"🔍 Upload-Debug | Error type: {type(e).__name__}")
        logger.error(f"🔍 Upload-Debug | Error details: {str(e)}")
        return False

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
        
        # Attach the T3 text‑to‑token encoder to S3Gen
        if hasattr(model, "s3gen") and hasattr(model, "t3"):
            model.s3gen.text_encoder = model.t3
            logger.info("📌 Attached text_encoder to model.s3gen")
        
        logger.info("✅ Model initialized on CUDA")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

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
    """Save voice profile directly to target location"""
    global model, forked_handler
    
    # Get final profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"💾 Saving voice profile directly to: {profile_path}")
    
    # Check if profile already exists
    if profile_path.exists():
        logger.info(f"✅ Voice profile already exists for {voice_id}")
        return profile_path
    
    try:
        # Use forked repository handler if available
        if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
            logger.info(f"📁 Using forked repository ChatterboxVC.set_target_voice method")
            # ChatterboxVC doesn't have a save_voice_profile method, so we'll use the model method
            if hasattr(model, 'save_voice_profile'):
                model.save_voice_profile(str(temp_voice_file), str(profile_path))
                logger.info(f"✅ Voice profile saved using model method to: {profile_path}")
            else:
                logger.warning(f"⚠️ Model doesn't have save_voice_profile method")
        elif hasattr(model, 'save_voice_profile'):
            logger.info(f"📁 Using enhanced save_voice_profile method")
            
            # Pass the audio file path directly (not the loaded tensor)
            logger.info(f"🎵 Using audio file: {temp_voice_file}")
            
            # Save profile using file path
            model.save_voice_profile(str(temp_voice_file), str(profile_path))
            logger.info(f"✅ Voice profile saved directly to: {profile_path}")
        else:
            # Fallback: create a placeholder 
            logger.warning(f"Enhanced profile saving not available, creating placeholder for {voice_id}")
            with open(profile_path, 'w') as f:
                f.write(f"voice_id: {voice_id}")
            logger.info(f"📝 Created placeholder: {profile_path}")
            
        # Verify the file was created
        if profile_path.exists():
            file_size = profile_path.stat().st_size
            logger.info(f"✅ Verified profile file: {profile_path} ({file_size} bytes)")
        else:
            logger.error(f"❌ Profile file not created: {profile_path}")
                
        return profile_path
        
    except Exception as e:
        logger.error(f"Failed to save voice profile: {e}")
        raise

def load_voice_profile(voice_id):
    """Load existing voice profile"""
    global model, forked_handler
    
    # Get profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"🔍 Loading voice profile from: {profile_path}")
    
    if not profile_path.exists():
        raise FileNotFoundError(f"No voice profile found for {voice_id}")
    
    try:
        # Use forked repository handler if available
        if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
            logger.info(f"📁 Using forked repository ChatterboxVC - will set target voice from profile")
            # ChatterboxVC doesn't have a load_voice_profile method, so we'll use the model method
            if hasattr(model, 'load_voice_profile'):
                profile = model.load_voice_profile(str(profile_path))
                logger.info(f"✅ Loaded voice profile using model method from {profile_path}")
                return profile
            else:
                logger.warning(f"⚠️ Model doesn't have load_voice_profile method")
        elif hasattr(model, 'load_voice_profile'):
            # Use enhanced method from forked repository
            profile = model.load_voice_profile(str(profile_path))
            logger.info(f"✅ Loaded voice profile from {profile_path}")
            return profile
        else:
            # Fallback: return None to indicate we should use the original audio file method
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
    
    # Handle voice generation request only
    name = input.get('name')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio
    responseFormat = input.get('responseFormat', 'base64')  # Response format from frontend

    if not name or not audio_data:
        return {"status": "error", "message": "Both name and audio_data are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    
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

        # Try to load existing profile, or create new one
        profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
        if profile_path.exists():
            logger.info(f"🎵 Loading existing profile: {voice_id}")
            profile = load_voice_profile(voice_id)
        else:
            logger.info(f"🎵 Creating new profile: {voice_id}")
            save_voice_profile(temp_voice_file, voice_id)
            profile = load_voice_profile(voice_id)
        
        # Track which generation method was used
        generation_method = "unknown"
        
        # Generate speech with the template message
        if profile is not None:
            # Use profile-based generation (forked repository method)
            logger.info("🔄 Using profile-based generation")
            generation_method = "profile_based"
            
            # Use the correct high-level method: ChatterboxTTS.generate() with voice_profile_path
            logger.info("🔍 Using standard ChatterboxTTS.generate() with voice_profile_path")
            try:
                # Use the saved voice profile file directly
                profile_path_str = str(profile_path)
                logger.info(f"🎵 Using voice profile: {profile_path_str}")
                
                audio_tensor = model.generate(
                    text=template_message,
                    voice_profile_path=profile_path_str,  # ✅ Correct parameter name
                    temperature=0.8,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                generation_method = "standard_generate_with_voice_profile"
                logger.info("✅ Used standard ChatterboxTTS.generate() with voice_profile_path")
                
            except Exception as e:
                logger.warning(f"⚠️ Standard generate with voice_profile_path failed: {e}")
                logger.warning(f"⚠️ Exception type: {type(e).__name__}")
                
                # Fallback: Use original audio file method
                logger.warning("⚠️ Falling back to audio_prompt_path method")
                audio_tensor = model.generate(
                    text=template_message,
                    audio_prompt_path=str(temp_voice_file),
                    temperature=0.8
                )
                generation_method = "fallback_audio_prompt"
                logger.info("✅ Used fallback audio_prompt_path method")
        else:
            # Use original audio file method (fallback)
            logger.info("🔄 Using audio file method")
            audio_tensor = model.generate(template_message, audio_prompt_path=str(temp_voice_file))
            generation_method = "audio_file"

        # Generate output filename in voice_samples directory
        sample_filename = VOICE_SAMPLES_DIR / f"{voice_id}_sample_{timestamp}.wav"
        
        # Save as WAV directly to final location
        torchaudio.save(sample_filename, audio_tensor, model.sr)
        logger.info(f"💾 Saved sample: {sample_filename.name}")
        
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

    # Upload files to Firebase and get file paths
    profile_path_firebase = f"audio/voices/en/profiles/{voice_id}.npy"
    audio_path_firebase = f"audio/voices/en/samples/{voice_id}_sample_{timestamp}.wav"
    
    # Upload profile file
    profile_uploaded = False
    if profile_path.exists():
        try:
            with open(profile_path, 'rb') as f:
                profile_data = f.read()
            profile_uploaded = upload_to_firebase(
                profile_data,
                profile_path_firebase,
                "application/octet-stream"
            )
            logger.info(f"📦 Profile uploaded: {profile_path_firebase}")
        except Exception as e:
            logger.error(f"Failed to upload profile file: {e}")
    else:
        logger.error(f"❌ Profile file does not exist: {profile_path}")
    
    # Upload audio sample
    audio_uploaded = False
    try:
        with open(sample_filename, 'rb') as f:
            wav_bytes = f.read()
        audio_uploaded = upload_to_firebase(
            wav_bytes,
            audio_path_firebase,
            "audio/wav"
        )
        logger.info(f"🎵 Audio uploaded: {audio_path_firebase}")
    except Exception as e:
        logger.error(f"Failed to upload audio file: {e}")

    # Log final summary (minimal)
    logger.info(f"🎯 Generation method: {generation_method}")
    logger.info(f"📦 Profile uploaded: {'YES' if profile_uploaded else 'NO'}")
    logger.info(f"🎵 Audio uploaded: {'YES' if audio_uploaded else 'NO'}")
    
    # Return file paths instead of URLs
    response = {
        "status": "success",
        "profile_path": profile_path_firebase if profile_uploaded else None,
        "audio_path": audio_path_firebase if audio_uploaded else None,
        "metadata": {
            "sample_rate": model.sr,
            "audio_shape": list(audio_tensor.shape),
            "voice_id": voice_id,
            "voice_name": name,
            "profile_path_local": str(profile_path),
            "profile_exists": profile_path.exists(),
            "has_profile_support": hasattr(model, 'save_voice_profile') and hasattr(model, 'load_voice_profile'),
            "generation_method": generation_method,
            "sample_file": str(sample_filename),
            "template_message": template_message,
            "forked_handler_used": forked_handler is not None
        }
    }
    logger.info(f"📤 Voice clone completed successfully")
    return response 



if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    initialize_model()
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

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

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    try:
        storage_client = storage.Client()  # auto-reads credentials from GOOGLE_APPLICATION_CREDENTIALS
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase storage client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase storage: {e}")
        return False

def upload_to_firebase(data: bytes, dst: str, ctype: str):
    """Upload data to Firebase and return success status"""
    global bucket
    if bucket is None:
        if not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot upload")
            return False
    
    try:
        blob = bucket.blob(dst)
        blob.upload_from_string(data, content_type=ctype)
        logger.info(f"✅ Uploaded to Firebase: {dst}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload to Firebase: {e}")
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
        # Comprehensive repository verification
        import chatterbox
        import subprocess
        import os
        
        logger.info("🔍 ===== REPOSITORY VERIFICATION =====")
        logger.info(f"📂 chatterbox module path: {chatterbox.__file__}")
        logger.info(f"📂 chatterbox module location: {os.path.dirname(chatterbox.__file__)}")
        
        # Check if it's from the forked repository
        if 'chatterbox_embed' in chatterbox.__file__:
            logger.info("🎯 chatterbox module is from FORKED repository")
        else:
            logger.warning("⚠️ chatterbox module is from ORIGINAL repository")
        
        # Check pip package info
        try:
            pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts']).decode().strip()
            logger.info("📦 Pip package info:")
            for line in pip_info.split('\n'):
                if line.strip():
                    logger.info(f"   {line}")
            
            if 'chrijaque/chatterbox_embed' in pip_info:
                logger.info("✅ Pip shows forked repository")
            else:
                logger.warning("⚠️ Pip shows original repository")
        except Exception as e:
            logger.warning(f"⚠️ Could not get pip info: {e}")
        
        # Check git repository info if available
        try:
            repo_path = os.path.dirname(chatterbox.__file__)
            git_path = os.path.join(repo_path, '.git')
            
            if os.path.exists(git_path):
                logger.info(f"📁 Found git repository at: {repo_path}")
                try:
                    commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD']).decode().strip()
                    logger.info(f"🔢 Git commit: {commit_hash}")
                    remote_url = subprocess.check_output(['git', '-C', repo_path, 'remote', 'get-url', 'origin']).decode().strip()
                    logger.info(f"🌐 Git remote: {remote_url}")
                    
                    if 'chrijaque/chatterbox_embed' in remote_url:
                        logger.info("✅ Git confirms forked repository")
                    else:
                        logger.warning("⚠️ Git shows original repository")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get git info: {e}")
            else:
                logger.info("📁 No git repository found (PyPI package installation)")
        except Exception as e:
            logger.warning(f"⚠️ Could not check git info: {e}")
        
        logger.info("🔍 ===== END REPOSITORY VERIFICATION =====")
        
        model = ChatterboxTTS.from_pretrained(device='cuda')
        
        # Initialize the forked repository handler if available
        if FORKED_HANDLER_AVAILABLE:
            logger.info("🔧 Initializing ChatterboxVC from forked repository...")
            try:
                # ChatterboxVC needs to be initialized with the s3gen model and device
                forked_handler = ChatterboxVC(
                    s3gen=model.s3gen,
                    device=model.device
                )
                logger.info("✅ ChatterboxVC initialized successfully")
                
                # Log handler capabilities
                handler_methods = [method for method in dir(forked_handler) if not method.startswith('_')]
                logger.info(f"📋 Available handler methods: {handler_methods}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChatterboxVC: {e}")
                forked_handler = None
        else:
            logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")
        
        # Verify s3gen module source
        logger.info("🔍 ===== S3GEN VERIFICATION =====")
        if hasattr(model, "s3gen"):
            logger.info(f"📂 s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 s3gen class: {model.s3gen.__class__}")
            logger.info(f"📂 s3gen class file: {model.s3gen.__class__.__module__}")
            
            # Check s3gen module file path
            try:
                import chatterbox.models.s3gen.s3gen as s3gen_module
                logger.info(f"📂 s3gen module file: {s3gen_module.__file__}")
                
                if 'chatterbox_embed' in s3gen_module.__file__:
                    logger.info("🎯 s3gen module is from FORKED repository")
                else:
                    logger.warning("⚠️ s3gen module is from ORIGINAL repository")
            except Exception as e:
                logger.warning(f"⚠️ Could not check s3gen module file: {e}")
            
            # Check if inference_from_text exists and its source
            if hasattr(model.s3gen, 'inference_from_text'):
                method = getattr(model.s3gen, 'inference_from_text')
                logger.info(f"📂 inference_from_text source: {method.__code__.co_filename}")
                logger.info(f"📂 inference_from_text line: {method.__code__.co_firstlineno}")
                
                if 'chatterbox_embed' in method.__code__.co_filename:
                    logger.info("🎯 inference_from_text is from FORKED repository")
                else:
                    logger.warning("⚠️ inference_from_text is from ORIGINAL repository")
            else:
                logger.warning("⚠️ inference_from_text method does NOT exist")
                
            # List all methods on s3gen
            s3gen_methods = [method for method in dir(model.s3gen) if not method.startswith('_')]
            logger.info(f"📋 Available s3gen methods: {s3gen_methods}")
        else:
            logger.warning("⚠️ Model does not have s3gen attribute")
        
        logger.info("🔍 ===== END S3GEN VERIFICATION =====")
        
        # Attach the T3 text‑to‑token encoder to S3Gen so that
        # s3gen.inference_from_text() works
        if hasattr(model, "s3gen") and hasattr(model, "t3"):
            model.s3gen.text_encoder = model.t3
            logger.info("📌 Attached text_encoder (model.t3) to model.s3gen")
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

    # Log final summary
    logger.info(f"🎯 Final generation method: {generation_method}")
    logger.info(f"📂 Repository used: {'FORKED' if 'chatterbox_embed' in str(model.s3gen.__class__.__module__) else 'ORIGINAL'}")
    logger.info(f"🔧 Forked handler used: {'YES' if forked_handler is not None else 'NO'}")
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

import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
import torch
import logging
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
TTS_GENERATED_DIR = Path("/tts_generated")
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

def initialize_model():
    global model
    
    logger.info("🔧 ===== MODEL INITIALIZATION =====")
    
    if model is not None:
        logger.info("✅ Model already initialized")
        logger.info(f"✅ Model type: {type(model)}")
        return model
    
    logger.info("🔄 Initializing ChatterboxTTS model for TTS generation...")
    
    # Check CUDA availability
    logger.info("🔍 Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    logger.info(f"🔍 CUDA available: {cuda_available}")
    
    if not cuda_available:
        logger.error("❌ CUDA is required but not available")
        raise RuntimeError("CUDA is required but not available")
    
    logger.info(f"✅ CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"✅ CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"✅ CUDA device capability: {torch.cuda.get_device_capability(0)}")
    
    try:
        # Debug: Check which chatterbox repository is being used
        import chatterbox
        import os
        
        logger.info(f"📦 chatterbox module loaded from: {chatterbox.__file__}")
        
        # Enhanced Debug: Log chatterbox installation details with dependency analysis
        repo_path = os.path.dirname(chatterbox.__file__)
        git_path = os.path.join(repo_path, '.git')
        
        # Check if it's a git repository
        if os.path.exists(git_path):
            logger.info(f"📁 chatterbox installed as git repo: {repo_path}")
            try:
                import subprocess
                commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD']).decode().strip()
                logger.info(f"🔢 Git commit: {commit_hash}")
                remote_url = subprocess.check_output(['git', '-C', repo_path, 'remote', 'get-url', 'origin']).decode().strip()
                logger.info(f"🌐 Git remote: {remote_url}")
                
                # Check if it's the forked repository
                if 'chrijaque/chatterbox_embed' in remote_url:
                    logger.info("✅ This is the CORRECT forked repository!")
                else:
                    logger.error("❌ This is NOT the forked repository - using wrong repo!")
            except Exception as e:
                logger.warning(f"⚠️ Could not get git info: {e}")
        else:
            logger.error(f"📁 chatterbox not installed as git repo (no .git directory found)")
            logger.error(f"❌ This indicates PyPI package installation instead of git repo")
        
        # Check pip installation details
        try:
            import subprocess
            pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts']).decode().strip()
            logger.info(f"📋 Pip package info:\n{pip_info}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get pip info: {e}")
        
        # Check for dependency conflicts
        try:
            import subprocess
            deps = subprocess.check_output(['pip', 'list']).decode().strip()
            chatterbox_deps = [line for line in deps.split('\n') if 'chatterbox' in line.lower()]
            if chatterbox_deps:
                logger.info(f"🔍 Found chatterbox-related packages:\n" + '\n'.join(chatterbox_deps))
            else:
                logger.info("🔍 No chatterbox-related packages found in pip list")
        except Exception as e:
            logger.warning(f"⚠️ Could not check dependencies: {e}")
        
        logger.info("🔄 Loading ChatterboxTTS model...")
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully on CUDA device")
        logger.info(f"✅ Model type: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")

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

        # Check model capabilities
        logger.info("🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_profile: {hasattr(model, 'load_voice_profile')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_profile: {hasattr(model, 'save_voice_profile')}")
        
    except Exception as e:
        logger.error("❌ Failed to initialize model")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error message: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, TTS_GENERATED_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def handler(event, responseFormat="base64"):
    """Handle TTS generation requests using saved voice embeddings"""
    global model
    
    logger.info("🚀 ===== TTS HANDLER STARTED =====")
    logger.info(f"📥 Received event: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input = event.get('input', {})
    logger.info(f"📥 Input type: {type(input)}")
    logger.info(f"📥 Input keys: {list(input.keys()) if isinstance(input, dict) else 'Not a dict'}")
    
    # Extract TTS parameters
    text = input.get('text')
    voice_id = input.get('voice_id')
    profile_base64 = input.get('profile_base64')  # New: voice profile data
    responseFormat = input.get('responseFormat', 'base64')
    
    logger.info(f"📋 Extracted parameters:")
    logger.info(f"  - text: {text[:50]}{'...' if text and len(text) > 50 else ''} (length: {len(text) if text else 0})")
    logger.info(f"  - voice_id: {voice_id}")
    logger.info(f"  - has_profile_base64: {bool(profile_base64)}")
    logger.info(f"  - profile_size: {len(profile_base64) if profile_base64 else 0}")
    logger.info(f"  - responseFormat: {responseFormat}")
    
    if not text or not voice_id or not profile_base64:
        logger.error("❌ Missing required parameters")
        logger.error(f"  - text provided: {bool(text)}")
        logger.error(f"  - voice_id provided: {bool(voice_id)}")
        logger.error(f"  - profile_base64 provided: {bool(profile_base64)}")
        return {"status": "error", "message": "text, voice_id, and profile_base64 are required"}
    
    logger.info(f"🎤 TTS request validated: voice_id={voice_id}, text_length={len(text)}")
    
    try:
        logger.info("🔍 ===== VOICE EMBEDDING PROCESSING =====")
        
        # Check if model is initialized
        if model is None:
            logger.error("❌ Model not initialized")
            return {"status": "error", "message": "Model not initialized"}
        
        logger.info(f"✅ Model is initialized: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")
        
        # Decode the voice profile data
        logger.info("🔄 Decoding voice profile data...")
        try:
            profile_data = base64.b64decode(profile_base64)
            logger.info(f"✅ Voice profile data decoded: {len(profile_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to decode voice profile data: {e}")
            return {"status": "error", "message": f"Failed to decode voice profile data: {e}"}
        
        # Save the voice profile data to a temporary file
        logger.info("🔄 Saving voice profile data to temporary file...")
        temp_profile_path = TEMP_VOICE_DIR / f"{voice_id}_temp.npy"
        
        try:
            with open(temp_profile_path, 'wb') as f:
                f.write(profile_data)
            logger.info(f"✅ Temporary profile file created: {temp_profile_path}")
            logger.info(f"✅ File size: {temp_profile_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to save temporary profile file: {e}")
            return {"status": "error", "message": f"Failed to save temporary profile file: {e}"}
        
        # Check if model has the required method
        logger.info(f"🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_profile: {hasattr(model, 'load_voice_profile')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_profile: {hasattr(model, 'save_voice_profile')}")
        
        # Load the profile using the forked repository method
        if hasattr(model, 'load_voice_profile'):
            logger.info("🔄 Loading profile using load_voice_profile method...")
            profile = model.load_voice_profile(str(temp_profile_path))
            logger.info(f"✅ Voice profile loaded successfully")
            logger.info(f"✅ Profile type: {type(profile)}")
            if hasattr(profile, 'shape'):
                logger.info(f"✅ Profile shape: {profile.shape}")
            if hasattr(profile, 'dtype'):
                logger.info(f"✅ Profile dtype: {profile.dtype}")
        else:
            logger.error("❌ Model doesn't have load_voice_profile method")
            logger.error("❌ This suggests the forked repository features are not available")
            return {"status": "error", "message": "Voice profile support not available"}
        
        # Generate speech using the profile
        logger.info(f"🎵 TTS: {voice_id} | Text length: {len(text)}")
        
        start_time = time.time()
        
        try:
            logger.info("🔄 Generating TTS...")
            
            # Use the generate method with saved_voice_path
            audio_tensor = model.generate(
                text,
                saved_voice_path=str(temp_profile_path),
                temperature=0.7,
                exaggeration=0.6
            )
            
            generation_time = time.time() - start_time
            logger.info(f"✅ TTS generated in {generation_time:.2f}s | Shape: {audio_tensor.shape}")
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Failed to generate TTS after {generation_time:.2f}s")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"❌ Error details: {e}")
            return {"status": "error", "message": f"Failed to generate TTS: {e}"}
        
        # Save the generated TTS to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = TTS_GENERATED_DIR / f"tts_{voice_id}_{timestamp}.wav"
        
        try:
            torchaudio.save(str(tts_filename), audio_tensor, model.sr)
            logger.info(f"💾 Saved: {tts_filename.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save TTS file: {e}")
            # Continue anyway, don't fail the request
        
        # Convert to base64
        try:
            audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
            logger.info(f"📤 Base64: {len(audio_base64)} chars")
        except Exception as e:
            logger.error(f"❌ Failed to convert audio to base64: {e}")
            return {"status": "error", "message": f"Failed to convert audio to base64: {e}"}
        
        # Create response
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "voice_id": voice_id,
                "voice_name": voice_id.replace('voice_', ''),  # Extract name from ID
                "text_input": text,
                "generation_time": generation_time,
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "tts_file": str(tts_filename),
                "timestamp": timestamp
            }
        }
        
        logger.info(f"📤 Response ready | Time: {generation_time:.2f}s | File: {tts_filename.name}")
        
        # Clean up temporary profile file
        try:
            if temp_profile_path.exists():
                os.unlink(temp_profile_path)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")
        
        logger.info("🎉 TTS completed successfully")
        return response
        
    except Exception as e:
        logger.error("💥 ===== TTS HANDLER FAILED =====")
        logger.error(f"❌ TTS request failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            
            # Save audio tensor to temporary file
            torchaudio.save(tmp_filename, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_filename)
            
            # Encode as base64
            base64_data = base64.b64encode(audio_data).decode('utf-8')
            
            return base64_data
            
    except Exception as e:
        logger.error(f"❌ Error converting audio to base64: {e}")
        raise

if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    
    try:
        logger.info("🔧 Initializing model...")
        initialize_model()
        logger.info("✅ Model ready")
        
        logger.info("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({'handler': handler })
        
    except Exception as e:
        logger.error(f"💥 TTS Handler startup failed: {e}")
        raise 
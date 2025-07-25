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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

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

def initialize_model():
    global model
    
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
        # Quick check for forked repository
        import chatterbox
        import subprocess
        
        # Check if it's the forked repository
        try:
            pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts']).decode().strip()
            if 'chrijaque/chatterbox_embed' in pip_info:
                logger.info("✅ Using forked repository")
            else:
                logger.warning("⚠️ Not using forked repository")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify repository: {e}")
        
        model = ChatterboxTTS.from_pretrained(device='cuda')
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
    global model
    
    # Get final profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"💾 Saving voice profile directly to: {profile_path}")
    
    # Check if profile already exists
    if profile_path.exists():
        logger.info(f"✅ Voice profile already exists for {voice_id}")
        return profile_path
    
    try:
        if hasattr(model, 'save_voice_profile'):
            logger.info(f"📁 Using enhanced save_voice_profile method")
            
            # Pass the audio file path directly (not the loaded tensor)
            logger.info(f"🎵 Using audio file: {temp_voice_file}")
            
            # Save profile using file path
            model.save_voice_profile(str(temp_voice_file), str(profile_path))
            logger.info(f"✅ Voice profile saved directly to: {profile_path}")
            
            # Verify the file was created
            if profile_path.exists():
                file_size = profile_path.stat().st_size
                logger.info(f"✅ Verified profile file: {profile_path} ({file_size} bytes)")
            else:
                logger.error(f"❌ Profile file not created: {profile_path}")
                
        else:
            # Fallback: create a placeholder 
            logger.warning(f"Enhanced profile saving not available, creating placeholder for {voice_id}")
            with open(profile_path, 'w') as f:
                f.write(f"voice_id: {voice_id}")
            logger.info(f"📝 Created placeholder: {profile_path}")
                
        return profile_path
        
    except Exception as e:
        logger.error(f"Failed to save voice profile: {e}")
        raise

def load_voice_profile(voice_id):
    """Load existing voice profile"""
    global model
    
    # Get profile path
    profile_path = VOICE_PROFILES_DIR / f"{voice_id}.npy"
    logger.info(f"🔍 Loading voice profile from: {profile_path}")
    
    if not profile_path.exists():
        raise FileNotFoundError(f"No voice profile found for {voice_id}")
    
    try:
        if hasattr(model, 'load_voice_profile'):
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
        audio_bytes = base64.b64decode(audio_data)
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
            
            # Create reference dictionary for inference
            ref_dict = {
                "embedding": profile,
                "prompt_token": torch.zeros(1, 1, dtype=torch.long).to(model.device),
                "prompt_token_len": torch.tensor([1]).to(model.device),
                "prompt_feat": torch.zeros(1, 2, 80).to(model.device),
                "prompt_feat_len": None,
            }
            
            try:
                # Use the inference method with profiles
                audio_tensor = model.inference(template_message, ref_dict=ref_dict)
            except AttributeError:
                # Fallback to generate method if inference doesn't exist
                logger.warning("⚠️ Using fallback generation method")
                audio_tensor = model.generate(template_message, audio_prompt_path=str(temp_voice_file))
                generation_method = "profile_fallback"
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

    # Convert to base64 string
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
    
    # Read the profile file to include in response
    profile_base64 = None
    if profile_path.exists():
        try:
            with open(profile_path, 'rb') as f:
                profile_data = f.read()
            profile_base64 = base64.b64encode(profile_data).decode('utf-8')
            logger.info(f"📦 Profile ready: {len(profile_base64)} chars")
        except Exception as e:
            logger.error(f"Failed to read profile file: {e}")

    if responseFormat == "base64":
        # Return base64 - ALWAYS return structured JSON, never raw strings
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "profile_base64": profile_base64,  # Include profile file
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "voice_id": voice_id,
                "voice_name": name,
                "profile_path": str(profile_path),
                "profile_exists": profile_path.exists(),
                "has_profile_support": hasattr(model, 'save_voice_profile') and hasattr(model, 'load_voice_profile'),
                "generation_method": generation_method,
                "sample_file": str(sample_filename),
                "template_message": template_message
            }
        }
        logger.info(f"📤 Voice clone completed successfully")
        return response
    elif responseFormat == "binary":
        # Still return structured JSON, not raw data
        with open(sample_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        response = {
            "status": "success", 
            "audio_base64": audio_data,
            "profile_base64": profile_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "voice_id": voice_id,
                "voice_name": name,
                "profile_path": str(profile_path),
                "profile_exists": profile_path.exists(),
                "has_profile_support": hasattr(model, 'save_voice_profile') and hasattr(model, 'load_voice_profile'),
                "generation_method": generation_method,
                "sample_file": str(sample_filename),
                "template_message": template_message
            }
        }
        logger.info(f"📤 Voice clone completed successfully")
        return response

    # Default response format - ALWAYS return structured JSON
    response = {
        "status": "success",
        "audio_base64": audio_base64,
        "profile_base64": profile_base64,
        "metadata": {
            "sample_rate": model.sr,
            "audio_shape": list(audio_tensor.shape),
            "voice_id": voice_id,
            "voice_name": name,
            "profile_path": str(profile_path),
            "profile_exists": profile_path.exists(),
            "has_profile_support": hasattr(model, 'save_voice_profile') and hasattr(model, 'load_voice_profile'),
            "generation_method": generation_method,
            "sample_file": str(sample_filename),
            "template_message": template_message
        }
    }
    
    logger.info(f"📤 Voice clone completed successfully")
    return response

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode as base64
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error converting audio to base64: {e}")
        raise

if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    initialize_model()
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

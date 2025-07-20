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

# Local directory paths (created at startup)
VOICE_CLONES_DIR = Path("./voice_clones")
VOICE_SAMPLES_DIR = Path("./voice_samples") 
TEMP_VOICE_DIR = Path("./temp_voice")

# Create directories at startup
logger.info("Creating local directories for testing...")
VOICE_CLONES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

logger.info(f"✅ Directories created:")
logger.info(f"  VOICE_CLONES_DIR: {VOICE_CLONES_DIR.absolute()}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR.absolute()}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR.absolute()}")

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
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("Model initialized successfully on CUDA device")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def get_voice_id(name):
    """Generate a unique ID for a voice based on the name"""
    # Create a clean, filesystem-safe voice ID from the name
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', name.lower().replace(' ', '_'))
    return f"voice_{clean_name}"

def get_voice_library():
    """Get list of all created voices with their sample files"""
    voices = []
    
    try:
        # Check if directories exist
        if not VOICE_CLONES_DIR.exists() or not VOICE_SAMPLES_DIR.exists():
            logger.info("Voice directories don't exist yet")
            return voices
        
        # Get all .npy files (voice embeddings)
        embedding_files = list(VOICE_CLONES_DIR.glob("*.npy"))
        
        for embedding_file in embedding_files:
            # Extract voice_id from filename (remove .npy extension)
            voice_id = embedding_file.stem
            
            # Find corresponding sample files
            sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
            
            if sample_files:
                # Get the most recent sample file
                latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
                
                # Extract name from voice_id (remove voice_ prefix)
                display_name = voice_id.replace("voice_", "").replace("_", " ").title()
                
                voice_info = {
                    "voice_id": voice_id,
                    "name": display_name,
                    "sample_file": str(latest_sample),
                    "embedding_file": str(embedding_file),
                    "created_date": latest_sample.stat().st_mtime
                }
                voices.append(voice_info)
                
        # Sort by creation date (newest first)
        voices.sort(key=lambda x: x["created_date"], reverse=True)
        
        logger.info(f"Found {len(voices)} voices in library")
        
    except Exception as e:
        logger.error(f"Error getting voice library: {e}")
    
    return voices

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_CLONES_DIR, VOICE_SAMPLES_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")
            
def generate_template_message(name):
    """Generate the template message for the voice clone"""
    return f"Hello, this is the voice clone of {name}. This voice is used to narrate whimsical stories and fairytales."

def save_voice_embedding(temp_voice_file, voice_id):
    """Save voice embedding directly to target location"""
    global model
    
    # Get final embedding path
    embedding_path = VOICE_CLONES_DIR / f"{voice_id}.npy"
    logger.info(f"💾 Saving embedding directly to: {embedding_path}")
    
    # Check if embedding already exists
    if embedding_path.exists():
        logger.info(f"✅ Voice embedding already exists for {voice_id}")
        return embedding_path
    
    try:
        if hasattr(model, 'save_voice_clone'):
            logger.info(f"📁 Using enhanced save_voice_clone method")
            
            # Load audio from temp file
            logger.info(f"🎵 Loading audio from: {temp_voice_file}")
            audio_input, sr = torchaudio.load(temp_voice_file)
            logger.info(f"🎵 Audio loaded: shape={audio_input.shape}, sr={sr}")
            
            # Save embedding directly to final location
            model.save_voice_clone(audio_input, str(embedding_path))
            logger.info(f"✅ Embedding saved directly to: {embedding_path}")
            
            # Verify the file was created
            if embedding_path.exists():
                file_size = embedding_path.stat().st_size
                logger.info(f"✅ Verified embedding file: {embedding_path} ({file_size} bytes)")
            else:
                logger.error(f"❌ Embedding file not created: {embedding_path}")
                
        else:
            # Fallback: create a placeholder 
            logger.warning(f"Enhanced embedding not available, creating placeholder for {voice_id}")
            with open(embedding_path, 'w') as f:
                f.write(f"voice_id: {voice_id}")
            logger.info(f"📝 Created placeholder: {embedding_path}")
                
        return embedding_path
        
    except Exception as e:
        logger.error(f"Failed to save voice embedding: {e}")
        raise

def load_voice_embedding(voice_id):
    """Load existing voice embedding"""
    global model
    
    # Get embedding path
    embedding_path = VOICE_CLONES_DIR / f"{voice_id}.npy"
    logger.info(f"🔍 Loading embedding from: {embedding_path}")
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"No voice embedding found for {voice_id}")
    
    try:
        if hasattr(model, 'load_voice_clone'):
            # Use enhanced method from forked repository
            embedding = model.load_voice_clone(str(embedding_path))
            logger.info(f"✅ Loaded embedding from {embedding_path}")
            return embedding
        else:
            # Fallback: return None to indicate we should use the original audio file method
            logger.warning(f"Enhanced embedding not available, will use original audio file method for {voice_id}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to load voice embedding: {e}")
        raise

def handler(event, responseFormat="base64"):
    input = event['input']
    
    # Check if this is a library request
    request_type = input.get('request_type', 'generate')
    
    if request_type == 'get_library':
        logger.info("Handling voice library request")
        try:
            voices = get_voice_library()
            return {
                "status": "success",
                "request_type": "get_library",
                "voices": voices,
                "total_voices": len(voices)
            }
        except Exception as e:
            logger.error(f"Error handling library request: {e}")
            return {
                "status": "error", 
                "request_type": "get_library",
                "message": str(e)
            }
    
    elif request_type == 'get_sample':
        logger.info("Handling voice sample request")
        voice_id = input.get('voice_id')
        if not voice_id:
            return {"status": "error", "message": "voice_id is required for sample request"}
        
        try:
            # Find the sample file
            sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
            if not sample_files:
                return {"status": "error", "message": f"No sample found for voice_id: {voice_id}"}
            
            # Get the most recent sample
            latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
            
            # Read and encode the audio file
            with open(latest_sample, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info(f"Returning sample for voice_id: {voice_id}")
            return {
                "status": "success",
                "request_type": "get_sample",
                "voice_id": voice_id,
                "audio_base64": audio_data,
                "sample_file": str(latest_sample)
            }
            
        except Exception as e:
            logger.error(f"Error getting voice sample: {e}")
            return {
                "status": "error", 
                "request_type": "get_sample",
                "message": str(e)
            }
    
    # Handle voice generation request (existing logic)
    name = input.get('name')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio

    if not name or not audio_data:
        return {"status": "error", "message": "Both name and audio_data are required"}

    logger.info(f"New request. Voice clone name: {name}")
    
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

        # Try to load existing embedding, or create new one
        embedding_path = VOICE_CLONES_DIR / f"{voice_id}.npy"
        if embedding_path.exists():
            logger.info(f"Loading existing voice embedding for {voice_id}")
            embedding = load_voice_embedding(voice_id)
        else:
            logger.info(f"Creating new voice embedding for {voice_id}")
            save_voice_embedding(temp_voice_file, voice_id)
            embedding = load_voice_embedding(voice_id)
        
        # Generate speech with the template message
        generation_method = "audio-file-based"  # default
        if embedding is not None:
            # Use embedding-based generation (forked repository method)
            logger.info("Using embedding-based generation")
            generation_method = "embedding-based"
            
            # Create reference dictionary for inference
            ref_dict = {
                "embedding": embedding,
                "prompt_token": torch.zeros(1, 1, dtype=torch.long).to(model.device),
                "prompt_token_len": torch.tensor([1]).to(model.device),
                "prompt_feat": torch.zeros(1, 2, 80).to(model.device),
                "prompt_feat_len": None,
            }
            
            try:
                # Use the inference method with embeddings
                audio_tensor = model.inference(template_message, ref_dict=ref_dict)
            except AttributeError:
                # Fallback to generate method if inference doesn't exist
                logger.warning("inference method not available, using generate method")
                audio_tensor = model.generate(template_message, audio_prompt_path=str(temp_voice_file))
        else:
            # Use original audio file method (fallback)
            logger.info("Using original audio file method (fallback)")
            audio_tensor = model.generate(template_message, audio_prompt_path=str(temp_voice_file))

        # Generate output filename in voice_samples directory
        sample_filename = VOICE_SAMPLES_DIR / f"{voice_id}_sample_{timestamp}.wav"
        logger.info(f"💾 Saving audio sample directly to: {sample_filename}")
        
        # Save as WAV directly to final location
        logger.info(f"🎵 Audio tensor shape: {audio_tensor.shape}, Sample rate: {model.sr}")
        torchaudio.save(sample_filename, audio_tensor, model.sr)
        
        # Verify the file was created
        if sample_filename.exists():
            file_size = sample_filename.stat().st_size
            logger.info(f"✅ Sample saved: {sample_filename} ({file_size} bytes)")
        else:
            logger.error(f"❌ Sample file not created: {sample_filename}")
        
        # Clean up temporary voice file
        try:
            os.unlink(temp_voice_file)
            logger.info(f"🗑️ Cleaned up temp file: {temp_voice_file}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            
        # List final directory contents for debugging
        list_files_for_debug()

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

    if responseFormat == "base64":
        # Return base64
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "voice_id": voice_id,
                "voice_name": name,
                "embedding_path": str(embedding_path),
                "embedding_exists": embedding_path.exists(),
                "has_embedding_support": hasattr(model, 'save_voice_clone') and hasattr(model, 'load_voice_clone'),
                "generation_method": generation_method,
                "sample_file": str(sample_filename),
                "template_message": template_message
            }
        }
    elif responseFormat == "binary":
        with open(sample_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        response = audio_data  # Just return the base64 string

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
    initialize_model()
    runpod.serverless.start({'handler': handler })

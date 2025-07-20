import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
import torch
import logging
import hashlib
from chatterbox.tts import S3Token2Wav
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

# Create directories if they don't exist
TEMP_VOICE_DIR = Path("temp_voice")  # Temporary storage for processing
VOICE_CLONES_DIR = Path("voice_clones")  # Persistent .npy embeddings
VOICE_SAMPLES_DIR = Path("voice_samples")  # Generated audio samples
TEMP_VOICE_DIR.mkdir(exist_ok=True)
VOICE_CLONES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)

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
        model = S3Token2Wav.from_pretrained(device='cuda')
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

def get_embedding_path(voice_id):
    """Get the path for a voice embedding file"""
    return VOICE_CLONES_DIR / f"{voice_id}.npy"

def generate_template_message(name):
    """Generate the template message for the voice clone"""
    return f"Hello, this is the voice clone of {name}. This voice is used to narrate whimsical stories and fairytales."

def save_voice_embedding(voice_file_path, voice_id):
    """Save voice embedding for future use"""
    global model
    
    embedding_path = get_embedding_path(voice_id)
    
    # Check if embedding already exists
    if embedding_path.exists():
        logger.info(f"Voice embedding already exists for {voice_id}")
        return embedding_path
    
    try:
        # Load audio for embedding extraction
        audio_input, sr = torchaudio.load(voice_file_path)
        
        # Save voice clone embedding
        model.save_voice_clone(audio_input, sr, str(embedding_path))
        logger.info(f"Saved voice embedding to {embedding_path}")
        return embedding_path
        
    except Exception as e:
        logger.error(f"Failed to save voice embedding: {e}")
        raise

def load_voice_embedding(voice_id):
    """Load existing voice embedding"""
    global model
    
    embedding_path = get_embedding_path(voice_id)
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"No voice embedding found for {voice_id}")
    
    try:
        embedding = model.load_voice_clone(str(embedding_path))
        logger.info(f"Loaded voice embedding from {embedding_path}")
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to load voice embedding: {e}")
        raise

def handler(event, responseFormat="base64"):
    input = event['input']    
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
        embedding_path = get_embedding_path(voice_id)
        if embedding_path.exists():
            logger.info(f"Loading existing voice embedding for {voice_id}")
            embedding = load_voice_embedding(voice_id)
        else:
            logger.info(f"Creating new voice embedding for {voice_id}")
            save_voice_embedding(temp_voice_file, voice_id)
            embedding = load_voice_embedding(voice_id)
        
        # Create reference dictionary for inference
        ref_dict = {
            "embedding": embedding,
            "prompt_token": torch.zeros(1, 1, dtype=torch.long).to(model.device),
            "prompt_token_len": torch.tensor([1]).to(model.device),
            "prompt_feat": torch.zeros(1, 2, 80).to(model.device),
            "prompt_feat_len": None,
        }
        
        # Generate speech with the template message
        try:
            # Use the inference method with embeddings
            audio_tensor = model.inference(template_message, ref_dict=ref_dict)
        except AttributeError:
            # Fallback to generate method if inference doesn't exist
            logger.warning("Using fallback generate method - embeddings may not be properly utilized")
            audio_tensor = model.generate(template_message, audio_prompt_path=str(temp_voice_file))

        # Generate output filename in voice_samples directory
        sample_filename = VOICE_SAMPLES_DIR / f"{voice_id}_sample_{timestamp}.wav"
        
        # Save as WAV
        torchaudio.save(sample_filename, audio_tensor, model.sr)
        logger.info(f"Saved voice sample to {sample_filename}")
        
        # Clean up temporary voice file if embedding was created successfully
        if not embedding_path.exists() or embedding_path.stat().st_size == 0:
            logger.warning(f"Keeping temp voice file {temp_voice_file} as embedding creation may have failed")
        else:
            try:
                os.unlink(temp_voice_file)
                logger.info(f"Cleaned up temporary voice file {temp_voice_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp voice file {temp_voice_file}: {cleanup_error}")

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

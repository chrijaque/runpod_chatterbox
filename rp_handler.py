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

# Create output directories if they don't exist
OUTPUT_DIR = Path("output")
VOICE_DIR = Path("voice")
OUTPUT_DIR.mkdir(exist_ok=True)
VOICE_DIR.mkdir(exist_ok=True)

def initialize_model():
    global model
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    logger.info("Initializing ChatterboxTTS model...")
    
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

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio

    if not prompt or not audio_data:
        return {"status": "error", "message": "Both prompt and audio_data are required"}

    logger.info(f"New request. Prompt: {prompt}")
    
    try:
        # Save the uploaded audio to voice directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        voice_file = VOICE_DIR / f"voice_{timestamp}.{audio_format}"
        audio_bytes = base64.b64decode(audio_data)
        with open(voice_file, 'wb') as f:
            f.write(audio_bytes)
        logger.info(f"Saved voice file to {voice_file}")

        # Load and resample the audio if needed
        audio_input, sr = torchaudio.load(voice_file)
        if sr != 44100:  # Ensure consistent sample rate
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio_input = resampler(audio_input)

        # Prompt Chatterbox
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=str(voice_file)
        )

        # Generate output filename with timestamp
        output_filename = OUTPUT_DIR / f"output_{timestamp}.wav"
        
        # Save as WAV
        torchaudio.save(output_filename, audio_tensor, model.sr)
        logger.info(f"Saved output file to {output_filename}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if 'voice_file' in locals():
            try:
                os.unlink(voice_file)
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
                "voice_file": str(voice_file),
                "output_file": str(output_filename)
            }
        }
    elif responseFormat == "binary":
        with open(output_filename, 'rb') as f:
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

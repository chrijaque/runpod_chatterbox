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
VOICE_CLONES_DIR = Path("/voice_clones")
TTS_GENERATED_DIR = Path("/tts_generated")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_CLONES_DIR: {VOICE_CLONES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")

def initialize_model():
    global model
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    logger.info("Initializing ChatterboxTTS model for TTS generation...")
    
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

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_CLONES_DIR, TTS_GENERATED_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def handler(event, responseFormat="base64"):
    """Handle TTS generation requests using saved voice embeddings"""
    global model
    
    input = event['input']
    
    # Extract TTS parameters
    text = input.get('text')
    voice_id = input.get('voice_id')
    responseFormat = input.get('responseFormat', 'base64')
    
    if not text or not voice_id:
        return {"status": "error", "message": "Both text and voice_id are required"}
    
    logger.info(f"🎤 TTS request: voice_id={voice_id}, text_length={len(text)}")
    
    try:
        # Load the voice embedding
        embedding_path = VOICE_CLONES_DIR / f"{voice_id}.npy"
        if not embedding_path.exists():
            return {"status": "error", "message": f"Voice embedding not found for {voice_id}"}
        
        logger.info(f"📁 Loading voice embedding from: {embedding_path}")
        
        # Load the embedding using the forked repository method
        if hasattr(model, 'load_voice_clone'):
            embedding = model.load_voice_clone(str(embedding_path))
            logger.info(f"✅ Voice embedding loaded successfully")
        else:
            return {"status": "error", "message": "Voice embedding support not available"}
        
        # Generate speech using the embedding
        logger.info(f"🎵 Generating TTS with voice embedding...")
        start_time = time.time()
        
        try:
            # Use the generate method with saved_voice_path
            audio_tensor = model.generate(
                text,
                saved_voice_path=str(embedding_path),
                temperature=0.7,
                exaggeration=0.6
            )
            generation_time = time.time() - start_time
            logger.info(f"✅ TTS generated successfully in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate TTS: {e}")
            return {"status": "error", "message": f"Failed to generate TTS: {e}"}
        
        # Save the generated TTS to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = TTS_GENERATED_DIR / f"tts_{voice_id}_{timestamp}.wav"
        
        try:
            torchaudio.save(str(tts_filename), audio_tensor, model.sr)
            logger.info(f"💾 TTS saved to: {tts_filename} ({tts_filename.stat().st_size} bytes)")
        except Exception as e:
            logger.error(f"❌ Failed to save TTS file: {e}")
            # Continue anyway, don't fail the request
        
        # Convert to base64
        audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
        
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
        
        logger.info(f"📤 TTS Response: audio_base64 length={len(audio_base64)}, generation_time={generation_time:.2f}s, saved_to={tts_filename}")
        
        # List final directory contents for debugging
        list_files_for_debug()
        
        return response
        
    except Exception as e:
        logger.error(f"❌ TTS request failed: {e}")
        return {"status": "error", "message": str(e)}

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
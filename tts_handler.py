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
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_CLONES_DIR: {VOICE_CLONES_DIR}")
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
        logger.info("🔄 Loading ChatterboxTTS model...")
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully on CUDA device")
        logger.info(f"✅ Model type: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")
        
        # Check model capabilities
        logger.info("🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_clone (model.s3gen): {hasattr(model.s3gen, 'load_voice_clone')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_clone: {hasattr(model, 'save_voice_clone')}")
        
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
    for directory in [VOICE_CLONES_DIR, TTS_GENERATED_DIR]:
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
    embedding_base64 = input.get('embedding_base64')  # New: embedded voice data
    responseFormat = input.get('responseFormat', 'base64')
    
    logger.info(f"📋 Extracted parameters:")
    logger.info(f"  - text: {text[:50]}{'...' if text and len(text) > 50 else ''} (length: {len(text) if text else 0})")
    logger.info(f"  - voice_id: {voice_id}")
    logger.info(f"  - has_embedding_base64: {bool(embedding_base64)}")
    logger.info(f"  - embedding_size: {len(embedding_base64) if embedding_base64 else 0}")
    logger.info(f"  - responseFormat: {responseFormat}")
    
    if not text or not voice_id or not embedding_base64:
        logger.error("❌ Missing required parameters")
        logger.error(f"  - text provided: {bool(text)}")
        logger.error(f"  - voice_id provided: {bool(voice_id)}")
        logger.error(f"  - embedding_base64 provided: {bool(embedding_base64)}")
        return {"status": "error", "message": "text, voice_id, and embedding_base64 are required"}
    
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
        
        # Decode the embedded voice data
        logger.info("🔄 Decoding embedded voice data...")
        try:
            embedding_data = base64.b64decode(embedding_base64)
            logger.info(f"✅ Embedded voice data decoded: {len(embedding_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to decode embedded voice data: {e}")
            return {"status": "error", "message": f"Failed to decode embedded voice data: {e}"}
        
        # Save the embedded voice data to a temporary file
        logger.info("🔄 Saving embedded voice data to temporary file...")
        temp_embedding_path = TEMP_VOICE_DIR / f"{voice_id}_temp.npy"
        
        try:
            with open(temp_embedding_path, 'wb') as f:
                f.write(embedding_data)
            logger.info(f"✅ Temporary embedding file created: {temp_embedding_path}")
            logger.info(f"✅ File size: {temp_embedding_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to save temporary embedding file: {e}")
            return {"status": "error", "message": f"Failed to save temporary embedding file: {e}"}
        
        # Check if model has the required method
        logger.info(f"🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_clone (model.s3gen): {hasattr(model.s3gen, 'load_voice_clone')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_clone: {hasattr(model, 'save_voice_clone')}")
        
        # Load the embedding using the forked repository method
        if hasattr(model.s3gen, 'load_voice_clone'):
            logger.info("🔄 Loading embedding using load_voice_clone method...")
            embedding = model.s3gen.load_voice_clone(str(temp_embedding_path))
            logger.info(f"✅ Voice embedding loaded successfully")
            logger.info(f"✅ Embedding type: {type(embedding)}")
            if hasattr(embedding, 'shape'):
                logger.info(f"✅ Embedding shape: {embedding.shape}")
            if hasattr(embedding, 'dtype'):
                logger.info(f"✅ Embedding dtype: {embedding.dtype}")
        else:
            logger.error("❌ Model.s3gen doesn't have load_voice_clone method")
            logger.error("❌ This suggests the forked repository features are not available")
            return {"status": "error", "message": "Voice embedding support not available"}
        
        # Generate speech using the embedding
        logger.info("🎵 ===== TTS GENERATION =====")
        logger.info(f"🎵 Input text: {text}")
        logger.info(f"🎵 Using voice: {voice_id}")
        logger.info(f"🎵 Embedding path: {embedding_path}")
        
        start_time = time.time()
        
        try:
            logger.info("🔄 Starting TTS generation...")
            logger.info(f"🔄 Generation parameters:")
            logger.info(f"  - text length: {len(text)}")
            logger.info(f"  - saved_voice_path: {embedding_path}")
            logger.info(f"  - temperature: 0.7")
            logger.info(f"  - exaggeration: 0.6")
            
            # Use the generate method with saved_voice_path
            audio_tensor = model.generate(
                text,
                saved_voice_path=str(embedding_path),
                temperature=0.7,
                exaggeration=0.6
            )
            
            generation_time = time.time() - start_time
            logger.info(f"✅ TTS generated successfully in {generation_time:.2f}s")
            logger.info(f"✅ Audio tensor type: {type(audio_tensor)}")
            logger.info(f"✅ Audio tensor shape: {audio_tensor.shape}")
            logger.info(f"✅ Audio tensor dtype: {audio_tensor.dtype}")
            logger.info(f"✅ Sample rate: {model.sr}")
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Failed to generate TTS after {generation_time:.2f}s")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"❌ Error details: {e}")
            return {"status": "error", "message": f"Failed to generate TTS: {e}"}
        
        # Save the generated TTS to file
        logger.info("💾 ===== SAVING TTS FILE =====")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = TTS_GENERATED_DIR / f"tts_{voice_id}_{timestamp}.wav"
        
        logger.info(f"💾 Target filename: {tts_filename}")
        logger.info(f"💾 TTS directory exists: {TTS_GENERATED_DIR.exists()}")
        
        try:
            logger.info("🔄 Saving audio tensor to file...")
            torchaudio.save(str(tts_filename), audio_tensor, model.sr)
            
            if tts_filename.exists():
                file_size = tts_filename.stat().st_size
                logger.info(f"✅ TTS saved successfully: {tts_filename}")
                logger.info(f"✅ File size: {file_size} bytes")
                logger.info(f"✅ File size (KB): {file_size / 1024:.1f} KB")
            else:
                logger.error(f"❌ File was not created: {tts_filename}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save TTS file: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error details: {e}")
            # Continue anyway, don't fail the request
        
        # Convert to base64
        logger.info("📤 ===== CONVERTING TO BASE64 =====")
        try:
            audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
            logger.info(f"✅ Audio converted to base64 successfully")
            logger.info(f"✅ Base64 length: {len(audio_base64)} characters")
            logger.info(f"✅ Base64 size (KB): {len(audio_base64) * 3 / 4 / 1024:.1f} KB")
        except Exception as e:
            logger.error(f"❌ Failed to convert audio to base64: {e}")
            return {"status": "error", "message": f"Failed to convert audio to base64: {e}"}
        
        # Create response
        logger.info("📤 ===== CREATING RESPONSE =====")
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
        
        logger.info(f"📤 Response created successfully")
        logger.info(f"📤 Response keys: {list(response.keys())}")
        logger.info(f"📤 Has audio_base64: {bool(response.get('audio_base64'))}")
        logger.info(f"📤 Has metadata: {bool(response.get('metadata'))}")
        logger.info(f"📤 Audio base64 length: {len(audio_base64)}")
        logger.info(f"📤 Generation time: {generation_time:.2f}s")
        logger.info(f"📤 TTS file: {tts_filename}")
        
        # Clean up temporary embedding file
        try:
            if temp_embedding_path.exists():
                os.unlink(temp_embedding_path)
                logger.info(f"🗑️ Cleaned up temporary embedding file: {temp_embedding_path}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temporary embedding file: {cleanup_error}")
        
        # List final directory contents for debugging
        logger.info("📂 ===== FINAL DIRECTORY CONTENTS =====")
        list_files_for_debug()
        
        logger.info("🎉 ===== TTS HANDLER COMPLETED SUCCESSFULLY =====")
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
    logger.info("🔄 Converting audio tensor to base64...")
    logger.info(f"🔄 Audio tensor shape: {audio_tensor.shape}")
    logger.info(f"🔄 Audio tensor dtype: {audio_tensor.dtype}")
    logger.info(f"🔄 Sample rate: {sample_rate}")
    
    try:
        # Save to temporary file
        logger.info("🔄 Creating temporary file...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            logger.info(f"🔄 Temporary file: {tmp_filename}")
            
            logger.info("🔄 Saving audio tensor to temporary file...")
            torchaudio.save(tmp_filename, audio_tensor, sample_rate)
            
            # Check if file was created
            if os.path.exists(tmp_filename):
                file_size = os.path.getsize(tmp_filename)
                logger.info(f"✅ Temporary file created: {file_size} bytes")
            else:
                logger.error(f"❌ Temporary file was not created: {tmp_filename}")
                raise Exception("Failed to create temporary audio file")
            
            # Read back as binary data
            logger.info("🔄 Reading audio data from temporary file...")
            with open(tmp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            logger.info(f"✅ Audio data read: {len(audio_data)} bytes")
            
            # Clean up temporary file
            logger.info("🔄 Cleaning up temporary file...")
            os.unlink(tmp_filename)
            logger.info("✅ Temporary file cleaned up")
            
            # Encode as base64
            logger.info("🔄 Encoding audio data to base64...")
            base64_data = base64.b64encode(audio_data).decode('utf-8')
            logger.info(f"✅ Base64 encoding completed: {len(base64_data)} characters")
            
            return base64_data
            
    except Exception as e:
        logger.error(f"❌ Error converting audio to base64: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    logger.info("🚀 ===== TTS HANDLER STARTING =====")
    logger.info("🚀 Starting TTS generation handler...")
    
    try:
        logger.info("🔧 Initializing model...")
        initialize_model()
        logger.info("✅ Model initialization completed")
        
        logger.info("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({'handler': handler })
        
    except Exception as e:
        logger.error("💥 ===== TTS HANDLER STARTUP FAILED =====")
        logger.error(f"❌ Failed to start TTS handler: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise 
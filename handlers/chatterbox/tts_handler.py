import runpod
import time  
import os
import tempfile
import base64
import logging
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Minimal, production-focused TTS handler for RunPod runtime."""

# Import the models from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    from chatterbox.tts import ChatterboxTTS
    FORKED_HANDLER_AVAILABLE = True
    logger.info("✅ Successfully imported ChatterboxVC and ChatterboxTTS from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Could not import models from forked repository: {e}")

# Initialize models once at startup
vc_model = None
tts_model = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")  # For voice clone samples
TTS_GENERATED_DIR = Path("/tts_generated")  # For TTS story generation
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Check if directories exist
logger.info(f"Directory existence check:")
logger.info(f"  VOICE_PROFILES_DIR exists: {VOICE_PROFILES_DIR.exists()}")
logger.info(f"  VOICE_SAMPLES_DIR exists: {VOICE_SAMPLES_DIR.exists()}")
logger.info(f"  TTS_GENERATED_DIR exists: {TTS_GENERATED_DIR.exists()}")
logger.info(f"  TEMP_VOICE_DIR exists: {TEMP_VOICE_DIR.exists()}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# No repository mutation or cache clearing at runtime – the runtime uses a fixed image

# Initialize models AFTER repository update
logger.info("🔧 Initializing models...")
try:
    if FORKED_HANDLER_AVAILABLE:
        # Initialize TTS model first (needed for s3gen)
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS ready")
        
        # Initialize VC model using the correct method
        vc_model = ChatterboxVC.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxVC ready")
        
    else:
        logger.error("❌ Forked repository models not available")
        vc_model = None
        tts_model = None
        
except Exception as e:
    logger.error(f"❌ Failed to initialize models: {e}")
    vc_model = None
    tts_model = None

# -------------------------------------------------------------------
# 🐞  Firebase / GCS credential debug helper
# -------------------------------------------------------------------
def _debug_gcs_creds():
    """Minimal Firebase credential check (kept for quick diagnostics)."""
    try:
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        logger.info("Firebase secret present: %s", bool(firebase_secret_path))
    except Exception:
        pass

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    try:
        firebase_secret = os.getenv('RUNPOD_SECRET_Firebase')
        firebase_secret_path = firebase_secret
        if firebase_secret and firebase_secret.startswith('{'):
            import json
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(json.loads(firebase_secret), tmp_file)
                firebase_secret_path = tmp_file.name
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = firebase_secret_path
        if firebase_secret_path and os.path.exists(firebase_secret_path):
            client = storage.Client.from_service_account_json(firebase_secret_path)
        else:
            client = storage.Client()
        storage_client = client
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase storage client ready")
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

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TTS_GENERATED_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def call_tts_model_generate_tts_story(text, voice_id, profile_base64, language, story_type, is_kids_voice, api_metadata):
    """
    Implement TTS generation using available model methods.
    
    Uses the TTS model's generate method for text-to-speech generation.
    """
    global tts_model
    
    logger.info("TTS generate: voice_id=%s, lang=%s, type=%s, kids=%s, text_len=%d", voice_id, language, story_type, is_kids_voice, len(text))
    
    start_time = time.time()
    
    try:
        # Check if TTS model is available
        if tts_model is None:
            logger.error("❌ TTS model not available")
            return {
                "status": "error",
                "message": "TTS model not available",
                "generation_time": time.time() - start_time
            }
        
        if not hasattr(tts_model, 'generate_tts_story'):
            return {
                "status": "error",
                "message": "TTS model missing generate_tts_story",
                "generation_time": time.time() - start_time
            }
        result = tts_model.generate_tts_story(
            text=text,
            voice_id=voice_id,
            profile_base64=profile_base64,
            language=language,
            story_type=story_type,
            is_kids_voice=is_kids_voice,
            metadata=api_metadata
        )
        generation_time = time.time() - start_time
        logger.info("✅ TTS generation completed in %.2fs", generation_time)
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ TTS generation failed after {generation_time:.2f}s: {e}")
        return {
            "status": "error",
            "message": str(e),
            "generation_time": generation_time
        }

def handler(event, responseFormat="base64"):
    """Pure API orchestration: Handle TTS generation requests"""
    global tts_model
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "message": "Failed to initialize Firebase storage"}
    
    # Check if TTS model is available
    if tts_model is None:
        logger.error("❌ TTS model not available")
        return {"status": "error", "message": "TTS model not available"}
    
    logger.info("✅ Using pre-initialized TTS model")
    
    # Handle TTS generation request according to API contract
    text = event["input"].get("text")
    voice_id = event["input"].get("voice_id")
    profile_base64 = event["input"].get("profile_base64")
    language = event["input"].get("language", "en")
    story_type = event["input"].get("story_type", "user")
    is_kids_voice = event["input"].get("is_kids_voice", False)
    api_metadata = event["input"].get("metadata", {})
    callback_url = api_metadata.get("callback_url") or event["metadata"].get("callback_url") if isinstance(event.get("metadata"), dict) else None
    
    if not text or not voice_id:
        return {"status": "error", "message": "Both text and voice_id are required"}

    logger.info(f"🎵 TTS request. Voice ID: {voice_id}")
    logger.info(f"📝 Text length: {len(text)} characters")
    logger.info(f"🌍 Language: {language}, Story type: {story_type}")
    logger.info(f"👶 Kids voice: {is_kids_voice}")
    
    try:
        # Call the TTS model's generate_tts_story method - it handles everything!
        logger.info("🔄 Calling TTS model's generate_tts_story method...")
        
        result = call_tts_model_generate_tts_story(
            text=text,
            voice_id=voice_id,
            profile_base64=profile_base64,
            language=language,
            story_type=story_type,
            is_kids_voice=is_kids_voice,
            api_metadata=api_metadata
        )
        
        # Return the result from the TTS model
        logger.info(f"📤 TTS generation completed successfully")
        # If callback_url provided, post completion payload
        try:
            if callback_url and isinstance(result, dict) and result.get("status") == "success":
                import requests
                payload = {
                    "story_id": api_metadata.get("story_id") or event["input"].get("story_id"),
                    "user_id": api_metadata.get("user_id") or event["input"].get("user_id"),
                    "voice_id": voice_id,
                    "audio_url": result.get("firebase_url") or result.get("audio_url") or result.get("audio_path"),
                    "storage_path": result.get("firebase_path"),
                    "language": language,
                    "metadata": {
                        **({} if not isinstance(api_metadata, dict) else api_metadata),
                        "generation_time": result.get("generation_time"),
                    },
                }
                try:
                    resp = requests.post(callback_url, json=payload, timeout=10)
                    logger.info(f"🔔 Callback POST {callback_url} -> {resp.status_code}")
                except Exception as cb_e:
                    logger.warning(f"⚠️ Callback POST failed: {cb_e}")
        except Exception as e:
            logger.warning(f"⚠️ Error preparing callback: {e}")
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"status": "error", "message": str(e)}

def handle_file_download(input):
    """Handle file download requests"""
    file_path = input.get("file_path")
    if not file_path:
        return {"status": "error", "message": "file_path is required"}
    
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return {
            "status": "success",
            "file_data": base64.b64encode(file_data).decode('utf-8'),
            "file_size": len(file_data)
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file: {e}"}

def list_available_files():
    """List all available files in the directories"""
    files = {}
    
    for directory_name, directory_path in [
        ("voice_profiles", VOICE_PROFILES_DIR),
        ("voice_samples", VOICE_SAMPLES_DIR),
        ("tts_generated", TTS_GENERATED_DIR),
        ("temp_voice", TEMP_VOICE_DIR)
    ]:
        if directory_path.exists():
            files[directory_name] = [f.name for f in directory_path.glob("*")]
        else:
            files[directory_name] = []
    
    return {"status": "success", "files": files}

if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    logger.info("✅ TTS Handler ready")
    runpod.serverless.start({'handler': handler })
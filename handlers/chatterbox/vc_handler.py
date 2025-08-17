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

"""Minimal, production-focused VC handler for RunPod runtime."""

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

def rename_in_firebase(src_path: str, dest_path: str, *, metadata: Optional[dict] = None, content_type: Optional[str] = None) -> Optional[str]:
    """
    Copy a blob to a new destination (rename), set metadata, make public, then delete the old blob.
    Returns new public URL or None.
    """
    global bucket
    try:
        if bucket is None and not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot rename")
            return None
        src_blob = bucket.blob(src_path)
        if not src_blob.exists():
            logger.warning(f"⚠️ Source blob does not exist: {src_path}")
            return None
        # Perform copy
        new_blob = bucket.copy_blob(src_blob, bucket, dest_path)
        # Set metadata if provided
        if metadata:
            new_blob.metadata = metadata
        # Set content type if provided
        if content_type:
            new_blob.content_type = content_type
        new_blob.make_public()
        # Delete original
        try:
            src_blob.delete()
        except Exception as del_e:
            logger.warning(f"⚠️ Could not delete original blob {src_path}: {del_e}")
        logger.info(f"✅ Renamed {src_path} → {dest_path}")
        return new_blob.public_url
    except Exception as e:
        logger.error(f"❌ Rename failed {src_path} → {dest_path}: {e}")
        return None

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

def call_vc_model_create_voice_clone(audio_file_path, voice_id, voice_name, language="en", is_kids_voice=False, api_metadata=None):
    """
    Implement voice cloning using available model methods.
    
    Uses the TTS model's save_voice_clone method to create voice profiles.
    """
    global vc_model, tts_model
    
    logger.info("VC clone: voice_id=%s, name=%s, lang=%s, kids=%s", voice_id, voice_name, language, is_kids_voice)
    
    start_time = time.time()
    
    try:
        # Check if models are available
        if vc_model is None or tts_model is None:
            logger.error("❌ Models not available")
            return {
                "status": "error",
                "message": "Models not available",
                "generation_time": time.time() - start_time
            }
        
        if not hasattr(vc_model, 'create_voice_clone'):
            return {
                "status": "error",
                "message": "VC model missing create_voice_clone",
                "generation_time": time.time() - start_time
            }
        result = vc_model.create_voice_clone(
            audio_file_path=str(audio_file_path),
            voice_id=voice_id,
            voice_name=voice_name,
            metadata=api_metadata
        )
        generation_time = time.time() - start_time
        logger.info("✅ Voice clone completed in %.2fs", generation_time)
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Voice clone failed after {generation_time:.2f}s: {e}")
        return {
            "status": "error",
            "message": str(e),
            "generation_time": generation_time
        }

def handler(event, responseFormat="base64"):
    input = event['input']    
    
    # This handler is for voice cloning only
    return handle_voice_clone_request(input, responseFormat)

def handle_voice_clone_request(input, responseFormat):
    """Pure API orchestration: Handle voice cloning requests"""
    global vc_model
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "message": "Failed to initialize Firebase storage"}
    
    # Check if VC model is available
    if vc_model is None:
        logger.error("❌ VC model not available")
        return {"status": "error", "message": "VC model not available"}
    
    logger.info("✅ Using pre-initialized VC model")
    
    # Handle voice generation request only
    name = input.get('name')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_path = input.get('audio_path')  # Firebase Storage path e.g. audio/voices/en/recorded/uid_ts.wav
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio
    responseFormat = input.get('responseFormat', 'base64')  # Response format from frontend
    language = input.get('language', 'en')  # Language for storage organization
    is_kids_voice = input.get('is_kids_voice', False)  # Kids voice flag
    # Naming hints (optional)
    meta_top = input.get('metadata', {}) if isinstance(input.get('metadata'), dict) else {}
    profile_filename_hint = input.get('profile_filename') or meta_top.get('profile_filename')
    sample_filename_hint = input.get('sample_filename') or meta_top.get('sample_filename')
    output_basename_hint = input.get('output_basename') or meta_top.get('output_basename')
    user_id = input.get('user_id') or meta_top.get('user_id')

    if not name or (not audio_data and not audio_path):
        return {"status": "error", "message": "name and either audio_data or audio_path are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    logger.info(f"Language: {language}, Kids voice: {is_kids_voice}")
    
    try:
        # Generate a unique voice ID based on the name
        voice_id = get_voice_id(name)
        logger.info(f"Generated voice ID: {voice_id}")
        
        # Prepare a local temp file from either base64 data or Firebase path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_voice_file = None
        if audio_path:
            # Infer extension from path
            lower = str(audio_path).lower()
            ext = ".wav"
            if lower.endswith(".mp3"): ext = ".mp3"
            elif lower.endswith(".ogg"): ext = ".ogg"
            elif lower.endswith(".m4a"): ext = ".m4a"
            temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}{ext}"
            try:
                if bucket is None and not initialize_firebase():
                    return {"status": "error", "message": "Failed to initialize Firebase storage"}
                blob = bucket.blob(audio_path)
                blob.download_to_filename(str(temp_voice_file))
                logger.info(f"Downloaded audio from Firebase {audio_path} to {temp_voice_file}")
            except Exception as dl_e:
                logger.error(f"❌ Failed to download audio_path {audio_path}: {dl_e}")
                return {"status": "error", "message": f"Failed to download audio_path: {dl_e}"}
        else:
            temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
            audio_bytes = base64.b64decode(audio_data)
            with open(temp_voice_file, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Saved temporary voice file to {temp_voice_file}")

        # Call the VC model's create_voice_clone method
        logger.info("🔄 Calling VC model's create_voice_clone method...")
        
        # Prepare API metadata including language and kids voice flag
        api_metadata = {
            'user_id': input.get('user_id'),
            'project_id': input.get('project_id'),
            'voice_type': input.get('voice_type'),
            'quality': input.get('quality'),
            'language': language,
            'is_kids_voice': is_kids_voice,
            # If request was pointer-based, pass the recorded_path to VC so it skips re-upload
            'recorded_path': audio_path if audio_path else None,
        }
        
        # Call the VC model - it handles everything!
        result = call_vc_model_create_voice_clone(
            audio_file_path=temp_voice_file,
            voice_id=voice_id,
            voice_name=name,
            api_metadata=api_metadata
        )
        
        # Clean up temporary voice file
        try:
            os.unlink(temp_voice_file)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")

        # Post-process result: if caller supplied audio_path, avoid duplicate recorded upload
        # and ensure recorded_audio_path reflects the original pointer
        try:
            if isinstance(result, dict) and audio_path:
                result.setdefault('metadata', {})
                result['recorded_audio_path'] = audio_path
        except Exception:
            pass
        logger.info(f"📤 Voice clone completed successfully")

        # No post-process renaming: model now uploads with standardized names directly

        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if 'temp_voice_file' in locals():
            try:
                os.unlink(temp_voice_file)
            except:
                pass
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

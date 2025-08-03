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

# Initialize models once at startup
logger.info("🔧 Initializing models from forked repository...")
try:
    if FORKED_HANDLER_AVAILABLE:
        # Initialize VC model
        vc_model = ChatterboxVC(device='cuda')
        logger.info("✅ ChatterboxVC model initialized successfully")
        
        # Initialize TTS model  
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS model initialized successfully")
        
        # Validate models have expected methods
        logger.info("🔍 Validating model methods...")
        
        # Check VC model methods
        vc_expected_methods = ['create_voice_clone']
        vc_available_methods = [method for method in dir(vc_model) if not method.startswith('_')]
        vc_missing_methods = [method for method in vc_expected_methods if not hasattr(vc_model, method)]
        
        logger.info(f"🔍 VC Available methods: {vc_available_methods}")
        logger.info(f"🔍 VC Expected methods: {vc_expected_methods}")
        if vc_missing_methods:
            logger.warning(f"⚠️ Missing VC methods: {vc_missing_methods}")
        else:
            logger.info("✅ All VC methods are available")
        
        # Check TTS model methods
        tts_expected_methods = ['generate_tts_story', 'generate_long_text']
        tts_available_methods = [method for method in dir(tts_model) if not method.startswith('_')]
        tts_missing_methods = [method for method in tts_expected_methods if not hasattr(tts_model, method)]
        
        logger.info(f"🔍 TTS Available methods: {tts_available_methods}")
        logger.info(f"🔍 TTS Expected methods: {tts_expected_methods}")
        if tts_missing_methods:
            logger.warning(f"⚠️ Missing TTS methods: {tts_missing_methods}")
        else:
            logger.info("✅ All TTS methods are available")
        
        # Log model details for debugging
        import inspect
        logger.info(f"📦 VC model type: {type(vc_model).__name__}")
        logger.info(f"📦 TTS model type: {type(tts_model).__name__}")
        
        try:
            vc_file = inspect.getfile(vc_model.__class__)
            tts_file = inspect.getfile(tts_model.__class__)
            logger.info(f"📦 VC model file: {vc_file}")
            logger.info(f"📦 TTS model file: {tts_file}")
            if "chatterbox_embed" in vc_file and "chatterbox_embed" in tts_file:
                logger.info("✅ Models are from the correct repository")
            else:
                logger.warning("⚠️ Models are NOT from the expected repository")
        except Exception as e:
            logger.warning(f"⚠️ Could not determine model files: {e}")
        
        # Debug: Check Git commit of forked repository
        logger.info("🔍 ===== FORKED REPOSITORY GIT DEBUG =====")
        try:
            import subprocess
            import os
            
            # Find the chatterbox_embed directory
            chatterbox_embed_path = None
            for root, dirs, files in os.walk("/workspace"):
                if "chatterbox_embed" in dirs:
                    chatterbox_embed_path = os.path.join(root, "chatterbox_embed")
                    break
            
            if chatterbox_embed_path and os.path.exists(chatterbox_embed_path):
                logger.info(f"📂 Found chatterbox_embed at: {chatterbox_embed_path}")
                
                # Check if it's a git repository
                git_dir = os.path.join(chatterbox_embed_path, ".git")
                if os.path.exists(git_dir):
                    logger.info("✅ Found .git directory - it's a git repository")
                    
                    # Get current commit hash
                    try:
                        result = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=chatterbox_embed_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            commit_hash = result.stdout.strip()
                            logger.info(f"🔍 Current commit hash: {commit_hash}")
                        else:
                            logger.warning(f"⚠️ Could not get commit hash: {result.stderr}")
                            commit_hash = "unknown"
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting commit hash: {e}")
                        commit_hash = "error"
                    
                    # Get commit message
                    try:
                        result = subprocess.run(
                            ["git", "log", "-1", "--pretty=format:%s"],
                            cwd=chatterbox_embed_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            commit_message = result.stdout.strip()
                            logger.info(f"📝 Last commit message: {commit_message}")
                        else:
                            logger.warning(f"⚠️ Could not get commit message: {result.stderr}")
                            commit_message = "unknown"
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting commit message: {e}")
                        commit_message = "error"
                    
                    # Get commit date
                    try:
                        result = subprocess.run(
                            ["git", "log", "-1", "--pretty=format:%ci"],
                            cwd=chatterbox_embed_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            commit_date = result.stdout.strip()
                            logger.info(f"📅 Last commit date: {commit_date}")
                        else:
                            logger.warning(f"⚠️ Could not get commit date: {result.stderr}")
                            commit_date = "unknown"
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting commit date: {e}")
                        commit_date = "error"
                    
                    # Get remote URL
                    try:
                        result = subprocess.run(
                            ["git", "remote", "get-url", "origin"],
                            cwd=chatterbox_embed_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            remote_url = result.stdout.strip()
                            logger.info(f"🌐 Remote URL: {remote_url}")
                        else:
                            logger.warning(f"⚠️ Could not get remote URL: {result.stderr}")
                            remote_url = "unknown"
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting remote URL: {e}")
                        remote_url = "error"
                    
                    # Check if there are uncommitted changes
                    try:
                        result = subprocess.run(
                            ["git", "status", "--porcelain"],
                            cwd=chatterbox_embed_path,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            if result.stdout.strip():
                                logger.warning("⚠️ Repository has uncommitted changes!")
                                logger.warning(f"📋 Changes: {result.stdout.strip()}")
                            else:
                                logger.info("✅ Repository is clean (no uncommitted changes)")
                        else:
                            logger.warning(f"⚠️ Could not check git status: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error checking git status: {e}")
                    
                    # Summary
                    logger.info("📊 ===== FORKED REPO SUMMARY =====")
                    logger.info(f"🔍 Commit Hash: {commit_hash}")
                    logger.info(f"📝 Commit Message: {commit_message}")
                    logger.info(f"📅 Commit Date: {commit_date}")
                    logger.info(f"🌐 Remote URL: {remote_url}")
                    
                else:
                    logger.warning("⚠️ No .git directory found - not a git repository")
            else:
                logger.warning("⚠️ Could not find chatterbox_embed directory")
                
        except Exception as e:
            logger.error(f"❌ Error during git debugging: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        logger.info("🔍 ===== END GIT DEBUG =====")
        
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
    """Minimal Firebase credential check"""
    import os
    logger.info("🔍 Firebase credentials check")
    
    # Check if RunPod secret is available
    firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
    if firebase_secret_path:
        if firebase_secret_path.startswith('{'):
            logger.info("✅ Using RunPod Firebase secret (JSON content)")
        else:
            logger.info("✅ Using RunPod Firebase secret (file path)")
    else:
        logger.warning("⚠️ No RunPod Firebase secret found")

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    try:
        # Debug: Check environment variables
        logger.info("🔍 Checking Firebase environment variables...")
        firebase_secret = os.getenv('RUNPOD_SECRET_Firebase')
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"🔍 RUNPOD_SECRET_Firebase exists: {firebase_secret is not None}")
        logger.info(f"🔍 GOOGLE_APPLICATION_CREDENTIALS exists: {google_creds is not None}")
        
        # Check if we're in RunPod and have the secret
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        
        if firebase_secret_path:
            if firebase_secret_path.startswith('{'):
                # It's JSON content, create a temporary file
                logger.info("✅ Using RunPod Firebase secret as JSON content")
                import tempfile
                import json
                
                # Validate JSON first
                try:
                    creds_data = json.loads(firebase_secret_path)
                    logger.info(f"✅ Valid JSON with project_id: {creds_data.get('project_id', 'unknown')}")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in RUNPOD_SECRET_Firebase: {e}")
                    raise
                
                # Create temporary file with the JSON content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(creds_data, tmp_file)
                    tmp_path = tmp_file.name
                
                logger.info(f"✅ Created temporary credentials file: {tmp_path}")
                storage_client = storage.Client.from_service_account_json(tmp_path)
                
            elif os.path.exists(firebase_secret_path):
                # It's a file path
                logger.info(f"✅ Using RunPod Firebase secret file: {firebase_secret_path}")
                storage_client = storage.Client.from_service_account_json(firebase_secret_path)
            else:
                logger.warning(f"⚠️ RUNPOD_SECRET_Firebase exists but is not JSON content or valid file path")
                # Fallback to GOOGLE_APPLICATION_CREDENTIALS
                logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
                storage_client = storage.Client()
        else:
            # No RunPod secret, fallback to GOOGLE_APPLICATION_CREDENTIALS
            logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
            storage_client = storage.Client()
        
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase storage client initialized successfully")
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
    Pure API orchestration: Call the TTS model's generate_tts_story method.
    
    The TTS model handles all the model logic:
    - Voice profile loading
    - Text-to-speech generation
    - Audio processing
    - Firebase upload
    - Error handling
    - Logging
    
    This API app only:
    - Calls the model method
    - Handles the returned data
    - Returns API responses
    """
    global tts_model
    
    logger.info(f"🎯 ===== CALLING TTS MODEL =====")
    logger.info(f"🔍 Parameters:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  language: {language}")
    logger.info(f"  story_type: {story_type}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  text_length: {len(text)} characters")
    
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
        
        # Check if generate_tts_story method exists
        if not hasattr(tts_model, 'generate_tts_story'):
            logger.error("❌ TTS model doesn't have generate_tts_story method")
            logger.error("🔍 This means the RunPod deployment is using an older version of the forked repository")
            
            # Debug: List all available methods
            available_methods = [method for method in dir(tts_model) if not method.startswith('_')]
            logger.info(f"🔍 Available methods in tts_model: {available_methods}")
            
            return {
                "status": "error",
                "message": "TTS model doesn't have generate_tts_story method. Please update the RunPod deployment with the latest forked repository version.",
                "generation_time": time.time() - start_time,
                "debug_info": {
                    "available_methods": available_methods,
                    "tts_model_type": type(tts_model).__name__,
                    "tts_model_module": tts_model.__class__.__module__
                }
            }
        
        # Call the TTS model's generate_tts_story method
        logger.info("🔄 Calling TTS model's generate_tts_story method...")
        
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
        logger.info(f"✅ TTS model generate_tts_story completed in {generation_time:.2f}s")
        
        return result
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ TTS model call failed after {generation_time:.2f}s: {e}")
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
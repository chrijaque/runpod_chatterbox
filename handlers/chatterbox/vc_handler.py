import runpod
import time  
import os
import tempfile
import base64
import logging
import sys
import glob
import pathlib
import shutil
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_python_cache():
    """Clear all Python cache files to force fresh loading."""
    logger.info("🧹 Clearing Python cache...")
    
    # Remove .pyc files
    pyc_files = glob.glob("/workspace/**/*.pyc", recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
        except Exception as e:
            logger.warning(f"  - Failed to remove {pyc_file}: {e}")
    
    # Remove __pycache__ directories
    pycache_dirs = list(pathlib.Path("/workspace").rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
        except Exception as e:
            logger.warning(f"  - Failed to remove {pycache_dir}: {e}")
    
    # Clear sys.modules for chatterbox
    modules_to_clear = [name for name in sys.modules.keys() if 'chatterbox' in name]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

# Clear cache BEFORE importing any chatterbox modules
clear_python_cache()

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

# Update repository to latest commit BEFORE initializing models
logger.info("🔧 Updating repository to latest commit...")
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
            logger.info("✅ Found .git directory - updating to latest commit...")
            
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
                    old_commit_hash = result.stdout.strip()
                    logger.info(f"🔍 Current commit: {old_commit_hash}")
                else:
                    logger.warning(f"⚠️ Could not get current commit: {result.stderr}")
                    old_commit_hash = "unknown"
            except Exception as e:
                logger.warning(f"⚠️ Error getting current commit: {e}")
                old_commit_hash = "error"
            
            # Fetch latest changes
            try:
                logger.info("🔄 Fetching latest changes...")
                result = subprocess.run(
                    ["git", "fetch", "origin"],
                    cwd=chatterbox_embed_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("✅ Successfully fetched latest changes")
                    
                    # Get the default branch
                    result = subprocess.run(
                        ["git", "remote", "show", "origin"],
                        cwd=chatterbox_embed_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'HEAD branch' in line:
                                default_branch = line.split()[-1]
                                logger.info(f"🔍 Default branch: {default_branch}")
                                
                                # Check what's in the remote branch
                                result = subprocess.run(
                                    ["git", "rev-parse", f"origin/{default_branch}"],
                                    cwd=chatterbox_embed_path,
                                    capture_output=True,
                                    text=True,
                                    timeout=10
                                )
                                if result.returncode == 0:
                                    remote_commit = result.stdout.strip()
                                    logger.info(f"🔍 Remote {default_branch} commit: {remote_commit}")
                                    
                                    # Reset to latest commit
                                    logger.info(f"🔍 Resetting to origin/{default_branch}...")
                                    result = subprocess.run(
                                        ["git", "reset", "--hard", f"origin/{default_branch}"],
                                        cwd=chatterbox_embed_path,
                                        capture_output=True,
                                        text=True,
                                        timeout=30
                                    )
                                    if result.returncode == 0:
                                        logger.info(f"✅ Successfully reset to latest {default_branch}")
                                        
                                        # Get new commit hash
                                        result = subprocess.run(
                                            ["git", "rev-parse", "HEAD"],
                                            cwd=chatterbox_embed_path,
                                            capture_output=True,
                                            text=True,
                                            timeout=10
                                        )
                                        if result.returncode == 0:
                                            new_commit_hash = result.stdout.strip()
                                            logger.info(f"🆕 New commit hash: {new_commit_hash}")
                                            
                                            if new_commit_hash != old_commit_hash:
                                                logger.info("🔄 Repository updated to latest commit!")
                                                
                                                # Reload the chatterbox modules to use the updated code
                                                logger.info("🔄 Reloading chatterbox modules...")
                                                modules_to_reload = [name for name in sys.modules.keys() if 'chatterbox' in name]
                                                for module_name in modules_to_reload:
                                                    del sys.modules[module_name]
                                                    logger.info(f"🔄 Reloaded: {module_name}")
                                                
                                                # Re-import the models
                                                try:
                                                    from chatterbox.vc import ChatterboxVC
                                                    from chatterbox.tts import ChatterboxTTS
                                                    logger.info("✅ Successfully re-imported models after update")
                                                except ImportError as e:
                                                    logger.warning(f"⚠️ Failed to re-import models: {e}")
                                            else:
                                                logger.info("✅ Already at latest commit")
                                        else:
                                            logger.warning(f"⚠️ Failed to get new commit hash: {result.stderr}")
                                    else:
                                        logger.warning(f"⚠️ Failed to reset to latest: {result.stderr}")
                                else:
                                    logger.warning(f"⚠️ Could not get remote commit: {result.stderr}")
                                break
                        else:
                            logger.warning(f"⚠️ Could not determine default branch: {result.stderr}")
                    else:
                        logger.warning(f"⚠️ Failed to fetch latest changes: {result.stderr}")
            except Exception as e:
                logger.warning(f"⚠️ Error updating repository: {e}")
        else:
            logger.warning("⚠️ No .git directory found - not a git repository")
    else:
        logger.warning("⚠️ Could not find chatterbox_embed directory")
except Exception as e:
    logger.error(f"❌ Error during repository update: {e}")

# Initialize models AFTER repository update
logger.info("🔧 Initializing models from forked repository...")
try:
    if FORKED_HANDLER_AVAILABLE:
        # Initialize TTS model first (needed for s3gen)
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS model initialized successfully")
        
        # Initialize VC model with s3gen from TTS model
        vc_model = ChatterboxVC(s3gen=tts_model.s3gen, device='cuda')
        
        # CRITICAL: Create a wrapper for the text encoder to handle T3.forward() signature
        class TextEncoderWrapper:
            def __init__(self, t3_model):
                self.t3_model = t3_model
                
            def __call__(self, text):
                # Try different ways to call the T3 model based on its signature
                try:
                    # Method 1: Try calling with text as positional argument
                    return self.t3_model(text)
                except TypeError as e:
                    if "takes 1 positional argument but 2 were given" in str(e):
                        # Method 2: Try calling with text as keyword argument
                        try:
                            return self.t3_model.forward(text=text)
                        except:
                            # Method 3: Try calling with text as first argument after self
                            try:
                                return self.t3_model.forward(text)
                            except:
                                # Method 4: Try calling with text as input parameter
                                try:
                                    return self.t3_model.forward(input=text)
                                except:
                                    # Method 5: Try calling with text as input_ids
                                    try:
                                        return self.t3_model.forward(input_ids=text)
                                    except Exception as final_e:
                                        logger.error(f"❌ All T3 forward methods failed: {final_e}")
                                        raise e
                    else:
                        raise e
        
        # Attach the wrapped text encoder to s3gen
        vc_model.s3gen.text_encoder = TextEncoderWrapper(tts_model.t3)
        logger.info("✅ ChatterboxVC model initialized successfully")
        logger.info("✅ Text encoder wrapper attached to s3gen")
        
        # Debug: Check T3 model's forward method signature
        try:
            import inspect
            t3_forward_sig = inspect.signature(tts_model.t3.forward)
            logger.info(f"🔍 T3 forward method signature: {t3_forward_sig}")
            logger.info(f"🔍 T3 forward method parameters: {list(t3_forward_sig.parameters.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get T3 forward signature: {e}")
        
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
        
        # Final Git status check
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd="/workspace/chatterbox_embed",
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                final_commit = result.stdout.strip()
                logger.info(f"🎯 Final commit hash: {final_commit}")
            else:
                logger.warning(f"⚠️ Could not get final commit: {result.stderr}")
        except Exception as e:
            logger.warning(f"⚠️ Error getting final commit: {e}")

        
        
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
        
        # Debug: Log Firebase credentials details
        if firebase_secret:
            logger.info(f"🔍 RUNPOD_SECRET_Firebase length: {len(firebase_secret)} characters")
            logger.info(f"🔍 RUNPOD_SECRET_Firebase preview: {firebase_secret[:200]}...")
            
            # Try to parse and validate the JSON
            try:
                import json
                cred_data = json.loads(firebase_secret)
                logger.info(f"🔍 Firebase Project ID: {cred_data.get('project_id', 'NOT FOUND')}")
                logger.info(f"🔍 Firebase Client Email: {cred_data.get('client_email', 'NOT FOUND')}")
                logger.info("✅ Firebase credentials JSON is valid")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Firebase credentials JSON is invalid: {e}")
                logger.error(f"❌ Credentials preview: {firebase_secret[:500]}...")
        else:
            logger.warning("⚠️ RUNPOD_SECRET_Firebase is not set!")
            logger.warning("⚠️ Firebase functionality will not work")
        
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
                
                # Set the environment variable for Google Cloud SDK
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tmp_path
                logger.info(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS to: {tmp_path}")
                
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
    
    logger.info(f"🎯 ===== CALLING VOICE CLONE =====")
    logger.info(f"🔍 Parameters:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  voice_name: {voice_name}")
    logger.info(f"  language: {language}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    
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
        
        # 🔍 DEBUG: Show all available methods for both models
        logger.info("🔍 ===== MODEL METHODS DEBUG =====")
        
        # VC Model methods
        vc_methods = [method for method in dir(vc_model) if not method.startswith('_')]
        logger.info(f"🔍 VC Model type: {type(vc_model).__name__}")
        logger.info(f"🔍 VC Model file: {vc_model.__class__.__module__}")
        logger.info(f"🔍 VC Available methods ({len(vc_methods)}): {vc_methods}")
        
        # TTS Model methods
        tts_methods = [method for method in dir(tts_model) if not method.startswith('_')]
        logger.info(f"🔍 TTS Model type: {type(tts_model).__name__}")
        logger.info(f"🔍 TTS Model file: {tts_model.__class__.__module__}")
        logger.info(f"🔍 TTS Available methods ({len(tts_methods)}): {tts_methods}")
        
        # Check for specific methods we need
        logger.info("🔍 ===== METHOD AVAILABILITY CHECK =====")
        logger.info(f"🔍 VC has 'generate': {hasattr(vc_model, 'generate')}")
        logger.info(f"🔍 VC has 'set_target_voice': {hasattr(vc_model, 'set_target_voice')}")
        logger.info(f"🔍 VC has 's3gen': {hasattr(vc_model, 's3gen')}")
        logger.info(f"🔍 TTS has 'save_voice_clone': {hasattr(tts_model, 'save_voice_clone')}")
        logger.info(f"🔍 TTS has 'load_voice_clone': {hasattr(tts_model, 'load_voice_clone')}")
        logger.info(f"🔍 TTS has 'generate': {hasattr(tts_model, 'generate')}")
        
        # Check method signatures if they exist
        if hasattr(tts_model, 'save_voice_clone'):
            import inspect
            try:
                sig = inspect.signature(tts_model.save_voice_clone)
                logger.info(f"🔍 TTS save_voice_clone signature: {sig}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get save_voice_clone signature: {e}")
        
        logger.info("🔍 ===== END MODEL METHODS DEBUG =====")
        
        # Try to use the VC model's create_voice_clone method
        if hasattr(vc_model, 'create_voice_clone'):
            logger.info("🔄 Using VC model's create_voice_clone method...")
            
            try:
                # Use the VC model's create_voice_clone method
                result = vc_model.create_voice_clone(
                    audio_file_path=str(audio_file_path),
                    voice_id=voice_id,
                    voice_name=voice_name,
                    language=language,
                    is_kids_voice=is_kids_voice,
                    api_metadata=api_metadata
                )
                
                generation_time = time.time() - start_time
                logger.info(f"✅ Voice clone completed in {generation_time:.2f}s")
                
                return result
                
            except Exception as method_error:
                logger.error(f"❌ create_voice_clone method failed: {method_error}")
                return {
                    "status": "error",
                    "message": f"create_voice_clone method failed: {method_error}",
                    "generation_time": time.time() - start_time,
                    "debug_info": {
                        "vc_methods": vc_methods,
                        "tts_methods": tts_methods,
                        "error": str(method_error)
                    }
                }
        else:
            logger.error("❌ VC model doesn't have create_voice_clone method")
            return {
                "status": "error",
                "message": "VC model doesn't have create_voice_clone method. Please update the RunPod deployment with the latest forked repository version.",
                "generation_time": time.time() - start_time,
                "debug_info": {
                    "vc_methods": vc_methods,
                    "tts_methods": tts_methods,
                    "available_vc_methods": [m for m in vc_methods if 'voice' in m.lower() or 'clone' in m.lower()]
                }
            }
        
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
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio
    responseFormat = input.get('responseFormat', 'base64')  # Response format from frontend
    language = input.get('language', 'en')  # Language for storage organization
    is_kids_voice = input.get('is_kids_voice', False)  # Kids voice flag

    if not name or not audio_data:
        return {"status": "error", "message": "Both name and audio_data are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    logger.info(f"Language: {language}, Kids voice: {is_kids_voice}")
    
    try:
        # Generate a unique voice ID based on the name
        voice_id = get_voice_id(name)
        logger.info(f"Generated voice ID: {voice_id}")
        
        # Save the uploaded audio to temp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
        audio_bytes = base64.b64decode(audio_data)
        with open(temp_voice_file, 'wb') as f:
            f.write(audio_bytes)
        logger.info(f"Saved temporary voice file to {temp_voice_file}")

        # Call the VC model's create_voice_clone method
        logger.info("🔄 Calling VC model's create_voice_clone method...")
        
        # Prepare API metadata
        api_metadata = {
            'user_id': input.get('user_id'),
            'project_id': input.get('project_id'),
            'voice_type': input.get('voice_type'),
            'quality': input.get('quality')
        }
        
        # Call the VC model - it handles everything!
        result = call_vc_model_create_voice_clone(
            audio_file_path=temp_voice_file,
            voice_id=voice_id,
            voice_name=name,
            language=language,
            is_kids_voice=is_kids_voice,
            api_metadata=api_metadata
        )
        
        # Clean up temporary voice file
        try:
            os.unlink(temp_voice_file)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")

        # Return the result from the VC-model
        logger.info(f"📤 Voice clone completed successfully")
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

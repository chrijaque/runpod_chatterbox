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
import hmac
import hashlib
import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Minimal, production-focused VC handler for RunPod runtime."""

def clear_python_cache():
    """Clear Python caches and loaded chatterbox modules to ensure fresh load."""
    try:
        # Remove .pyc files
        for pyc_file in glob.glob("/workspace/**/*.pyc", recursive=True):
            try:
                os.remove(pyc_file)
            except Exception:
                pass
        # Remove __pycache__ directories
        for pycache_dir in pathlib.Path("/workspace").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
            except Exception:
                pass
        # Clear chatterbox entries from sys.modules
        to_clear = [name for name in list(sys.modules.keys()) if 'chatterbox' in name]
        for name in to_clear:
            del sys.modules[name]
    except Exception:
        pass

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
    chatterbox_embed_path = None
    for root, dirs, files in os.walk("/workspace"):
        if "chatterbox_embed" in dirs:
            chatterbox_embed_path = os.path.join(root, "chatterbox_embed")
            break
    if chatterbox_embed_path and os.path.exists(chatterbox_embed_path):
        logger.info(f"📂 Found chatterbox_embed at: {chatterbox_embed_path}")
        git_dir = os.path.join(chatterbox_embed_path, ".git")
        if os.path.exists(git_dir):
            logger.info("✅ Found .git directory - updating to latest commit...")
            # Current commit
            try:
                old_commit = subprocess.run([
                    "git", "rev-parse", "HEAD"
                ], cwd=chatterbox_embed_path, capture_output=True, text=True, timeout=10)
                old_commit_hash = old_commit.stdout.strip() if old_commit.returncode == 0 else "unknown"
                logger.info(f"🔍 Current commit: {old_commit_hash}")
            except Exception:
                old_commit_hash = "unknown"
                logger.warning("⚠️ Could not get current commit")
            # Fetch + reset to default branch head
            try:
                logger.info("🔄 Fetching latest changes...")
                subprocess.run(["git", "fetch", "origin"], cwd=chatterbox_embed_path, capture_output=True, text=True, timeout=30)
                remote_show = subprocess.run(["git", "remote", "show", "origin"], cwd=chatterbox_embed_path, capture_output=True, text=True, timeout=10)
                default_branch = None
                if remote_show.returncode == 0:
                    for line in remote_show.stdout.split('\n'):
                        if 'HEAD branch' in line:
                            default_branch = line.split()[-1]
                            logger.info(f"🔍 Default branch: {default_branch}")
                            break
                if default_branch:
                    logger.info(f"🔄 Resetting to origin/{default_branch}...")
                    subprocess.run(["git", "reset", "--hard", f"origin/{default_branch}"], cwd=chatterbox_embed_path, capture_output=True, text=True, timeout=30)
                    new_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=chatterbox_embed_path, capture_output=True, text=True, timeout=10)
                    new_commit_hash = new_commit.stdout.strip() if new_commit.returncode == 0 else old_commit_hash
                    logger.info(f"🆕 New commit: {new_commit_hash}")
                    if new_commit_hash != old_commit_hash:
                        logger.info("🔄 Repository updated! Clearing modules to reload...")
                        # If updated, clear chatterbox modules to reload code
                        for name in [n for n in list(sys.modules.keys()) if 'chatterbox' in n]:
                            del sys.modules[name]
                        # Re-import models after update
                        try:
                            from chatterbox.vc import ChatterboxVC
                            from chatterbox.tts import ChatterboxTTS
                            logger.info("✅ Successfully re-imported models after update")
                        except ImportError as e:
                            logger.warning(f"⚠️ Failed to re-import models: {e}")
                    else:
                        logger.info("✅ Already at latest commit")
                else:
                    logger.warning("⚠️ Could not determine default branch")
            except Exception:
                logger.warning("⚠️ Error during git update")
        else:
            logger.warning("⚠️ No .git directory found")
    else:
        logger.warning("⚠️ Could not find chatterbox_embed directory")
except Exception:
    logger.error("❌ Error during repository update")

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
    """Comprehensive Firebase credential check and validation."""
    logger.info("🔍 ===== FIREBASE CREDENTIAL VALIDATION =====")
    try:
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        logger.info(f"🔑 Firebase secret present: {bool(firebase_secret_path)}")
        logger.info(f"🔑 Firebase secret length: {len(firebase_secret_path) if firebase_secret_path else 0}")
        
        if firebase_secret_path:
            # Check if it's JSON content
            if firebase_secret_path.startswith('{'):
                logger.info("🔑 Firebase secret appears to be JSON content")
                try:
                    import json
                    cred_data = json.loads(firebase_secret_path)
                    logger.info(f"🔑 JSON validation: SUCCESS")
                    logger.info(f"🔑 Project ID: {cred_data.get('project_id', 'NOT FOUND')}")
                    logger.info(f"🔑 Client Email: {cred_data.get('client_email', 'NOT FOUND')}")
                    logger.info(f"🔑 Private Key ID: {cred_data.get('private_key_id', 'NOT FOUND')}")
                    logger.info(f"🔑 Type: {cred_data.get('type', 'NOT FOUND')}")
                    
                    # Check for required fields
                    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                    missing_fields = [field for field in required_fields if field not in cred_data]
                    if missing_fields:
                        logger.error(f"❌ Missing required credential fields: {missing_fields}")
                    else:
                        logger.info("✅ All required credential fields present")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Firebase secret JSON is invalid: {e}")
                except Exception as e:
                    logger.error(f"❌ Error parsing Firebase secret: {e}")
            else:
                logger.info("🔑 Firebase secret appears to be a file path")
                if os.path.exists(firebase_secret_path):
                    logger.info(f"✅ Firebase secret file exists: {firebase_secret_path}")
                    try:
                        with open(firebase_secret_path, 'r') as f:
                            content = f.read()
                            logger.info(f"🔑 File content length: {len(content)}")
                            logger.info(f"🔑 File content preview: {content[:100]}...")
                    except Exception as e:
                        logger.error(f"❌ Error reading Firebase secret file: {e}")
                else:
                    logger.error(f"❌ Firebase secret file does not exist: {firebase_secret_path}")
        else:
            logger.error("❌ RUNPOD_SECRET_Firebase environment variable not set")
            
        # Check bucket identifier
        bucket_name = "godnathistorie-a25fa.firebasestorage.app"
        logger.info(f"🔑 Bucket identifier: {bucket_name}")
        logger.info(f"🔑 Bucket project ID: {bucket_name.replace('.firebasestorage.app', '')}")
        
    except Exception as e:
        logger.error(f"❌ Firebase credential validation failed: {e}")
    
    logger.info("🔍 ===== END FIREBASE CREDENTIAL VALIDATION =====")

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
        
        # CRITICAL: Patch metadata to ensure persistence
        if metadata:
            try:
                blob.patch()
                logger.info(f"✅ Metadata patched successfully for: {destination_blob_name}")
            except Exception as patch_e:
                logger.error(f"❌ Failed to patch metadata for {destination_blob_name}: {patch_e}")
        
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
            logger.info(f"🔍 Set metadata on renamed blob: {metadata}")
        # Set content type if provided
        if content_type:
            new_blob.content_type = content_type
        new_blob.make_public()
        
        # CRITICAL: Patch metadata to ensure persistence
        if metadata:
            try:
                new_blob.patch()
                logger.info(f"✅ Metadata patched successfully for renamed blob: {dest_path}")
            except Exception as patch_e:
                logger.error(f"❌ Failed to patch metadata for renamed blob {dest_path}: {patch_e}")
        
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
                "error": "Models not available",
                "generation_time": time.time() - start_time
            }
        
        if not hasattr(vc_model, 'create_voice_clone'):
            return {
                "status": "error",
                "error": "VC model missing create_voice_clone",
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
            "error": str(e),
            "generation_time": generation_time
        }

def handler(event, responseFormat="base64"):
    input = event['input']    
    
    # This handler is for voice cloning only
    return handle_voice_clone_request(input, responseFormat)

def handle_voice_clone_request(input, responseFormat):
    """Pure API orchestration: Handle voice cloning requests"""
    global vc_model
    
    # ===== COMPREHENSIVE INPUT PARAMETER LOGGING =====
    logger.info("🔍 ===== VC HANDLER INPUT PARAMETERS =====")
    logger.info(f"📥 Raw input keys: {list(input.keys())}")
    logger.info(f"📥 Input type: {type(input)}")
    
    # Log all input parameters
    for key, value in input.items():
        if key == 'audio_data' and value:
            logger.info(f"📥 {key}: [BASE64 DATA] Length: {len(value)} chars")
        elif isinstance(value, dict):
            logger.info(f"📥 {key}: {type(value)} with keys: {list(value.keys())}")
        else:
            logger.info(f"📥 {key}: {value}")
    
    # ===== FIREBASE CREDENTIAL VALIDATION =====
    _debug_gcs_creds()
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "error": "Failed to initialize Firebase storage"}
    
    # Check if VC model is available
    if vc_model is None:
        logger.error("❌ VC model not available")
        return {"status": "error", "error": "VC model not available"}
    
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
    callback_url = input.get('callback_url') or meta_top.get('callback_url')
    
    # ===== METADATA BREAKDOWN LOGGING =====
    logger.info("🔍 ===== METADATA BREAKDOWN =====")
    logger.info(f"📋 Top-level metadata: {meta_top}")
    logger.info(f"📋 Top-level metadata type: {type(meta_top)}")
    logger.info(f"📋 Top-level metadata keys: {list(meta_top.keys()) if isinstance(meta_top, dict) else 'Not a dict'}")
    
    # Log nested metadata if it exists
    nested_metadata = input.get('metadata', {})
    if isinstance(nested_metadata, dict):
        logger.info(f"📋 Nested metadata: {nested_metadata}")
        logger.info(f"📋 Nested metadata keys: {list(nested_metadata.keys())}")
        for key, value in nested_metadata.items():
            logger.info(f"📋   {key}: {value} (type: {type(value)})")
    else:
        logger.info(f"📋 Nested metadata: {nested_metadata} (type: {type(nested_metadata)})")
    
    logger.info("🔍 ===== END METADATA BREAKDOWN =====")

    if not name or (not audio_data and not audio_path):
        return {"status": "error", "error": "name and either audio_data or audio_path are required"}

    logger.info(f"New request. Voice clone name: {name}")
    logger.info(f"Response format requested: {responseFormat}")
    logger.info(f"Language: {language}, Kids voice: {is_kids_voice}")
    
    try:
        # Honor server-provided voice_id; fallback to legacy generation only if missing
        voice_id = input.get('voice_id') or meta_top.get('voice_id')
        if not voice_id:
            voice_id = get_voice_id(name)
            logger.info(f"No voice_id provided; generated legacy id: {voice_id}")
        else:
            logger.info(f"Using provided voice_id: {voice_id}")

        # Enforce deterministic filenames: {voiceId}.npy|.mp3; recorded filename is only a local temp hint
        try:
            target_profile_name = profile_filename_hint or f"{voice_id}.npy"
            target_sample_name = sample_filename_hint or f"{voice_id}.mp3"
        except Exception:
            target_profile_name = profile_filename_hint or f"{voice_id}.npy"
            target_sample_name = sample_filename_hint or f"{voice_id}.mp3"
        
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
                    return {"status": "error", "error": "Failed to initialize Firebase storage"}
                blob = bucket.blob(audio_path)
                blob.download_to_filename(str(temp_voice_file))
                logger.info(f"Downloaded audio from Firebase {audio_path} to {temp_voice_file}")
            except Exception as dl_e:
                logger.error(f"❌ Failed to download audio_path {audio_path}: {dl_e}")
                return {"status": "error", "error": f"Failed to download audio_path: {dl_e}"}
        else:
            temp_voice_file = TEMP_VOICE_DIR / f"{voice_id}_{timestamp}.{audio_format}"
            audio_bytes = base64.b64decode(audio_data)
            with open(temp_voice_file, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Saved temporary voice file to {temp_voice_file}")

        # Call the VC model's create_voice_clone method
        logger.info("🔄 Calling VC model's create_voice_clone method...")
        
        # ===== API METADATA PREPARATION LOGGING =====
        logger.info("🔍 ===== API METADATA PREPARATION =====")
        
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
            # Explicit filenames (if provided/derived) so model uploads with the exact names
            'profile_filename': target_profile_name,
            'sample_filename': target_sample_name,
            # Strong metadata contract for Storage uploads downstream
            'storage_metadata': {
                'user_id': input.get('user_id') or '',
                'voice_id': voice_id,
                'voice_name': name,
                'language': language,
                'is_kids_voice': str(is_kids_voice).lower(),
                'model_type': input.get('model_type') or 'chatterbox',
            }
        }
        
        logger.info(f"📋 API metadata prepared: {api_metadata}")
        logger.info(f"📋 API metadata type: {type(api_metadata)}")
        logger.info(f"📋 API metadata keys: {list(api_metadata.keys())}")
        
        # Log nested metadata structure
        nested_metadata = api_metadata.get('storage_metadata', {})
        logger.info(f"📋 Nested storage metadata: {nested_metadata}")
        logger.info(f"📋 Nested storage metadata type: {type(nested_metadata)}")
        logger.info(f"📋 Nested storage metadata keys: {list(nested_metadata.keys())}")
        
        # Log each metadata field with type information
        for key, value in nested_metadata.items():
            logger.info(f"📋   Storage metadata {key}: {value} (type: {type(value)})")
        
        logger.info("🔍 ===== END API METADATA PREPARATION =====")
        
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

        # ===== POST-GENERATION METADATA VERIFICATION =====
        logger.info("🔍 ===== POST-GENERATION METADATA VERIFICATION =====")
        
        # Verify metadata was set on uploaded files
        try:
            if isinstance(result, dict) and result.get("status") == "success":
                # Build Firebase paths based on language and kids voice
                kids_segment = 'kids/' if is_kids_voice else ''
                base_firebase_path = f"audio/voices/{language}/{kids_segment}"
                
                # Check profile file metadata
                profile_filename = result.get("profile_path")
                if profile_filename:
                    # Construct full Firebase path
                    profile_firebase_path = f"{base_firebase_path}profiles/{profile_filename}"
                    logger.info(f"🔍 Verifying metadata on profile: {profile_firebase_path}")
                    try:
                        blob = bucket.blob(profile_firebase_path)
                        if blob.exists():
                            blob.reload()
                            actual_metadata = blob.metadata or {}
                            logger.info(f"📋 Profile metadata found: {actual_metadata}")
                            expected_metadata = {
                                'user_id': user_id or '',
                                'voice_id': voice_id,
                                'voice_name': name,
                                'language': language,
                                'is_kids_voice': str(is_kids_voice).lower(),
                            }
                            logger.info(f"📋 Expected profile metadata: {expected_metadata}")
                            
                            # Check if metadata matches
                            if actual_metadata == expected_metadata:
                                logger.info("✅ Profile metadata matches expected")
                            else:
                                logger.warning("⚠️ Profile metadata mismatch, attempting to fix...")
                                blob.metadata = expected_metadata
                                blob.patch()
                                logger.info("✅ Profile metadata fixed")
                        else:
                            logger.warning(f"⚠️ Profile blob does not exist: {profile_firebase_path}")
                    except Exception as profile_e:
                        logger.warning(f"⚠️ Could not verify profile metadata: {profile_e}")
                
                # Check sample file metadata
                sample_filename = result.get("sample_audio_path")
                if sample_filename:
                    # Construct full Firebase path
                    sample_firebase_path = f"{base_firebase_path}samples/{sample_filename}"
                    logger.info(f"🔍 Verifying metadata on sample: {sample_firebase_path}")
                    try:
                        blob = bucket.blob(sample_firebase_path)
                        if blob.exists():
                            blob.reload()
                            actual_metadata = blob.metadata or {}
                            logger.info(f"📋 Sample metadata found: {actual_metadata}")
                            expected_metadata = {
                                'user_id': user_id or '',
                                'voice_id': voice_id,
                                'voice_name': name,
                                'language': language,
                                'is_kids_voice': str(is_kids_voice).lower(),
                            }
                            logger.info(f"📋 Expected sample metadata: {expected_metadata}")
                            
                            # Check if metadata matches
                            if actual_metadata == expected_metadata:
                                logger.info("✅ Sample metadata matches expected")
                            else:
                                logger.warning("⚠️ Sample metadata mismatch, attempting to fix...")
                                blob.metadata = expected_metadata
                                blob.patch()
                                logger.info("✅ Sample metadata fixed")
                        else:
                            logger.warning(f"⚠️ Sample blob does not exist: {sample_firebase_path}")
                    except Exception as sample_e:
                        logger.warning(f"⚠️ Could not verify sample metadata: {sample_e}")
                
                # Check recorded file metadata (if available)
                recorded_filename = result.get("recorded_audio_path")
                if recorded_filename:
                    # Check if recorded_filename is already a full Firebase path
                    if recorded_filename.startswith("audio/voices/"):
                        recorded_firebase_path = recorded_filename
                    else:
                        # Construct full Firebase path if it's just a filename
                        recorded_firebase_path = f"{base_firebase_path}recorded/{recorded_filename}"
                    logger.info(f"🔍 Verifying metadata on recorded: {recorded_firebase_path}")
                    try:
                        blob = bucket.blob(recorded_firebase_path)
                        if blob.exists():
                            blob.reload()
                            actual_metadata = blob.metadata or {}
                            logger.info(f"📋 Recorded metadata found: {actual_metadata}")
                            expected_metadata = {
                                'user_id': user_id or '',
                                'voice_id': voice_id,
                                'voice_name': name,
                                'language': language,
                                'is_kids_voice': str(is_kids_voice).lower(),
                            }
                            logger.info(f"📋 Expected recorded metadata: {expected_metadata}")
                            
                            # Check if metadata matches
                            if actual_metadata == expected_metadata:
                                logger.info("✅ Recorded metadata matches expected")
                            else:
                                logger.warning("⚠️ Recorded metadata mismatch, attempting to fix...")
                                blob.metadata = expected_metadata
                                blob.patch()
                                logger.info("✅ Recorded metadata fixed")
                        else:
                            logger.info(f"ℹ️ Recorded blob does not exist (expected for pointer-based recordings): {recorded_firebase_path}")
                    except Exception as recorded_e:
                        logger.warning(f"⚠️ Could not verify recorded metadata: {recorded_e}")
        except Exception as verify_e:
            logger.warning(f"⚠️ Metadata verification failed: {verify_e}")
        
        logger.info("🔍 ===== END POST-GENERATION METADATA VERIFICATION =====")
        
        # No post-process renaming: model now uploads with standardized names directly

        # ===== SUCCESS CALLBACK LOGGING =====
        logger.info("🔍 ===== SUCCESS CALLBACK PAYLOAD =====")
        
        # Attempt callback on success
        try:
            if callback_url:
                try:
                    # Build storage paths using the exact filenames
                    kids_segment = 'kids/' if is_kids_voice else ''
                    profile_path = f"audio/voices/{language}/{kids_segment}profiles/{target_profile_name}"
                    sample_path = f"audio/voices/{language}/{kids_segment}samples/{target_sample_name}"
                    payload = {
                        'status': 'success',
                        'user_id': user_id,
                        'voice_id': voice_id,
                        'voice_name': name,
                        'language': language,
                        'profile_path': profile_path,
                        'sample_path': sample_path,
                    }
                    
                    logger.info(f"📤 Success callback URL: {callback_url}")
                    logger.info(f"📤 Success callback payload: {payload}")
                    logger.info(f"📤 Success callback payload type: {type(payload)}")
                    logger.info(f"📤 Success callback payload keys: {list(payload.keys())}")
                    
                    _post_signed_callback(callback_url, payload)
                    logger.info("✅ Success callback sent successfully")
                except Exception as cb_e:
                    logger.warning(f"⚠️ Success callback failed: {cb_e}")
                    logger.warning(f"⚠️ Success callback exception type: {type(cb_e)}")
        except Exception as e:
            logger.warning(f"⚠️ Success callback preparation failed: {e}")
        
        logger.info("🔍 ===== END SUCCESS CALLBACK PAYLOAD =====")

        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if 'temp_voice_file' in locals():
            try:
                os.unlink(temp_voice_file)
            except:
                pass
        # ===== ERROR CALLBACK LOGGING =====
        logger.info("🔍 ===== ERROR CALLBACK PAYLOAD =====")
        
        # Attempt error callback
        try:
            if 'callback_url' in locals() and callback_url:
                try:
                    payload = {
                        'status': 'error',
                        'user_id': user_id,
                        'voice_id': input.get('voice_id') or meta_top.get('voice_id') or '',
                        'voice_name': name,
                        'language': language,
                        'error': str(e),
                    }
                    
                    logger.info(f"📤 Error callback URL: {callback_url}")
                    logger.info(f"📤 Error callback payload: {payload}")
                    logger.info(f"📤 Error callback payload type: {type(payload)}")
                    logger.info(f"📤 Error callback payload keys: {list(payload.keys())}")
                    
                    _post_signed_callback(callback_url, payload)
                    logger.info("✅ Error callback sent successfully")
                except Exception as cb_e:
                    logger.warning(f"⚠️ Error callback failed: {cb_e}")
                    logger.warning(f"⚠️ Error callback exception type: {type(cb_e)}")
        except Exception as callback_prep_e:
            logger.warning(f"⚠️ Error callback preparation failed: {callback_prep_e}")
        
        logger.info("🔍 ===== END ERROR CALLBACK PAYLOAD =====")
        return {"status": "error", "error": str(e)}


def _post_signed_callback(callback_url: str, payload: dict):
    """POST JSON payload to callback_url with HMAC headers compatible with app callback."""
    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
    if not secret:
        raise RuntimeError('DAEZEND_API_SHARED_SECRET not set; cannot sign callback')

    parsed = urlparse(callback_url)
    path_for_signing = parsed.path or '/api/voice-clone/callback'
    ts = str(int(time.time() * 1000))
    body_bytes = json.dumps(payload).encode('utf-8')
    prefix = f"POST\n{path_for_signing}\n{ts}\n".encode('utf-8')
    message = prefix + body_bytes
    sig = hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()

    req = Request(callback_url, data=body_bytes, headers={
        'Content-Type': 'application/json',
        'X-Daezend-Timestamp': ts,
        'X-Daezend-Signature': sig,
    }, method='POST')
    with urlopen(req, timeout=15) as resp:
        _ = resp.read()

if __name__ == '__main__':
    logger.info("🚀 Voice Clone Handler starting...")
    logger.info("✅ Voice Clone Handler ready")
    runpod.serverless.start({'handler': handler })

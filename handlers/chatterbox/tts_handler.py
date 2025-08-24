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

"""Minimal, production-focused TTS handler for RunPod runtime."""

def clear_python_cache():
    """Clear Python caches and loaded chatterbox modules to ensure fresh load."""
    try:
        for pyc_file in glob.glob("/workspace/**/*.pyc", recursive=True):
            try:
                os.remove(pyc_file)
            except Exception:
                pass
        for pycache_dir in pathlib.Path("/workspace").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
            except Exception:
                pass
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
VOICE_SAMPLES_DIR = Path("/voice_samples")  # For voice clone samples
TTS_GENERATED_DIR = Path("/tts_generated")  # For TTS story generation
TEMP_VOICE_DIR = Path("/temp_voice")

logger.info(f"Using directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
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
    logger.info("🔍 ===== TTS FIREBASE CREDENTIAL VALIDATION =====")
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
    
    logger.info("🔍 ===== END TTS FIREBASE CREDENTIAL VALIDATION =====")

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
            logger.info(f"🔍 RUNPOD_SECRET_Firebase: Loaded successfully")
            
            # Try to parse and validate the JSON
            try:
                import json
                cred_data = json.loads(firebase_secret)
                logger.info(f"🔍 Firebase Project ID: {cred_data.get('project_id', 'NOT FOUND')}")
                logger.info(f"🔍 Firebase Client Email: {cred_data.get('client_email', 'NOT FOUND')}")
                logger.info("✅ Firebase credentials JSON is valid")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Firebase credentials JSON is invalid: {e}")
                logger.error(f"❌ Credentials validation failed")
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
        # Set metadata/content type and persist
        if metadata:
            new_blob.metadata = metadata
        if content_type:
            new_blob.content_type = content_type
        try:
            new_blob.patch()
        except Exception as patch_e:
            logger.warning(f"⚠️ Could not patch metadata for {dest_path}: {patch_e}")
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
    
    logger.info(f"🎯 ===== CALLING TTS GENERATION =====")
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
        
        # Try to use the TTS model's generate_tts_story method
        if hasattr(tts_model, 'generate_tts_story'):
            logger.info("🔄 Using TTS model's generate_tts_story method...")
            try:
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
                logger.info(f"✅ TTS generation completed in {generation_time:.2f}s")
                return result
            except Exception as method_error:
                logger.error(f"❌ generate_tts_story method failed: {method_error}")
                return {
                    "status": "error",
                    "message": f"generate_tts_story method failed: {method_error}",
                    "generation_time": time.time() - start_time
                }
        else:
            logger.error("❌ TTS model doesn't have generate_tts_story method")
            return {
                "status": "error",
                "message": "TTS model doesn't have generate_tts_story method. Please update the RunPod deployment with the latest forked repository version.",
                "generation_time": time.time() - start_time
            }
        
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
    
    # ===== COMPREHENSIVE INPUT PARAMETER LOGGING =====
    logger.info("🔍 ===== TTS HANDLER INPUT PARAMETERS =====")
    logger.info(f"📥 Raw event keys: {list(event.keys())}")
    logger.info(f"📥 Event type: {type(event)}")
    
    # Log event structure
    if "input" in event:
        logger.info(f"📥 Input keys: {list(event['input'].keys())}")
        for key, value in event["input"].items():
            if key == 'profile_base64' and value:
                logger.info(f"📥 {key}: [BASE64 DATA] Length: {len(value)} chars")
            elif isinstance(value, dict):
                logger.info(f"📥 {key}: {type(value)} with keys: {list(value.keys())}")
            else:
                logger.info(f"📥 {key}: {value}")
    
    if "metadata" in event:
        logger.info(f"📥 Top-level metadata keys: {list(event['metadata'].keys()) if isinstance(event['metadata'], dict) else 'Not a dict'}")
        logger.info(f"📥 Top-level metadata: {event['metadata']}")
    
    # Initialize Firebase at the start
    if not initialize_firebase():
        logger.error("❌ Failed to initialize Firebase, cannot proceed")
        return {"status": "error", "error": "Failed to initialize Firebase storage"}
    
    # Check if TTS model is available
    if tts_model is None:
        logger.error("❌ TTS model not available")
        return {"status": "error", "error": "TTS model not available"}
    
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
    
    # ===== METADATA BREAKDOWN LOGGING =====
    logger.info("🔍 ===== TTS METADATA BREAKDOWN =====")
    logger.info(f"📋 API metadata: {api_metadata}")
    logger.info(f"📋 API metadata type: {type(api_metadata)}")
    logger.info(f"📋 API metadata keys: {list(api_metadata.keys()) if isinstance(api_metadata, dict) else 'Not a dict'}")
    
    # Log each metadata field with type information
    if isinstance(api_metadata, dict):
        for key, value in api_metadata.items():
            logger.info(f"📋   API metadata {key}: {value} (type: {type(value)})")
    
    # Log top-level metadata
    top_metadata = event.get("metadata", {})
    logger.info(f"📋 Top-level metadata: {top_metadata}")
    logger.info(f"📋 Top-level metadata type: {type(top_metadata)}")
    if isinstance(top_metadata, dict):
        logger.info(f"📋 Top-level metadata keys: {list(top_metadata.keys())}")
        for key, value in top_metadata.items():
            logger.info(f"📋   Top-level metadata {key}: {value} (type: {type(value)})")
    
    logger.info("🔍 ===== END TTS METADATA BREAKDOWN =====")
    
    # ===== FIREBASE CREDENTIAL VALIDATION =====
    _debug_gcs_creds()
    
    if not text or not voice_id:
        return {"status": "error", "error": "Both text and voice_id are required"}

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
        
        # ===== POST-GENERATION METADATA VERIFICATION =====
        logger.info("🔍 ===== TTS POST-GENERATION METADATA VERIFICATION =====")
        
        # Verify metadata was set on uploaded files
        try:
            if isinstance(result, dict) and result.get("status") == "success":
                # Check audio file metadata
                firebase_path = result.get("firebase_path")
                if firebase_path:
                    logger.info(f"🔍 Verifying metadata on TTS audio: {firebase_path}")
                    try:
                        blob = bucket.blob(firebase_path)
                        if blob.exists():
                            blob.reload()
                            actual_metadata = blob.metadata or {}
                            logger.info(f"📋 TTS audio metadata found: {actual_metadata}")
                            expected_metadata = {
                                'user_id': user_id or '',
                                'story_id': story_id or '',
                                'voice_id': voice_id,
                                'language': language,
                                'story_type': story_type,
                                'story_name': story_name or '',
                            }
                            logger.info(f"📋 Expected TTS audio metadata: {expected_metadata}")
                            
                            # Check if metadata matches
                            if actual_metadata == expected_metadata:
                                logger.info("✅ TTS audio metadata matches expected")
                            else:
                                logger.warning("⚠️ TTS audio metadata mismatch, attempting to fix...")
                                blob.metadata = expected_metadata
                                blob.patch()
                                logger.info("✅ TTS audio metadata fixed")
                        else:
                            logger.warning(f"⚠️ TTS audio blob does not exist: {firebase_path}")
                            
                            # Try to construct the path if it's just a filename
                            if not firebase_path.startswith('audio/'):
                                # Build Firebase path based on language and story type
                                constructed_path = f"audio/stories/{language}/user/{(user_id or 'user')}/{firebase_path}"
                                logger.info(f"🔍 Trying constructed path: {constructed_path}")
                                try:
                                    blob = bucket.blob(constructed_path)
                                    if blob.exists():
                                        blob.reload()
                                        actual_metadata = blob.metadata or {}
                                        logger.info(f"📋 TTS audio metadata found (constructed path): {actual_metadata}")
                                        expected_metadata = {
                                            'user_id': user_id or '',
                                            'story_id': story_id or '',
                                            'voice_id': voice_id,
                                            'language': language,
                                            'story_type': story_type,
                                            'story_name': story_name or '',
                                        }
                                        logger.info(f"📋 Expected TTS audio metadata: {expected_metadata}")
                                        
                                        # Check if metadata matches
                                        if actual_metadata == expected_metadata:
                                            logger.info("✅ TTS audio metadata matches expected (constructed path)")
                                        else:
                                            logger.warning("⚠️ TTS audio metadata mismatch, attempting to fix...")
                                            blob.metadata = expected_metadata
                                            blob.patch()
                                            logger.info("✅ TTS audio metadata fixed (constructed path)")
                                    else:
                                        logger.warning(f"⚠️ TTS audio blob does not exist (constructed path): {constructed_path}")
                                except Exception as constructed_e:
                                    logger.warning(f"⚠️ Could not verify TTS audio metadata (constructed path): {constructed_e}")
                    except Exception as audio_e:
                        logger.warning(f"⚠️ Could not verify TTS audio metadata: {audio_e}")
        except Exception as verify_e:
            logger.warning(f"⚠️ TTS metadata verification failed: {verify_e}")
        
        logger.info("🔍 ===== END TTS POST-GENERATION METADATA VERIFICATION =====")
        
        # Post-process: rename output file to requested naming if model saved with default name.
        try:
            if isinstance(result, dict) and result.get("status") == "success":
                # Extract hints
                import re
                user_id = api_metadata.get("user_id") or event["input"].get("user_id")
                voice_id = event["input"].get("voice_id")
                story_type = event["input"].get("story_type", "user")
                language = event["input"].get("language", "en")
                story_name = api_metadata.get("story_name") or event["input"].get("story_name")
                story_id = api_metadata.get("story_id") or event["input"].get("story_id")
                output_basename = api_metadata.get("output_basename") or event["input"].get("output_basename")
                if not story_name and output_basename:
                    story_name = output_basename.split("_")[0]
                safe_story = re.sub(r'[^a-z0-9]+', '_', (story_name or 'story').lower()).strip('_')
                base = output_basename or f"{safe_story}_{voice_id}_{story_type}"
                # Determine existing firebase path
                firebase_path = result.get("firebase_path")
                audio_url = result.get("firebase_url") or result.get("audio_url")
                # If we only have URL, derive path from it
                if not firebase_path and audio_url:
                    from urllib.parse import urlparse
                    p = urlparse(audio_url).path
                    firebase_path = p[1:] if p.startswith('/') else p
                # Build target path
                if firebase_path:
                    ext = firebase_path.split('.')[-1].lower() if '.' in firebase_path else 'mp3'
                    # Store under audio/stories/{language}/user/{user_id}/{file}
                    target_path = f"audio/stories/{language}/user/{(user_id or 'user')}/{base}.{ext}"
                    if target_path != firebase_path:
                        new_url = rename_in_firebase(
                            firebase_path,
                            target_path,
                            metadata={
                                'user_id': (user_id or ''),
                                'story_id': (story_id or ''),
                                'voice_id': (voice_id or ''),
                                'language': (language or ''),
                                'story_type': (story_type or ''),
                                'story_name': (safe_story or ''),
                            },
                            content_type='audio/mpeg' if ext == 'mp3' else 'audio/wav'
                        )
                        if new_url:
                            result['firebase_path'] = target_path
                            result['firebase_url'] = new_url
                            result['audio_url'] = new_url
                        else:
                            # Persist metadata on the original blob as a fallback
                            try:
                                global bucket
                                if bucket is None:
                                    initialize_firebase()
                                if bucket is not None:
                                    b = bucket.blob(firebase_path)
                                    if b.exists():
                                        b.metadata = {
                                            'user_id': (user_id or ''),
                                            'story_id': (story_id or ''),
                                            'voice_id': (voice_id or ''),
                                            'language': (language or ''),
                                            'story_type': (story_type or ''),
                                            'story_name': (safe_story or ''),
                                        }
                                        try:
                                            b.patch()
                                        except Exception:
                                            pass
                            except Exception as meta_e:
                                logger.warning(f"⚠️ Could not set metadata on original blob: {meta_e}")
        except Exception as post_e:
            logger.warning(f"⚠️ TTS post-process rename failed: {post_e}")
        # ===== TTS SUCCESS CALLBACK LOGGING =====
        logger.info("🔍 ===== TTS SUCCESS CALLBACK PAYLOAD =====")
        
        # If callback_url provided, post completion payload
        try:
            if callback_url and isinstance(result, dict) and result.get("status") == "success":
                import requests
                payload = {
                    "story_id": story_id,
                    "user_id": user_id,
                    "voice_id": voice_id,
                    "audio_url": result.get("firebase_url") or result.get("audio_url") or result.get("audio_path"),
                    "storage_path": result.get("firebase_path"),
                    "language": language,
                    "metadata": {
                        **({} if not isinstance(api_metadata, dict) else api_metadata),
                        "generation_time": result.get("generation_time"),
                    },
                }
                
                logger.info(f"📤 TTS callback URL: {callback_url}")
                logger.info(f"📤 TTS callback payload: {payload}")
                logger.info(f"📤 TTS callback payload type: {type(payload)}")
                logger.info(f"📤 TTS callback payload keys: {list(payload.keys())}")
                
                # Log nested metadata in callback
                callback_metadata = payload.get("metadata", {})
                logger.info(f"📤 TTS callback metadata: {callback_metadata}")
                logger.info(f"📤 TTS callback metadata type: {type(callback_metadata)}")
                if isinstance(callback_metadata, dict):
                    logger.info(f"📤 TTS callback metadata keys: {list(callback_metadata.keys())}")
                    for key, value in callback_metadata.items():
                        logger.info(f"📤   TTS callback metadata {key}: {value} (type: {type(value)})")
                
                try:
                    resp = requests.post(callback_url, json=payload, timeout=10)
                    logger.info(f"✅ TTS callback POST {callback_url} -> {resp.status_code}")
                except Exception as cb_e:
                    logger.warning(f"⚠️ TTS callback POST failed: {cb_e}")
                    logger.warning(f"⚠️ TTS callback exception type: {type(cb_e)}")
        except Exception as e:
            logger.warning(f"⚠️ Error preparing TTS callback: {e}")
            logger.warning(f"⚠️ TTS callback preparation exception type: {type(e)}")
        
        logger.info("🔍 ===== END TTS SUCCESS CALLBACK PAYLOAD =====")
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"status": "error", "error": str(e)}

def handle_file_download(input):
    """Handle file download requests"""
    file_path = input.get("file_path")
    if not file_path:
        return {"status": "error", "error": "file_path is required"}
    
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        return {
            "status": "success",
            "file_data": base64.b64encode(file_data).decode('utf-8'),
            "file_size": len(file_data)
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to read file: {e}"}

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
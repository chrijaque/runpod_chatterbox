import firebase_admin
from firebase_admin import credentials, storage
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any
import os
import shutil

logger = logging.getLogger(__name__)

class FirebaseService:
    """Firebase Storage service for file operations"""
    
    def __init__(self, credentials_json: str, bucket_name: str):
        self.credentials_json = credentials_json
        self.bucket_name = bucket_name
        self.bucket = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Parse JSON credentials from environment
                import json
                import tempfile
                
                try:
                    # Parse the JSON credentials
                    cred_data = json.loads(self.credentials_json)
                    
                    # Validate required fields
                    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                    missing_fields = [field for field in required_fields if field not in cred_data]
                    
                    if missing_fields:
                        logger.error(f"❌ Missing required credential fields: {missing_fields}")
                        raise ValueError(f"Missing required credential fields: {missing_fields}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON credentials: {e}")
                    raise
                except Exception as e:
                    logger.error(f"❌ Error processing credentials: {e}")
                    raise
                
                # Create temporary file with the credentials
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(cred_data, tmp_file)
                    tmp_path = tmp_file.name
                
                # Initialize Firebase with the temporary file
                try:
                    cred = credentials.Certificate(tmp_path)
                    firebase_admin.initialize_app(cred)
                    logger.info("✅ Firebase initialized successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Firebase initialization failed: {e}")
                    raise
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to clean up temporary credentials file: {e}")
            
            # Try to get the bucket
            try:
                self.bucket = storage.bucket(self.bucket_name)
                
                # Test if the bucket actually exists by trying to list blobs
                try:
                    # This will fail if the bucket doesn't exist
                    list(self.bucket.list_blobs(max_results=1))
                    logger.info(f"✅ Connected to Firebase bucket: {self.bucket_name}")
                except Exception as list_error:
                    logger.warning(f"⚠️ Bucket exists but not accessible: {list_error}")
                    logger.info(f"🔧 Attempting to create bucket: {self.bucket_name}")
                    
                    # Try to create the bucket
                    if self._create_bucket():
                        self.bucket = storage.bucket(self.bucket_name)
                        logger.info(f"✅ Successfully created and connected to bucket: {self.bucket_name}")
                    else:
                        logger.error(f"❌ Failed to create bucket: {self.bucket_name}")
                        self.bucket = None
                        
            except Exception as bucket_error:
                logger.warning(f"⚠️ Bucket not found: {bucket_error}")
                logger.info(f"🔧 Attempting to create bucket: {self.bucket_name}")
                
                # Try to create the bucket
                if self._create_bucket():
                    self.bucket = storage.bucket(self.bucket_name)
                    logger.info(f"✅ Successfully created and connected to bucket: {self.bucket_name}")
                else:
                    logger.error(f"❌ Failed to create bucket: {self.bucket_name}")
                    self.bucket = None
            
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {e}")
            self.bucket = None
    
    def _create_bucket(self) -> bool:
        """Create Firebase Storage bucket if it doesn't exist"""
        try:
            from google.cloud import storage as gcs_storage
            
            # Create a client using the same credentials
            import json
            import tempfile
            
            # Parse the JSON credentials
            cred_data = json.loads(self.credentials_json)
            
            # Create temporary file with the credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(cred_data, tmp_file)
                tmp_path = tmp_file.name
            
            client = gcs_storage.Client.from_service_account_json(tmp_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up temporary credentials file: {e}")
            logger.info(f"🔍 Google Cloud Storage client created")
            
            # Extract project ID from bucket name
            # Our bucket format is: project-id.firebasestorage.app
            # For creation, we need just the project ID
            project_id = self.bucket_name.replace('.firebasestorage.app', '')
            
            logger.info(f"🔧 Creating bucket with project ID: {project_id}")
            logger.info(f"🔧 Full bucket name will be: {self.bucket_name}")
            
            # Create the bucket
            logger.info(f"🔍 Creating bucket with name: {self.bucket_name} and project: {project_id}")
            bucket = client.create_bucket(self.bucket_name, project=project_id)
            logger.info(f"✅ Successfully created bucket: {self.bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create bucket {self.bucket_name}: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            
            # Check if bucket already exists
            if "already exists" in str(e).lower():
                logger.info(f"ℹ️ Bucket {self.bucket_name} already exists")
                return True
            
            return False
    
    def upload_file(self, file_path: Path, destination_blob_name: str) -> Optional[str]:
        """
        Upload file to Firebase Storage and return public URL
        
        :param file_path: Local file path
        :param destination_blob_name: Destination path in Firebase
        :return: Public URL or None if failed
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return None
        
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(str(file_path))
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"✅ File uploaded to Firebase: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Firebase upload failed: {e}")
            return None
    
    def download_file(self, blob_name: str, local_path: Path) -> bool:
        """
        Download file from Firebase Storage
        
        :param blob_name: Firebase blob name
        :param local_path: Local file path to save to
        :return: True if successful, False otherwise
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return False
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            logger.info(f"✅ File downloaded from Firebase: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Firebase download failed: {e}")
            return False
    
    def file_exists(self, blob_name: str) -> bool:
        """
        Check if file exists in Firebase Storage
        
        :param blob_name: Firebase blob name
        :return: True if exists, False otherwise
        """
        if not self.bucket:
            return False
        
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"❌ Firebase existence check failed: {e}")
            return False
    
    def get_public_url(self, blob_name: str) -> Optional[str]:
        """
        Get public URL for a file in Firebase Storage
        
        :param blob_name: Firebase blob name
        :return: Public URL or None if not found
        """
        if not self.bucket:
            return None
        
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.public_url
            return None
        except Exception as e:
            logger.error(f"❌ Firebase URL retrieval failed: {e}")
            return None
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete file from Firebase Storage
        
        :param blob_name: Firebase blob name
        :return: True if successful, False otherwise
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return False
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"✅ File deleted from Firebase: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Firebase deletion failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if Firebase is connected"""
        return self.bucket is not None 

    def upload_from_runpod_directory(self, runpod_file_path: str, firebase_path: str) -> Optional[str]:
        """
        Upload file from RunPod directory to Firebase Storage
        
        :param runpod_file_path: Path to file in RunPod container (e.g., /voice_samples/file.wav)
        :param firebase_path: Destination path in Firebase (e.g., voices/samples/file.wav)
        :return: Public URL or None if failed
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return None
        
        runpod_path = Path(runpod_file_path)
        if not runpod_path.exists():
            logger.error(f"RunPod file not found: {runpod_file_path}")
            return None
        
        try:
            # Ensure the directory structure exists
            self._ensure_directory_structure(firebase_path)
            
            blob = self.bucket.blob(firebase_path)
            blob.upload_from_filename(str(runpod_path))
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"✅ Uploaded from RunPod to Firebase: {runpod_file_path} → {public_url}")
            return public_url
            
        except Exception as e:
            # If bucket doesn't exist yet, provide helpful message
            if "404" in str(e) and "bucket does not exist" in str(e):
                logger.info(f"📁 Firebase bucket will be created automatically on first upload: {firebase_path}")
                return None
            else:
                logger.error(f"❌ Firebase upload from RunPod failed: {e}")
                return None

    def upload_runpod_voice_sample(self, voice_id: str, sample_filename: str, language: str = "en", is_kids_voice: bool = False, *, user_id: Optional[str] = None, voice_name: Optional[str] = None) -> Optional[str]:
        """
        Upload voice sample from RunPod to Firebase with organized structure
        
        :param voice_id: Voice identifier
        :param sample_filename: Filename in RunPod directory
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether this is a kids voice
        :return: Public URL or None if failed
        """
        runpod_path = f"/voice_samples/{sample_filename}"
        
        # Build Firebase path based on language and kids voice
        safe_name = (voice_name or voice_id or "voice").lower().replace(" ", "_")
        if is_kids_voice:
            firebase_path = f"audio/voices/{language}/kids/samples/sample_{safe_name}_{user_id or 'user'}.mp3"
        else:
            firebase_path = f"audio/voices/{language}/samples/sample_{safe_name}_{user_id or 'user'}.mp3"

        url = self.upload_from_runpod_directory(runpod_path, firebase_path)
        if url:
            try:
                blob = self.bucket.blob(firebase_path)
                blob.metadata = {
                    "user_id": user_id or "",
                    "voice_id": voice_id,
                    "voice_name": voice_name or safe_name,
                    "language": language,
                    "is_kids_voice": str(bool(is_kids_voice)).lower(),
                }
                blob.patch()
            except Exception as e:
                logger.warning(f"⚠️ Could not set metadata for {firebase_path}: {e}")
        return url

    def upload_runpod_tts_generation(self, generation_id: str, tts_filename: str, language: str = "en", is_kids_voice: bool = False, story_type: str = "user", *, user_id: Optional[str] = None, voice_id: Optional[str] = None, voice_name: Optional[str] = None, story_name: Optional[str] = None, output_basename: Optional[str] = None) -> Optional[str]:
        """
        Upload TTS generation from RunPod to Firebase with organized stories structure
        
        :param generation_id: Generation identifier
        :param tts_filename: Filename in RunPod directory
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether this is a kids voice
        :param story_type: Type of story ("user" or "app")
        :return: Public URL or None if failed
        """
        runpod_path = f"/voice_samples/{tts_filename}"  # TTS files are in voice_samples
        
        # Build Firebase path based on language and story type
        safe_story = (story_name or generation_id or "story").lower().replace(" ", "_")
        base = output_basename or f"{safe_story}_{voice_id or 'voice'}_{story_type}"
        ext = tts_filename.split('.')[-1].lower() if '.' in tts_filename else 'mp3'
        firebase_path = f"audio/stories/{language}/{story_type}/{base}.{ext}"

        url = self.upload_from_runpod_directory(runpod_path, firebase_path)
        if url:
            try:
                blob = self.bucket.blob(firebase_path)
                blob.metadata = {
                    "user_id": user_id or "",
                    "voice_id": voice_id or "",
                    "voice_name": voice_name or "",
                    "language": language,
                    "story_type": story_type,
                    "story_name": safe_story,
                }
                blob.patch()
            except Exception as e:
                logger.warning(f"⚠️ Could not set metadata for {firebase_path}: {e}")
        return url

    def upload_base64_profile(self, profile_base64: str, firebase_path: str) -> Optional[str]:
        """
        Upload base64 encoded profile data directly to Firebase
        
        :param profile_base64: Base64 encoded profile data
        :param firebase_path: Destination path in Firebase
        :return: Public URL or None if failed
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return None
        
        try:
            import base64
            import tempfile
            from pathlib import Path
            
            # Decode base64 data
            profile_data = base64.b64decode(profile_base64)
            
            # Create temporary file with .npy extension
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
                temp_file.write(profile_data)
                temp_path = Path(temp_file.name)
            
            try:
                # Upload to Firebase
                blob = self.bucket.blob(firebase_path)
                blob.upload_from_filename(str(temp_path))
                blob.make_public()
                
                public_url = blob.public_url
                logger.info(f"✅ Base64 profile uploaded to Firebase: {public_url}")
                return public_url
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean up temp file: {e}")
            
        except Exception as e:
            logger.error(f"❌ Firebase base64 profile upload failed: {e}")
            return None

    def upload_voice_profile(self, voice_id: str, profile_filename: str, language: str = "en", is_kids_voice: bool = False, *, user_id: Optional[str] = None, voice_name: Optional[str] = None) -> Optional[str]:
        """
        Upload voice profile from RunPod to Firebase
        
        :param voice_id: Voice identifier
        :param profile_filename: Profile filename in RunPod directory
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether this is a kids voice
        :return: Public URL or None if failed
        """
        runpod_path = f"/voice_profiles/{profile_filename}"
        
        # Build Firebase path based on language and kids voice
        safe_name = (voice_name or voice_id or "voice").lower().replace(" ", "_")
        if is_kids_voice:
            firebase_path = f"audio/voices/{language}/kids/profiles/voice_{safe_name}_{user_id or 'user'}.npy"
        else:
            firebase_path = f"audio/voices/{language}/profiles/voice_{safe_name}_{user_id or 'user'}.npy"

        url = self.upload_from_runpod_directory(runpod_path, firebase_path)
        if url:
            try:
                blob = self.bucket.blob(firebase_path)
                blob.metadata = {
                    "user_id": user_id or "",
                    "voice_id": voice_id,
                    "voice_name": voice_name or safe_name,
                    "language": language,
                    "is_kids_voice": str(bool(is_kids_voice)).lower(),
                }
                blob.patch()
            except Exception as e:
                logger.warning(f"⚠️ Could not set metadata for {firebase_path}: {e}")
        return url

    def upload_base64_audio(self, audio_base64: str, firebase_path: str) -> Optional[str]:
        """
        Upload base64 encoded audio data directly to Firebase
        
        :param audio_base64: Base64 encoded audio data
        :param firebase_path: Destination path in Firebase
        :return: Public URL or None if failed
        """
        if not self.bucket:
            logger.error("Firebase not initialized")
            return None
        
        try:
            import base64
            import tempfile
            from pathlib import Path
            
            # Ensure the directory structure exists
            self._ensure_directory_structure(firebase_path)
            
            # Decode base64 data
            audio_data = base64.b64decode(audio_base64)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = Path(temp_file.name)
            
            try:
                # Upload to Firebase
                logger.info(f"🔍 Uploading to Firebase path: {firebase_path}")
                logger.info(f"🔍 Using bucket: {self.bucket.name}")
                blob = self.bucket.blob(firebase_path)
                blob.upload_from_filename(str(temp_path))
                blob.make_public()
                
                public_url = blob.public_url
                logger.info(f"✅ Base64 audio uploaded to Firebase: {public_url}")
                return public_url
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean up temp file: {e}")
            
        except Exception as e:
            logger.error(f"❌ Firebase base64 upload failed: {e}")
            return None
    
    def _ensure_directory_structure(self, firebase_path: str):
        """
        Ensure the directory structure exists in Firebase Storage
        Creates placeholder files to establish the directory structure
        
        :param firebase_path: Full path to the file (e.g., 'audio/voices/en/samples/file.wav')
        """
        try:
            # Extract directory path
            path_parts = firebase_path.split('/')
            if len(path_parts) <= 1:
                return  # No directory to create
            
            # Create each level of the directory structure
            for i in range(1, len(path_parts)):
                directory_path = '/'.join(path_parts[:i])
                placeholder_path = f"{directory_path}/.placeholder"
                
                # Check if placeholder already exists
                placeholder_blob = self.bucket.blob(placeholder_path)
                if not placeholder_blob.exists():
                    # Create placeholder file
                    placeholder_blob.upload_from_string(
                        f"Directory placeholder for {directory_path}", 
                        content_type="text/plain"
                    )
                    logger.info(f"📁 Created directory: {directory_path}")
                else:
                    logger.debug(f"📁 Directory already exists: {directory_path}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not ensure directory structure for {firebase_path}: {e}")
            # Don't fail the upload if directory creation fails

    def upload_user_recording(self, voice_id: str, recording_filename: str, language: str = "en", is_kids_voice: bool = False) -> Optional[str]:
        """
        Upload user's raw recording to Firebase
        
        :param voice_id: Voice identifier
        :param recording_filename: Recording filename
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether this is a kids voice
        :return: Public URL or None if failed
        """
        # Build Firebase path based on language and kids voice
        if is_kids_voice:
            firebase_path = f"audio/voices/{language}/kids/recorded/{voice_id}_{recording_filename}"
        else:
            firebase_path = f"audio/voices/{language}/recorded/{voice_id}_{recording_filename}"
        
        return self.upload_from_runpod_directory(f"/temp_voice/{recording_filename}", firebase_path)

    def get_shared_access_url(self, firebase_path: str) -> Optional[str]:
        """
        Get shared access URL for a file in Firebase Storage
        
        :param firebase_path: Path in Firebase (e.g., voices/voice_123/samples/sample.wav)
        :return: Public URL that both apps can access
        """
        return self.get_public_url(firebase_path)

    def list_voice_files(self, voice_id: str, language: str = "en", is_kids_voice: bool = False) -> Dict[str, List[str]]:
        """
        List all files for a specific voice in Firebase
        
        :param voice_id: Voice identifier
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether this is a kids voice
        :return: Dictionary with file categories and URLs
        """
        if not self.bucket:
            return {"recorded": [], "samples": [], "profiles": []}
        
        try:
            # Build prefix based on language and kids voice
            if is_kids_voice:
                prefix = f"audio/voices/{language}/kids/"
            else:
                prefix = f"audio/voices/{language}/"
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            files = {
                "recorded": [],
                "samples": [],
                "profiles": []
            }
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                    
                # Filter by voice_id
                if voice_id not in blob.name:
                    continue
                    
                url = blob.public_url
                if "/recorded/" in blob.name:
                    files["recorded"].append(url)
                elif "/samples/" in blob.name:
                    files["samples"].append(url)
                elif "/profiles/" in blob.name:
                    files["profiles"].append(url)
            
            return files
            
        except Exception as e:
            # If bucket doesn't exist yet, return empty results instead of error
            if "404" in str(e) and "bucket does not exist" in str(e):
                logger.info(f"📁 Firebase bucket not created yet for {language} {('kids' if is_kids_voice else 'regular')} voices")
                return {"recorded": [], "samples": [], "profiles": []}
            else:
                logger.error(f"❌ Failed to list voice files: {e}")
                return {"recorded": [], "samples": [], "profiles": []}

    def list_voices_by_language(self, language: str = "en", is_kids_voice: bool = False) -> List[Dict[str, Any]]:
        """
        List all voices for a specific language and type
        
        :param language: Language code (e.g., "en", "es", "fr")
        :param is_kids_voice: Whether to list kids voices
        :return: List of voice information
        """
        if not self.bucket:
            logger.warning("❌ No Firebase bucket available")
            return []
        
        try:
            # Build prefix based on language and kids voice
            if is_kids_voice:
                prefix = f"audio/voices/{language}/kids/"
            else:
                prefix = f"audio/voices/{language}/"
            
            logger.info(f"🔍 Searching Firebase with prefix: {prefix}")
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            # Convert to list to get count
            blob_list = list(blobs)
            logger.info(f"📁 Found {len(blob_list)} blobs in Firebase")
            
            voices = {}
            
            for blob in blob_list:
                if blob.name.endswith('/'):
                    continue
                    
                # Skip placeholder files
                if blob.name.endswith('/.placeholder'):
                    continue
                

                
                # Extract path information
                path_parts = blob.name.split('/')
                
                # Determine file type based on directory structure
                file_type = None
                voice_id = None
                voice_name = None
                language = None
                is_kids_voice = None
                
                # Check directory structure to determine file type
                if len(path_parts) >= 4:
                    if path_parts[-2] == 'samples':
                        file_type = 'sample'
                    elif path_parts[-2] == 'profiles':
                        file_type = 'profile'
                    elif path_parts[-2] == 'recorded':
                        file_type = 'recorded'
                    else:
                        logger.info(f"   ⚠️ Skipping file with unknown directory: {blob.name}")
                        continue
                    
                    # Try to get metadata from the blob
                    try:
                        blob.reload()  # Ensure we have the latest metadata
                        metadata = blob.metadata or {}
                        
                        # Extract voice_id and other metadata from blob metadata
                        voice_id = metadata.get('voice_id')
                        voice_name = metadata.get('voice_name')
                        language = metadata.get('language', 'en')
                        is_kids_voice = metadata.get('is_kids_voice', 'false').lower() == 'true'
                        

                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not read metadata from {blob.name}: {e}")
                        metadata = {}
                    
                    # If no metadata or no voice_id in metadata, fallback to filename parsing
                    if not voice_id:
                        filename = path_parts[-1]
                        if file_type == 'sample' and '_sample' in filename:
                            voice_id = filename.split('_sample')[0]
                        elif file_type == 'profile':
                            voice_id = filename.replace('.npy', '')
                        elif file_type == 'recorded' and '_recorded' in filename:
                            voice_id = filename.split('_recorded')[0]
                        else:
                            # Fallback: use filename without extension as voice_id
                            voice_id = filename.rsplit('.', 1)[0]
                        

                    
                    # Skip if we still couldn't determine voice_id
                    if not voice_id:
                        continue
                
                # Skip if we couldn't determine file type or voice_id
                if not file_type or not voice_id:
                    continue
                    
                if voice_id not in voices:
                    voices[voice_id] = {
                        "voice_id": voice_id,
                        "name": voice_name or voice_id.replace('voice_', '').replace('_', ' ').title(),
                        "recorded": [],
                        "samples": [],
                        "profiles": [],
                        "created_date": None,
                        "language": language or "en",
                        "is_kids_voice": is_kids_voice or False
                    }
                
                url = blob.public_url
                
                # Categorize based on file_type determined from directory structure
                if file_type == 'recorded':
                    voices[voice_id]["recorded"].append(url)
                elif file_type == 'sample':
                    voices[voice_id]["samples"].append(url)
                elif file_type == 'profile':
                    voices[voice_id]["profiles"].append(url)
                
                # Try to extract creation time from metadata or filename
                try:
                    # First try to get creation date from metadata
                    if metadata and 'created_date' in metadata:
                        from datetime import datetime
                        created_date = datetime.fromisoformat(metadata['created_date'])
                        voices[voice_id]["created_date"] = int(created_date.timestamp())
                        logger.debug(f"   📅 Extracted timestamp from metadata: {metadata['created_date']}")
                    else:
                        # Fallback: try to extract from filename timestamp
                        filename = path_parts[-1]
                        if '_sample_' in filename:
                            timestamp_part = filename.split('_sample_')[-1].split('.')[0]
                        elif '_recording_' in filename:
                            timestamp_part = filename.split('_recording_')[-1].split('.')[0]
                        else:
                            # Try to find timestamp at the end of filename
                            timestamp_part = filename.split('_')[-1].split('.')[0]
                        
                        # Parse timestamp like 20250727_153420
                        if len(timestamp_part) >= 15 and timestamp_part.replace('_', '').isdigit():
                            from datetime import datetime
                            created_date = datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')
                            voices[voice_id]["created_date"] = int(created_date.timestamp())
                            logger.debug(f"   📅 Extracted timestamp from filename: {timestamp_part}")
                except Exception as e:
                    logger.debug(f"Could not parse timestamp from {blob.name}: {e}")
            
            # Filter to only include voices that have sample files (for library display)
            result = []
            for voice in voices.values():
                if voice["samples"]:  # Only include voices that have sample files
                    result.append(voice)
                    logger.info(f"✅ Voice {voice['voice_id']} has {len(voice['samples'])} sample(s)")
                else:
                    logger.info(f"⚠️ Skipping voice {voice['voice_id']} - no sample files found")
            
            logger.info(f"✅ Found {len(result)} voices with sample files")
            return result
            
        except Exception as e:
            # If bucket doesn't exist yet, return empty results instead of error
            if "404" in str(e) and "bucket does not exist" in str(e):
                logger.info(f"📁 Firebase bucket not created yet for {language} {('kids' if is_kids_voice else 'regular')} voices")
                return []
            else:
                logger.error(f"❌ Failed to list voices by language: {e}")
                return []

    def list_stories_by_language(self, language: str = "en", story_type: str = "user") -> List[Dict[str, Any]]:
        """
        List all stories for a specific language and type
        
        :param language: Language code (e.g., "en", "es", "fr")
        :param story_type: Type of story ("user" or "app")
        :return: List of story information
        """
        if not self.bucket:
            return []
        
        try:
            # Build prefix based on language and story type
            prefix = f"audio/stories/{language}/{story_type}/"
            logger.info(f"🔍 Searching Firebase with prefix: {prefix}")
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            # Convert to list for processing
            blob_list = list(blobs)
            logger.info(f"📁 Found {len(blob_list)} blobs in Firebase with prefix: {prefix}")
            
            stories = {}
            
            for blob in blob_list:
                if blob.name.endswith('/'):
                    continue
                    
                # Extract filename from blob path
                filename = blob.name.split('/')[-1]
                
                # List all audio files without name pattern restrictions
                if filename.endswith('.wav') or filename.endswith('.mp3') or filename.endswith('.m4a') or filename.endswith('.aac'):
                    logger.info(f"📁 Found audio file: {filename}")
                    
                    # Use filename (without extension) as generation_id
                    generation_id = filename.rsplit('.', 1)[0]  # Remove file extension
                    
                    # Try to extract voice_id from filename
                    voice_id = generation_id
                    voice_name = generation_id
                    
                    # If filename starts with common prefixes, extract voice info
                    if filename.startswith('TTS_'):
                        # Handle TTS_ prefixed files: TTS_voice_christianmp3test2_20250728_075206.wav
                        parts = generation_id.split('_')
                        if len(parts) >= 3:
                            voice_id = '_'.join(parts[1:-2]) if len(parts) >= 5 else '_'.join(parts[1:])
                            voice_name = voice_id.replace('voice_', '') if voice_id.startswith('voice_') else voice_id
                    elif filename.startswith('voice_'):
                        # Handle voice_ prefixed files: voice_startingv1.mp3
                        voice_id = generation_id
                        voice_name = generation_id.replace('voice_', '')
                    else:
                        # Handle any other naming pattern
                        voice_id = generation_id
                        voice_name = generation_id
                    
                    # Get creation time from blob metadata or use current time
                    created_date = None
                    if hasattr(blob, 'time_created') and blob.time_created:
                        created_date = int(blob.time_created.timestamp())
                    else:
                        # Try to extract timestamp from filename if it exists
                        import re
                        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
                        if timestamp_match:
                            try:
                                from datetime import datetime
                                timestamp_str = timestamp_match.group(1)
                                created_date = int(datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').timestamp())
                            except Exception as e:
                                logger.debug(f"Could not parse timestamp from {filename}: {e}")
                    
                    if not created_date:
                        created_date = int(blob.time_created.timestamp()) if hasattr(blob, 'time_created') and blob.time_created else None
                    
                    stories[generation_id] = {
                        "generation_id": generation_id,
                        "voice_id": voice_id,
                        "voice_name": voice_name,
                        "audio_file": blob.public_url,
                        "created_date": created_date,
                        "language": language,
                        "story_type": story_type,
                        "file_size": blob.size if hasattr(blob, 'size') else 0,
                        "filename": filename,
                        "timestamp": blob.time_created.isoformat() if hasattr(blob, 'time_created') and blob.time_created else None
                    }
                    
                    logger.info(f"✅ Added story: {generation_id} (voice: {voice_name})")
            
            logger.info(f"📚 Total stories found: {len(stories)}")
            return list(stories.values())
            
        except Exception as e:
            # If bucket doesn't exist yet, return empty results instead of error
            if "404" in str(e) and "bucket does not exist" in str(e):
                logger.info(f"📁 Firebase bucket not created yet for {language} {story_type} stories")
                return []
            else:
                logger.error(f"❌ Failed to list stories by language: {e}")
                return []

    def get_story_audio_url(self, generation_id: str, language: str = "en", story_type: str = "user") -> Optional[str]:
        """
        Get story audio URL from Firebase
        
        :param generation_id: Generation identifier
        :param language: Language code (e.g., "en", "es", "fr")
        :param story_type: Type of story ("user" or "app")
        :return: Public URL or None if not found
        """
        if not self.bucket:
            return None
        
        try:
            # List all files in the story directory
            prefix = f"audio/stories/{language}/{story_type}/{generation_id}_"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                return blob.public_url
            
            return None
            
        except Exception as e:
            # If bucket doesn't exist yet, return None instead of error
            if "404" in str(e) and "bucket does not exist" in str(e):
                logger.info(f"📁 Firebase bucket not created yet for {language} {story_type} stories")
                return None
            else:
                logger.error(f"❌ Failed to get story audio URL: {e}")
                return None

    def cleanup_runpod_file(self, runpod_file_path: str) -> bool:
        """
        Clean up file from RunPod directory after successful Firebase upload
        
        :param runpod_file_path: Path to file in RunPod container
        :return: True if successful, False otherwise
        """
        try:
            runpod_path = Path(runpod_file_path)
            if runpod_path.exists():
                os.unlink(runpod_path)
                logger.info(f"✅ Cleaned up RunPod file: {runpod_file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to cleanup RunPod file: {e}")
            return False

    def get_voice_profile_base64(self, voice_id: str, language: str = "en", is_kids_voice: bool = False) -> Optional[str]:
        """Get voice profile as base64 from Firebase"""
        try:
            if not self.is_connected():
                logger.error("Not connected to Firebase")
                return None
            
            # First, let's list all files in the profiles directory to see what's available
            profiles_prefix = f"audio/voices/{language}/profiles/"
            if is_kids_voice:
                profiles_prefix = f"audio/voices/{language}/kids/profiles/"
            
            logger.info(f"🔍 Listing files in profiles directory: {profiles_prefix}")
            blobs = list(self.bucket.list_blobs(prefix=profiles_prefix))
            
            # Log all available profile files
            for blob in blobs:
                if blob.name.endswith('.npy'):
                    logger.info(f"📁 Found profile file: {blob.name}")
            
            # Try to find the voice profile with the correct pattern
            # The voice profile is stored as: {voice_id}.npy
            target_filename = f"{profiles_prefix}{voice_id}.npy"
            logger.info(f"🔍 Looking for voice profile: {target_filename}")
            
            matching_blobs = [blob for blob in blobs if blob.name == target_filename]
            
            if not matching_blobs:
                logger.warning(f"No voice profile found for {voice_id} in {profiles_prefix}")
                return None
            
            # Use the first matching profile file
            blob = matching_blobs[0]
            logger.info(f"🔍 Using voice profile: {blob.name}")
            
            # Download the file content
            profile_data = blob.download_as_bytes()
            
            # Convert to base64
            import base64
            profile_base64 = base64.b64encode(profile_data).decode('utf-8')
            
            logger.info(f"✅ Successfully retrieved voice profile for {voice_id}")
            return profile_base64
            
        except Exception as e:
            logger.error(f"Error getting voice profile for {voice_id}: {e}")
            return None

    def get_storage_usage(self) -> Dict[str, int]:
        """
        Get storage usage statistics
        
        :return: Dictionary with storage metrics
        """
        if not self.bucket:
            return {}
        
        try:
            total_size = 0
            file_count = 0
            
            blobs = self.bucket.list_blobs()
            for blob in blobs:
                total_size += blob.size
                file_count += 1
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage usage: {e}")
            return {} 

    def test_list_all_files(self, prefix: str = "audio/stories/en/user/"):
        """
        Test method to list all files in Firebase bucket
        """
        if not self.bucket:
            logger.error("❌ No Firebase bucket available")
            return []
        
        try:
            logger.info(f"🔍 Testing Firebase file listing with prefix: {prefix}")
            blobs = self.bucket.list_blobs(prefix=prefix)
            blob_list = list(blobs)
            
            logger.info(f"📁 Found {len(blob_list)} files with prefix '{prefix}':")
            for blob in blob_list:
                logger.info(f"   📄 {blob.name}")
            
            return [blob.name for blob in blob_list]
            
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return [] 

    def upload_to_firebase(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
        """
        Upload data directly to Firebase Storage
        
        :param data: Binary data to upload
        :param destination_blob_name: Destination path in Firebase
        :param content_type: MIME type of the file
        :param metadata: Optional metadata to store with the file
        :return: Public URL or None if failed
        """
        if not self.bucket:
            logger.error("❌ Firebase not initialized")
            return None
        
        try:
            # Ensure the directory structure exists
            self._ensure_directory_structure(destination_blob_name)
            
            blob = self.bucket.blob(destination_blob_name)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            # Set content type and CORS headers
            blob.content_type = content_type
            
            # Upload the data
            blob.upload_from_string(data, content_type=content_type)
            
            # Make the blob publicly accessible
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Firebase upload failed: {e}")
            return None 
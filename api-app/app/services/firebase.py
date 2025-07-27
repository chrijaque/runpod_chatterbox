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
    
    def __init__(self, credentials_file: str, bucket_name: str):
        self.credentials_file = credentials_file
        self.bucket_name = bucket_name
        self.bucket = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_file)
                logger.info(f"🔍 Initializing Firebase with storageBucket: {self.bucket_name}")
                # Don't specify storageBucket in the config, let it use the default
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firebase initialized successfully")
            
            # Try to get the bucket
            try:
                logger.info(f"🔍 Getting bucket with name: {self.bucket_name}")
                self.bucket = storage.bucket(self.bucket_name)
                logger.info(f"🔍 Bucket object created: {self.bucket.name}")
                
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
            client = gcs_storage.Client.from_service_account_json(self.credentials_file)
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

    def upload_runpod_voice_sample(self, voice_id: str, sample_filename: str, language: str = "en", is_kids_voice: bool = False) -> Optional[str]:
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
        if is_kids_voice:
            firebase_path = f"audio/voices/{language}/kids/samples/{voice_id}_{sample_filename}"
        else:
            firebase_path = f"audio/voices/{language}/samples/{voice_id}_{sample_filename}"
        
        return self.upload_from_runpod_directory(runpod_path, firebase_path)

    def upload_runpod_tts_generation(self, generation_id: str, tts_filename: str, language: str = "en", is_kids_voice: bool = False, story_type: str = "user") -> Optional[str]:
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
        firebase_path = f"audio/stories/{language}/{story_type}/{generation_id}_{tts_filename}"
        
        return self.upload_from_runpod_directory(runpod_path, firebase_path)

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

    def upload_voice_profile(self, voice_id: str, profile_filename: str, language: str = "en", is_kids_voice: bool = False) -> Optional[str]:
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
        if is_kids_voice:
            firebase_path = f"audio/voices/{language}/kids/profiles/{voice_id}_{profile_filename}"
        else:
            firebase_path = f"audio/voices/{language}/profiles/{voice_id}_{profile_filename}"
        
        return self.upload_from_runpod_directory(runpod_path, firebase_path)

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
            logger.info(f"📁 Found {len(blob_list)} blobs in Firebase with prefix: {prefix}")
            
            # Log first few blobs for debugging
            for i, blob in enumerate(blob_list[:5]):
                logger.info(f"   Blob {i+1}: {blob.name}")
            
            if len(blob_list) > 5:
                logger.info(f"   ... and {len(blob_list) - 5} more blobs")
            
            voices = {}
            
            for blob in blob_list:
                if blob.name.endswith('/'):
                    continue
                    
                # Extract voice_id from path
                # Path format: audio/voices/en/samples/voice_name_sample_timestamp.wav
                # or: audio/voices/en/profiles/voice_name.npy
                path_parts = blob.name.split('/')
                
                # Find the voice_id from the filename
                filename = path_parts[-1]
                
                # Handle different file types
                if filename.endswith('.npy'):
                    # Profile file: voice_name.npy
                    voice_id = filename.replace('.npy', '')
                    logger.debug(f"   📄 Profile file: {filename} -> voice_id: {voice_id}")
                elif '_sample_' in filename:
                    # Sample file: voice_name_sample_timestamp.wav
                    voice_id = filename.split('_sample_')[0]
                    logger.debug(f"   🎵 Sample file: {filename} -> voice_id: {voice_id}")
                elif filename.startswith('recording_'):
                    # Recording file: recording_1_timestamp.wav
                    # Extract voice_id from the path structure
                    if len(path_parts) >= 4:
                        # Path: audio/voices/en/recorded/voice_name_recording_1_timestamp.wav
                        voice_id = path_parts[-1].split('_recording_')[0]
                        logger.debug(f"   🎤 Recording file: {filename} -> voice_id: {voice_id}")
                    else:
                        logger.debug(f"   ⚠️ Skipping recording file with insufficient path parts: {blob.name}")
                        continue
                else:
                    # Skip unknown file types
                    logger.debug(f"   ⚠️ Skipping unknown file type: {filename}")
                    continue
                
                if voice_id not in voices:
                    voices[voice_id] = {
                        "voice_id": voice_id,
                        "name": voice_id.replace('voice_', '').replace('_', ' ').title(),
                        "recorded": [],
                        "samples": [],
                        "profiles": [],
                        "created_date": None
                    }
                
                url = blob.public_url
                if "/recorded/" in blob.name:
                    voices[voice_id]["recorded"].append(url)
                elif "/samples/" in blob.name:
                    voices[voice_id]["samples"].append(url)
                elif "/profiles/" in blob.name:
                    voices[voice_id]["profiles"].append(url)
                
                # Try to extract creation time from filename
                if '_sample_' in filename:
                    try:
                        timestamp_part = filename.split('_sample_')[-1].replace('.wav', '')
                        # Parse timestamp like 20250727_153420
                        if len(timestamp_part) >= 15:
                            from datetime import datetime
                            created_date = datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')
                            voices[voice_id]["created_date"] = created_date.timestamp()
                    except Exception as e:
                        logger.debug(f"Could not parse timestamp from {filename}: {e}")
            
            result = list(voices.values())
            logger.info(f"✅ Processed {len(result)} voices from Firebase")
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
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            stories = {}
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                    
                # Extract generation_id from filename
                filename = blob.name.split('/')[-1]
                
                # Handle different story file naming patterns
                if filename.endswith('.wav'):
                    # Extract generation_id from filename
                    # Pattern: gen_20250727_153420.wav or similar
                    generation_id = filename.replace('.wav', '')
                    
                    if generation_id not in stories:
                        stories[generation_id] = {
                            "generation_id": generation_id,
                            "voice_id": "unknown",  # We'll need to extract this from metadata
                            "voice_name": "Unknown Voice",
                            "audio_files": [],
                            "created_date": None
                        }
                    
                    url = blob.public_url
                    stories[generation_id]["audio_files"].append(url)
                    
                    # Try to extract creation time from filename
                    if '_' in filename:
                        try:
                            timestamp_part = filename.split('_')[-1].replace('.wav', '')
                            # Parse timestamp like 20250727_153420
                            if len(timestamp_part) >= 15:
                                from datetime import datetime
                                created_date = datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')
                                stories[generation_id]["created_date"] = created_date.timestamp()
                        except Exception as e:
                            logger.debug(f"Could not parse timestamp from {filename}: {e}")
            
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
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
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.bucket_name
                })
                logger.info("✅ Firebase initialized successfully")
            
            self.bucket = storage.bucket()
            logger.info(f"✅ Connected to Firebase bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {e}")
            self.bucket = None
    
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
            return []
        
        try:
            # Build prefix based on language and kids voice
            if is_kids_voice:
                prefix = f"audio/voices/{language}/kids/"
            else:
                prefix = f"audio/voices/{language}/"
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            voices = {}
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                    
                # Extract voice_id from filename
                filename = blob.name.split('/')[-1]
                if '_' in filename:
                    voice_id = filename.split('_')[0]
                    
                    if voice_id not in voices:
                        voices[voice_id] = {
                            "voice_id": voice_id,
                            "recorded": [],
                            "samples": [],
                            "profiles": []
                        }
                    
                    url = blob.public_url
                    if "/recorded/" in blob.name:
                        voices[voice_id]["recorded"].append(url)
                    elif "/samples/" in blob.name:
                        voices[voice_id]["samples"].append(url)
                    elif "/profiles/" in blob.name:
                        voices[voice_id]["profiles"].append(url)
            
            return list(voices.values())
            
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
                if '_' in filename:
                    generation_id = filename.split('_')[0]
                    
                    if generation_id not in stories:
                        stories[generation_id] = {
                            "generation_id": generation_id,
                            "audio_files": [],
                            "created_at": None
                        }
                    
                    url = blob.public_url
                    stories[generation_id]["audio_files"].append(url)
                    
                    # Try to extract creation time from filename
                    if '_' in filename:
                        try:
                            timestamp_part = filename.split('_')[-1].replace('.wav', '')
                            if len(timestamp_part) >= 14:  # YYYYMMDD_HHMMSS format
                                stories[generation_id]["created_at"] = timestamp_part
                        except:
                            pass
            
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
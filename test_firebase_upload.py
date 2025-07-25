#!/usr/bin/env python3
"""
Test script to verify Firebase bucket creation and directory structure
without using RunPod GPU resources.
"""

import os
import sys
import base64
import logging
from pathlib import Path

# Add the api-app directory to the path
sys.path.append('api-app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_audio_data():
    """Create a small test audio file (just a placeholder)"""
    # Create a minimal WAV file header (44 bytes)
    wav_header = (
        b'RIFF' +           # Chunk ID
        b'\x24\x00\x00\x00' +  # Chunk size (36 bytes)
        b'WAVE' +           # Format
        b'fmt ' +           # Subchunk1 ID
        b'\x10\x00\x00\x00' +  # Subchunk1 size (16 bytes)
        b'\x01\x00' +       # Audio format (PCM)
        b'\x01\x00' +       # Number of channels (1)
        b'\x44\xAC\x00\x00' +  # Sample rate (44100)
        b'\x88\x58\x01\x00' +  # Byte rate
        b'\x02\x00' +       # Block align
        b'\x10\x00' +       # Bits per sample (16)
        b'data' +           # Subchunk2 ID
        b'\x00\x00\x00\x00'   # Subchunk2 size (0 bytes for silence)
    )
    
    # Convert to base64
    return base64.b64encode(wav_header).decode('utf-8')

def create_test_profile_data():
    """Create a small test profile file (just a placeholder)"""
    # Create a minimal numpy array representation
    import numpy as np
    
    # Create a small test array
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Save to bytes
    import io
    buffer = io.BytesIO()
    np.save(buffer, test_array)
    buffer.seek(0)
    
    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_firebase_upload():
    """Test Firebase upload functionality"""
    try:
        logger.info("🚀 Starting Firebase upload test...")
        
        # Import Firebase service
        from app.services.firebase import FirebaseService
        from app.config import settings
        
        logger.info("📋 Configuration check:")
        logger.info(f"   Firebase credentials: {settings.FIREBASE_CREDENTIALS_FILE}")
        logger.info(f"   Firebase bucket: {settings.FIREBASE_STORAGE_BUCKET}")
        logger.info(f"   Credentials exist: {Path(settings.FIREBASE_CREDENTIALS_FILE).exists()}")
        
        # Initialize Firebase service
        logger.info("🔧 Initializing Firebase service...")
        firebase_service = FirebaseService(
            credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
            bucket_name=settings.get_firebase_bucket_name()
        )
        
        if not firebase_service.is_connected():
            logger.error("❌ Firebase service not connected")
            return False
        
        logger.info("✅ Firebase service initialized successfully")
        
        # Create test data
        logger.info("📝 Creating test data...")
        test_audio_base64 = create_test_audio_data()
        test_profile_base64 = create_test_profile_data()
        
        logger.info(f"   Test audio size: {len(test_audio_base64)} chars")
        logger.info(f"   Test profile size: {len(test_profile_base64)} chars")
        
        # Test 1: Upload user recording
        logger.info("\n🔍 Test 1: Upload user recording...")
        recording_path = "audio/voices/en/recorded/test_voice_001_recording_1.wav"
        recording_url = firebase_service.upload_base64_audio(test_audio_base64, recording_path)
        
        if recording_url:
            logger.info(f"✅ User recording uploaded: {recording_url}")
        else:
            logger.error("❌ User recording upload failed")
            return False
        
        # Test 2: Upload voice sample
        logger.info("\n🔍 Test 2: Upload voice sample...")
        sample_path = "audio/voices/en/samples/test_voice_001_sample.wav"
        sample_url = firebase_service.upload_base64_audio(test_audio_base64, sample_path)
        
        if sample_url:
            logger.info(f"✅ Voice sample uploaded: {sample_url}")
        else:
            logger.error("❌ Voice sample upload failed")
            return False
        
        # Test 3: Upload voice profile
        logger.info("\n🔍 Test 3: Upload voice profile...")
        profile_path = "audio/voices/en/profiles/test_voice_001.npy"
        profile_url = firebase_service.upload_base64_profile(test_profile_base64, profile_path)
        
        if profile_url:
            logger.info(f"✅ Voice profile uploaded: {profile_url}")
        else:
            logger.error("❌ Voice profile upload failed")
            return False
        
        # Test 4: Test kids voice structure
        logger.info("\n🔍 Test 4: Test kids voice structure...")
        kids_recording_path = "audio/voices/en/kids/recorded/test_kids_voice_001_recording_1.wav"
        kids_recording_url = firebase_service.upload_base64_audio(test_audio_base64, kids_recording_path)
        
        if kids_recording_url:
            logger.info(f"✅ Kids voice recording uploaded: {kids_recording_url}")
        else:
            logger.error("❌ Kids voice recording upload failed")
            return False
        
        # Test 5: Test different language
        logger.info("\n🔍 Test 5: Test different language...")
        spanish_recording_path = "audio/voices/es/recorded/test_spanish_voice_001_recording_1.wav"
        spanish_recording_url = firebase_service.upload_base64_audio(test_audio_base64, spanish_recording_path)
        
        if spanish_recording_url:
            logger.info(f"✅ Spanish voice recording uploaded: {spanish_recording_url}")
        else:
            logger.error("❌ Spanish voice recording upload failed")
            return False
        
        # Test 6: Test TTS stories structure
        logger.info("\n🔍 Test 6: Test TTS stories structure...")
        story_path = "audio/stories/en/user/test_story_001.wav"
        story_url = firebase_service.upload_base64_audio(test_audio_base64, story_path)
        
        if story_url:
            logger.info(f"✅ TTS story uploaded: {story_url}")
        else:
            logger.error("❌ TTS story upload failed")
            return False
        
        logger.info("\n🎉 All Firebase upload tests passed!")
        logger.info("✅ Bucket creation: Working")
        logger.info("✅ Directory structure: Working")
        logger.info("✅ File uploads: Working")
        logger.info("✅ Multiple languages: Working")
        logger.info("✅ Kids voices: Working")
        logger.info("✅ TTS stories: Working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_firebase_listing():
    """Test Firebase listing functionality"""
    try:
        logger.info("\n🔍 Testing Firebase listing functionality...")
        
        from app.services.firebase import FirebaseService
        from app.config import settings
        
        firebase_service = FirebaseService(
            credentials_file=settings.FIREBASE_CREDENTIALS_FILE,
            bucket_name=settings.get_firebase_bucket_name()
        )
        
        # Test listing voices by language
        logger.info("📋 Testing voice listing...")
        voices = firebase_service.list_voices_by_language("en", is_kids_voice=False)
        logger.info(f"   Found {len(voices)} English voices")
        
        kids_voices = firebase_service.list_voices_by_language("en", is_kids_voice=True)
        logger.info(f"   Found {len(kids_voices)} English kids voices")
        
        spanish_voices = firebase_service.list_voices_by_language("es", is_kids_voice=False)
        logger.info(f"   Found {len(spanish_voices)} Spanish voices")
        
        # Test listing stories
        logger.info("📋 Testing story listing...")
        stories = firebase_service.list_stories_by_language("en", story_type="user")
        logger.info(f"   Found {len(stories)} English user stories")
        
        logger.info("✅ Firebase listing tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Listing test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 ===== FIREBASE UPLOAD TEST SCRIPT =====")
    
    # Check if we're in the right directory
    if not Path("api-app").exists():
        logger.error("❌ Please run this script from the project root directory")
        logger.error("❌ Expected to find 'api-app' directory")
        return False
    
    # Check if Firebase credentials exist
    if not Path("api-app/firebase_creds.json").exists():
        logger.error("❌ Firebase credentials not found")
        logger.error("❌ Expected: api-app/firebase_creds.json")
        return False
    
    # Run tests
    upload_success = test_firebase_upload()
    listing_success = test_firebase_listing()
    
    if upload_success and listing_success:
        logger.info("\n🎉 ===== ALL TESTS PASSED =====")
        logger.info("✅ Firebase bucket creation: Working")
        logger.info("✅ Directory structure creation: Working")
        logger.info("✅ File uploads: Working")
        logger.info("✅ File listing: Working")
        logger.info("\n🚀 Your Firebase setup is ready for production!")
        return True
    else:
        logger.error("\n❌ ===== SOME TESTS FAILED =====")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
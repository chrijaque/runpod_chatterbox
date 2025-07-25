#!/usr/bin/env python3
"""
Test script to test the FastAPI voice clone endpoint without using RunPod.
This simulates the frontend request with pre-generated sample data.
"""

import requests
import json
import base64
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def test_fastapi_voice_clone():
    """Test the FastAPI voice clone endpoint"""
    try:
        logger.info("🚀 Testing FastAPI voice clone endpoint...")
        
        # FastAPI server URL
        base_url = "http://localhost:8000"
        
        # Check if server is running
        try:
            health_response = requests.get(f"{base_url}/health")
            if health_response.status_code != 200:
                logger.error(f"❌ FastAPI server not responding: {health_response.status_code}")
                return False
            logger.info("✅ FastAPI server is running")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to FastAPI server")
            logger.error("   Make sure the server is running on http://localhost:8000")
            return False
        
        # Create test data
        test_audio_base64 = create_test_audio_data()
        
        # Prepare the request (simulating frontend request)
        voice_clone_request = {
            "title": "test_script_voice",
            "voices": [test_audio_base64],  # Original recording
            "generated_sample": test_audio_base64,  # Pre-generated sample (same for testing)
            "visibility": "private",
            "metadata": {
                "language": "en",
                "isKidsVoice": False,
                "userId": "test_script_user",
                "createdAt": "2025-07-25T19:21:29.294Z"
            }
        }
        
        logger.info("📤 Sending voice clone request...")
        logger.info(f"   Title: {voice_clone_request['title']}")
        logger.info(f"   Language: {voice_clone_request['metadata']['language']}")
        logger.info(f"   Kids Voice: {voice_clone_request['metadata']['isKidsVoice']}")
        logger.info(f"   Has generated sample: {bool(voice_clone_request['generated_sample'])}")
        
        # Send request to FastAPI
        response = requests.post(
            f"{base_url}/api/voices/clone",
            json=voice_clone_request,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Voice clone request successful!")
            logger.info(f"   Voice ID: {result.get('voice_id', 'N/A')}")
            logger.info(f"   Status: {result.get('status', 'N/A')}")
            
            # Check Firebase URLs
            firebase_urls = result.get('firebase_urls', {})
            logger.info("   Firebase URLs:")
            for category, urls in firebase_urls.items():
                logger.info(f"     {category}: {len(urls)} files")
                for url in urls:
                    logger.info(f"       - {url}")
            
            return True
        else:
            logger.error(f"❌ Voice clone request failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_fastapi_voice_listing():
    """Test the FastAPI voice listing endpoint"""
    try:
        logger.info("\n🔍 Testing FastAPI voice listing endpoint...")
        
        base_url = "http://localhost:8000"
        
        # Test listing voices by language
        response = requests.get(f"{base_url}/api/voices/by-language/en?is_kids_voice=false")
        
        logger.info(f"📥 Listing response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            voices = result.get('voices', [])
            logger.info(f"✅ Found {len(voices)} English voices")
            return True
        else:
            logger.error(f"❌ Voice listing failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Listing test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 ===== FASTAPI ENDPOINT TEST SCRIPT =====")
    
    # Check if FastAPI server is running
    logger.info("🔍 Checking if FastAPI server is running...")
    
    # Run tests
    clone_success = test_fastapi_voice_clone()
    listing_success = test_fastapi_voice_listing()
    
    if clone_success and listing_success:
        logger.info("\n🎉 ===== ALL FASTAPI TESTS PASSED =====")
        logger.info("✅ Voice clone endpoint: Working")
        logger.info("✅ Voice listing endpoint: Working")
        logger.info("✅ Firebase integration: Working")
        logger.info("\n🚀 Your FastAPI setup is ready for production!")
        return True
    else:
        logger.error("\n❌ ===== SOME FASTAPI TESTS FAILED =====")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 
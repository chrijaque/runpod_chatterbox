#!/usr/bin/env python3
"""
Test script for unified ChatterboxTTS + Higgs Audio setup
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path

def test_api_endpoints():
    """Test the API endpoints with both models"""
    
    # API configuration
    base_url = "http://localhost:8000"
    
    # Test data
    test_audio_base64 = "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
    
    print("🧪 ===== TESTING UNIFIED API SETUP =====")
    
    # Test 1: Voice Cloning with ChatterboxTTS
    print("\n🔍 Test 1: Voice Cloning (ChatterboxTTS)")
    try:
        response = requests.post(f"{base_url}/api/voices/clone", json={
            "name": "test_voice_chatterbox",
            "audio_data": test_audio_base64,
            "audio_format": "wav",
            "language": "en",
            "is_kids_voice": False,
            "model_type": "chatterbox"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ChatterboxTTS voice cloning: {data.get('status')}")
            print(f"   Model used: {data.get('model_type', 'chatterbox')}")
        else:
            print(f"❌ ChatterboxTTS voice cloning failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing ChatterboxTTS: {e}")
    
    # Test 2: Voice Cloning with Higgs Audio
    print("\n🔍 Test 2: Voice Cloning (Higgs Audio)")
    try:
        response = requests.post(f"{base_url}/api/voices/clone", json={
            "name": "test_voice_higgs",
            "audio_data": test_audio_base64,
            "audio_format": "wav",
            "language": "en",
            "is_kids_voice": False,
            "model_type": "higgs"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Higgs Audio voice cloning: {data.get('status')}")
            print(f"   Model used: {data.get('model_type', 'higgs')}")
        else:
            print(f"❌ Higgs Audio voice cloning failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing Higgs Audio: {e}")
    
    # Test 3: TTS Generation with ChatterboxTTS
    print("\n🔍 Test 3: TTS Generation (ChatterboxTTS)")
    try:
        response = requests.post(f"{base_url}/api/tts/generate", json={
            "voice_id": "test_voice_chatterbox",
            "text": "Hello, this is a test of the unified TTS system.",
            "profile_base64": "dGVzdF9wcm9maWxlX2RhdGE=",  # base64 "test_profile_data"
            "language": "en",
            "story_type": "user",
            "is_kids_voice": False,
            "model_type": "chatterbox"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ChatterboxTTS TTS generation: {data.get('status')}")
            print(f"   Model used: {data.get('model_type', 'chatterbox')}")
        else:
            print(f"❌ ChatterboxTTS TTS generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing ChatterboxTTS TTS: {e}")
    
    # Test 4: TTS Generation with Higgs Audio
    print("\n🔍 Test 4: TTS Generation (Higgs Audio)")
    try:
        response = requests.post(f"{base_url}/api/tts/generate", json={
            "voice_id": "test_voice_higgs",
            "text": "Hello, this is a test of the unified TTS system with Higgs Audio.",
            "profile_base64": "dGVzdF9wcm9maWxlX2RhdGE=",  # base64 "test_profile_data"
            "language": "en",
            "story_type": "user",
            "is_kids_voice": False,
            "model_type": "higgs"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Higgs Audio TTS generation: {data.get('status')}")
            print(f"   Model used: {data.get('model_type', 'higgs')}")
        else:
            print(f"❌ Higgs Audio TTS generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error testing Higgs Audio TTS: {e}")

def test_frontend_integration():
    """Test frontend model toggle functionality"""
    
    print("\n🧪 ===== TESTING FRONTEND INTEGRATION =====")
    
    # Check if frontend files exist
    frontend_files = [
        "frontend/src/components/ModelToggle.tsx",
        "frontend/src/app/page.tsx",
        "frontend/src/app/tts/page.tsx"
    ]
    
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Check for model toggle component
    model_toggle_path = "frontend/src/components/ModelToggle.tsx"
    if Path(model_toggle_path).exists():
        with open(model_toggle_path, 'r') as f:
            content = f.read()
            if 'ModelType' in content and 'chatterbox' in content and 'higgs' in content:
                print("✅ ModelToggle component properly configured")
            else:
                print("❌ ModelToggle component missing required elements")

def test_docker_setup():
    """Test Docker setup"""
    
    print("\n🧪 ===== TESTING DOCKER SETUP =====")
    
    # Check Docker files
    docker_files = [
        "dockerfiles/unified/Dockerfile.vc",
        "dockerfiles/unified/Dockerfile.tts",
        "requirements/unified.txt"
    ]
    
    for file_path in docker_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
    
    # Check unified handlers
    handler_files = [
        "handlers/unified_vc_handler.py",
        "handlers/unified_tts_handler.py"
    ]
    
    for file_path in handler_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")

def main():
    """Main test function"""
    
    print("🚀 ===== UNIFIED SETUP TESTING =====")
    print("This script tests the unified ChatterboxTTS + Higgs Audio setup")
    
    # Test Docker setup
    test_docker_setup()
    
    # Test frontend integration
    test_frontend_integration()
    
    # Test API endpoints (if server is running)
    print("\n💡 To test API endpoints, start the FastAPI server:")
    print("   cd api-app && python -m uvicorn app.main:app --reload")
    print("   Then run: python test_unified_setup.py")
    
    # Uncomment to test API endpoints when server is running
    # test_api_endpoints()
    
    print("\n✅ ===== TESTING COMPLETED =====")
    print("\n📋 Next Steps:")
    print("1. Deploy unified Docker images to RunPod")
    print("2. Test with real audio files")
    print("3. Compare model performance")
    print("4. Update documentation")

if __name__ == "__main__":
    main() 
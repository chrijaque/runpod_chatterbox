#!/usr/bin/env python3
"""
Test script for TTS file access methods
"""

import json
import os
from pathlib import Path

def test_file_listing():
    """Test the file listing functionality"""
    print("🧪 Testing file listing functionality...")
    
    # Simulate the list_available_files function
    TTS_GENERATED_DIR = Path("/tts_generated")
    
    try:
        if TTS_GENERATED_DIR.exists():
            files = list(TTS_GENERATED_DIR.glob("*.wav"))
            print(f"📂 Found {len(files)} TTS files:")
            
            for file in files:
                file_size = file.stat().st_size
                print(f"  📄 {file.name}: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            return {
                "status": "success",
                "files": [{"name": f.name, "size_bytes": f.stat().st_size, "size_mb": f.stat().st_size/1024/1024} for f in files]
            }
        else:
            print(f"❌ TTS directory not found: {TTS_GENERATED_DIR}")
            return {"status": "error", "message": f"TTS directory not found: {TTS_GENERATED_DIR}"}
            
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return {"status": "error", "message": f"Failed to list files: {e}"}

def test_file_download(file_path):
    """Test the file download functionality"""
    print(f"🧪 Testing file download: {file_path}")
    
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ File exists: {file_path}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            # Read file and convert to base64
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            import base64
            audio_base64 = base64.b64encode(file_data).decode('utf-8')
            print(f"✅ File read successfully: {len(audio_base64):,} base64 chars")
            
            return {
                "status": "success",
                "method": "direct_file_access",
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": file_size/1024/1024,
                "audio_base64": audio_base64[:100] + "...",  # Truncate for display
                "message": "File downloaded via direct file system access"
            }
        else:
            print(f"❌ File not found: {file_path}")
            return {"status": "error", "message": f"File not found: {file_path}"}
            
    except Exception as e:
        print(f"❌ Direct file access failed: {e}")
        return {"status": "error", "message": f"Direct file access failed: {e}"}

def simulate_tts_response():
    """Simulate a TTS response with file path"""
    print("🧪 Simulating TTS response with file path...")
    
    # Simulate the response you'd get from a large TTS file
    response = {
        "status": "success",
        "audio_base64": None,
        "file_path": "/tts_generated/tts_voice_chrisrepo1_20250724_144927.wav",
        "file_size_mb": 31.9,
        "metadata": {
            "voice_id": "voice_chrisrepo1",
            "generation_time": 407.47,
            "response_type": "file_path_only"
        }
    }
    
    print("📤 TTS Response:")
    print(json.dumps(response, indent=2))
    
    return response

if __name__ == "__main__":
    print("🧪 Testing TTS file access methods...")
    print("=" * 60)
    
    # Test 1: File listing
    print("\n1️⃣ Testing file listing:")
    list_result = test_file_listing()
    print(f"Result: {list_result['status']}")
    
    # Test 2: Simulate TTS response
    print("\n2️⃣ Simulating TTS response:")
    tts_response = simulate_tts_response()
    
    # Test 3: File download (if files exist)
    if list_result['status'] == 'success' and list_result['files']:
        print("\n3️⃣ Testing file download:")
        first_file = list_result['files'][0]
        file_path = f"/tts_generated/{first_file['name']}"
        download_result = test_file_download(file_path)
        print(f"Download result: {download_result['status']}")
    else:
        print("\n3️⃣ Skipping file download (no files found)")
    
    print("\n" + "=" * 60)
    print("🎉 File access tests completed!")
    print("\n📋 Usage Instructions:")
    print("1. For TTS generation: Send normal TTS request")
    print("2. For file listing: Send {'action': 'list_files'}")
    print("3. For file download: Send {'action': 'download_file', 'file_path': '/path/to/file.wav'}") 
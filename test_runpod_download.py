#!/usr/bin/env python3
"""
Test script to download TTS files from RunPod during testing phase
"""

import json
import time
import base64
import requests
from pathlib import Path

def download_tts_file(runpod_url, file_path, local_dir="./tts_generated"):
    """Download TTS file from RunPod and save locally"""
    print(f"📥 Downloading: {file_path}")
    
    # Create local directory
    Path(local_dir).mkdir(exist_ok=True)
    
    # Prepare download request
    download_payload = {
        "input": {
            "action": "download_file",
            "file_path": file_path
        }
    }
    
    try:
        # Send download request
        print("📤 Sending download request...")
        response = requests.post(runpod_url, json=download_payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            # Extract file info
            audio_base64 = result.get("audio_base64")
            file_size_mb = result.get("file_size_mb", 0)
            
            if audio_base64:
                # Convert base64 to file
                audio_data = base64.b64decode(audio_base64)
                
                # Extract filename from path
                filename = Path(file_path).name
                local_file = Path(local_dir) / filename
                
                # Save locally
                with open(local_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"✅ Downloaded successfully!")
                print(f"📁 Local file: {local_file}")
                print(f"📊 File size: {file_size_mb:.1f} MB")
                print(f"📊 Data size: {len(audio_data):,} bytes")
                
                return str(local_file)
            else:
                print("❌ No audio data in response")
                return None
        else:
            print(f"❌ Download failed: {result.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Download timed out - container may have been destroyed")
        return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def list_tts_files(runpod_url):
    """List available TTS files on RunPod"""
    print("📂 Listing available TTS files...")
    
    list_payload = {
        "input": {
            "action": "list_files"
        }
    }
    
    try:
        response = requests.post(runpod_url, json=list_payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            files = result.get("files", [])
            if files:
                print(f"📂 Found {len(files)} files:")
                for file in files:
                    print(f"  📄 {file['name']}: {file['size_mb']:.1f} MB")
                return files
            else:
                print("📂 No files found")
                return []
        else:
            print(f"❌ List failed: {result.get('message', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"❌ List error: {e}")
        return []

def test_tts_generation_and_download(runpod_url, text, voice_id, profile_base64):
    """Test complete TTS generation and download workflow"""
    print("🧪 Testing TTS generation and download workflow...")
    print("=" * 60)
    
    # Step 1: Generate TTS
    print("1️⃣ Generating TTS...")
    tts_payload = {
        "input": {
            "text": text,
            "voice_id": voice_id,
            "profile_base64": profile_base64
        }
    }
    
    try:
        response = requests.post(runpod_url, json=tts_payload, timeout=300)  # 5 min timeout
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            file_path = result.get("file_path")
            response_type = result.get("metadata", {}).get("response_type")
            
            print(f"✅ TTS generation successful!")
            print(f"📁 File path: {file_path}")
            print(f"📊 Response type: {response_type}")
            
            if response_type == "file_path_only" and file_path:
                # Step 2: Immediately download the file
                print("\n2️⃣ Downloading file immediately...")
                local_file = download_tts_file(runpod_url, file_path)
                
                if local_file:
                    print(f"\n🎉 Success! File downloaded to: {local_file}")
                    return local_file
                else:
                    print("\n❌ Download failed - container may have been destroyed")
                    return None
            else:
                print("✅ Audio data included in response - no download needed")
                return "audio_in_response"
        else:
            print(f"❌ TTS generation failed: {result.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        return None

if __name__ == "__main__":
    print("🧪 RunPod TTS Download Testing")
    print("=" * 50)
    
    # Configuration
    RUNPOD_URL = "YOUR_RUNPOD_ENDPOINT_HERE"  # Replace with your endpoint
    
    print("📋 Usage Instructions:")
    print("1. Replace RUNPOD_URL with your actual RunPod endpoint")
    print("2. Run TTS generation")
    print("3. Immediately download the file before container cleanup")
    print("4. Files will be saved in ./tts_generated/ directory")
    
    print("\n🔧 Test Functions Available:")
    print("- list_tts_files(runpod_url)")
    print("- download_tts_file(runpod_url, file_path)")
    print("- test_tts_generation_and_download(runpod_url, text, voice_id, profile_base64)")
    
    print("\n⏰ Important:")
    print("- Download files IMMEDIATELY after generation")
    print("- Container stays alive for 30s-2min after job completion")
    print("- Files are permanently lost after container cleanup") 
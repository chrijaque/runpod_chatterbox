#!/usr/bin/env python3
"""
Complete RunPod to Local TTS Workflow
"""

import json
import time
import requests
from pathlib import Path

def runpod_to_local_workflow(runpod_url, text, voice_id, profile_base64):
    """
    Complete workflow: RunPod TTS Generation → Immediate Local Download
    """
    print("🚀 RunPod to Local TTS Workflow")
    print("=" * 50)
    
    # Step 1: Generate TTS in RunPod
    print("1️⃣ Generating TTS in RunPod...")
    tts_payload = {
        "input": {
            "text": text,
            "voice_id": voice_id,
            "profile_base64": profile_base64
        }
    }
    
    try:
        response = requests.post(runpod_url, json=tts_payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success":
            file_path = result.get("file_path")
            response_type = result.get("metadata", {}).get("response_type")
            
            print(f"✅ TTS generation successful!")
            print(f"📁 RunPod file: {file_path}")
            print(f"📊 Response type: {response_type}")
            
            # Step 2: Immediately download to local
            if response_type == "file_path_only" and file_path:
                print("\n2️⃣ Downloading to local ./tts_generated/ directory...")
                
                download_payload = {
                    "input": {
                        "action": "download_file",
                        "file_path": file_path
                    }
                }
                
                download_response = requests.post(runpod_url, json=download_payload, timeout=60)
                download_response.raise_for_status()
                download_result = download_response.json()
                
                if download_result.get("status") == "success":
                    audio_base64 = download_result.get("audio_base64")
                    if audio_base64:
                        # Save to local directory
                        import base64
                        audio_data = base64.b64decode(audio_base64)
                        
                        local_dir = Path("./tts_generated")
                        local_dir.mkdir(exist_ok=True)
                        
                        filename = Path(file_path).name
                        local_file = local_dir / filename
                        
                        with open(local_file, 'wb') as f:
                            f.write(audio_data)
                        
                        print(f"✅ File downloaded successfully!")
                        print(f"📁 Local file: {local_file}")
                        print(f"📊 File size: {len(audio_data):,} bytes ({len(audio_data)/1024/1024:.1f} MB)")
                        
                        return str(local_file)
                    else:
                        print("❌ No audio data in download response")
                        return None
                else:
                    print(f"❌ Download failed: {download_result.get('message')}")
                    return None
            else:
                print("✅ Audio data included in response - no download needed")
                return "audio_in_response"
        else:
            print(f"❌ TTS generation failed: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return None

def check_local_files():
    """Check what files are in the local tts_generated directory"""
    print("\n📂 Checking local ./tts_generated/ directory...")
    
    local_dir = Path("./tts_generated")
    
    if not local_dir.exists():
        print("❌ Local directory doesn't exist yet")
        return []
    
    files = list(local_dir.glob("*.wav"))
    
    if not files:
        print("📂 Directory exists but no files found")
        return []
    
    print(f"✅ Found {len(files)} local TTS files:")
    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        file_size = file.stat().st_size
        print(f"  📄 {file.name}: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    return files

if __name__ == "__main__":
    print("🧪 RunPod to Local TTS Workflow")
    print("=" * 40)
    
    # Configuration
    RUNPOD_URL = "YOUR_RUNPOD_ENDPOINT_HERE"
    
    print("📋 Complete Workflow:")
    print("1. RunPod generates TTS file")
    print("2. File saved in RunPod container")
    print("3. Immediately download to local ./tts_generated/")
    print("4. File persists locally after container cleanup")
    
    print("\n🔧 Usage:")
    print("runpod_to_local_workflow(runpod_url, text, voice_id, profile_base64)")
    print("check_local_files()")
    
    print("\n⏰ Key Points:")
    print("- Download happens IMMEDIATELY after generation")
    print("- Files saved to ./tts_generated/ directory")
    print("- Local files persist after RunPod container cleanup")
    print("- Perfect for testing and development") 
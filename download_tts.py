#!/usr/bin/env python3
"""
Backup TTS Download Script
Downloads TTS files from RunPod if automatic download fails
"""

import json
import base64
import requests
from pathlib import Path
from datetime import datetime

def download_tts_file(runpod_url, file_path, local_dir="./tts_generated"):
    """Download TTS file from RunPod to local directory"""
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
        print("📤 Sending download request to RunPod...")
        response = requests.post(runpod_url, json=download_payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            audio_base64 = result.get("audio_base64")
            file_size_mb = result.get("file_size_mb", 0)
            
            if audio_base64:
                # Convert base64 to file
                audio_data = base64.b64decode(audio_base64)
                
                # Extract filename and save locally
                filename = Path(file_path).name
                local_file = Path(local_dir) / filename
                
                with open(local_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"✅ Download successful!")
                print(f"📁 Local file: {local_file}")
                print(f"📊 File size: {file_size_mb:.1f} MB")
                print(f"📊 Data size: {len(audio_data):,} bytes")
                
                return str(local_file)
            else:
                print("❌ No audio data in response")
                return None
        else:
            print(f"❌ Download failed: {result.get('message')}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Download timed out - container may have been destroyed")
        return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

def list_runpod_files(runpod_url):
    """List available TTS files on RunPod"""
    print("📂 Listing available TTS files on RunPod...")
    
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
            print(f"❌ List failed: {result.get('message')}")
            return []
            
    except Exception as e:
        print(f"❌ List error: {e}")
        return []

def check_local_files():
    """Check what files are in the local directory"""
    print("\n📂 Checking local ./tts_generated/ directory...")
    
    local_dir = Path("./tts_generated")
    
    if not local_dir.exists():
        print("❌ Local directory doesn't exist")
        return []
    
    files = list(local_dir.glob("*.wav"))
    
    if not files:
        print("📂 No files found")
        return []
    
    print(f"✅ Found {len(files)} local TTS files:")
    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        file_size = file.stat().st_size
        file_time = file.stat().st_mtime
        file_date = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"  📄 {file.name}")
        print(f"     📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"     🕒 Created: {file_date}")
        print()
    
    return files

def main():
    """Main function"""
    print("🧪 Backup TTS Download Script")
    print("=" * 40)
    
    # Configuration - REPLACE WITH YOUR RUNPOD URL
    RUNPOD_URL = "YOUR_RUNPOD_ENDPOINT_HERE"
    
    print("📋 Usage:")
    print("1. Replace RUNPOD_URL with your actual RunPod endpoint")
    print("2. Run: download_tts_file(RUNPOD_URL, '/path/to/file.wav')")
    print("3. Or run: list_runpod_files(RUNPOD_URL) to see available files")
    
    print("\n🔧 Available functions:")
    print("- download_tts_file(runpod_url, file_path)")
    print("- list_runpod_files(runpod_url)")
    print("- check_local_files()")
    
    print("\n⏰ Use this if automatic download fails!")
    
    # Check current local files
    check_local_files()

if __name__ == "__main__":
    main() 
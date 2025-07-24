#!/usr/bin/env python3
"""
Test script to check RunPod file access
"""

import json
import requests

def test_runpod_file_listing():
    """Test file listing on RunPod"""
    print("🧪 Testing RunPod file listing...")
    
    # Simulate the request you'd send to RunPod
    payload = {
        "input": {
            "action": "list_files"
        }
    }
    
    print("📤 Request payload:")
    print(json.dumps(payload, indent=2))
    
    print("\n📋 To test this:")
    print("1. Send this payload to your RunPod TTS endpoint")
    print("2. Check the response for available files")
    print("3. Use the file_path to download specific files")
    
    return payload

def test_runpod_file_download(file_path):
    """Test file download on RunPod"""
    print(f"🧪 Testing RunPod file download: {file_path}")
    
    # Simulate the request you'd send to RunPod
    payload = {
        "input": {
            "action": "download_file",
            "file_path": file_path
        }
    }
    
    print("📤 Request payload:")
    print(json.dumps(payload, indent=2))
    
    print("\n📋 To test this:")
    print("1. Send this payload to your RunPod TTS endpoint")
    print("2. The response should contain the audio_base64 data")
    print("3. Convert base64 to audio file locally")
    
    return payload

if __name__ == "__main__":
    print("🧪 RunPod File Access Testing")
    print("=" * 50)
    
    # Test 1: File listing
    print("\n1️⃣ File Listing Test:")
    list_payload = test_runpod_file_listing()
    
    # Test 2: File download (using the file from your logs)
    print("\n2️⃣ File Download Test:")
    file_path = "/tts_generated/tts_voice_chrisrepo1_20250724_154322.wav"
    download_payload = test_runpod_file_download(file_path)
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Use these payloads with your RunPod API")
    print("2. Check if files are accessible")
    print("3. Download the audio files to your local machine") 
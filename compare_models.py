#!/usr/bin/env python3
"""
Model Comparison Script
Compare ChatterboxTTS vs Higgs Audio performance and quality
"""

import time
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any

def load_test_audio(file_path: str) -> str:
    """Load test audio file and convert to base64"""
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def test_voice_cloning(audio_base64: str, model_type: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test voice cloning with specified model"""
    
    print(f"🧪 Testing voice cloning with {model_type}...")
    
    start_time = time.time()
    
    response = requests.post(f"{api_url}/api/voices/clone", json={
        "name": f"test_voice_{model_type}",
        "audio_data": audio_base64,
        "audio_format": "wav",
        "language": "en",
        "is_kids_voice": False,
        "model_type": model_type
    })
    
    end_time = time.time()
    duration = end_time - start_time
    
    if response.status_code == 200:
        data = response.json()
        return {
            "status": "success",
            "model": model_type,
            "duration": duration,
            "voice_id": data.get("voice_id"),
            "profile_path": data.get("profile_path"),
            "sample_path": data.get("sample_path"),
            "metadata": data.get("metadata", {})
        }
    else:
        return {
            "status": "error",
            "model": model_type,
            "duration": duration,
            "error": response.text
        }

def test_tts_generation(voice_id: str, text: str, model_type: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test TTS generation with specified model"""
    
    print(f"🧪 Testing TTS generation with {model_type}...")
    
    start_time = time.time()
    
    response = requests.post(f"{api_url}/api/tts/generate", json={
        "voice_id": voice_id,
        "text": text,
        "profile_base64": "dGVzdF9wcm9maWxlX2RhdGE=",  # Placeholder
        "language": "en",
        "story_type": "user",
        "is_kids_voice": False,
        "model_type": model_type
    })
    
    end_time = time.time()
    duration = end_time - start_time
    
    if response.status_code == 200:
        data = response.json()
        return {
            "status": "success",
            "model": model_type,
            "duration": duration,
            "audio_path": data.get("audio_path"),
            "metadata": data.get("metadata", {})
        }
    else:
        return {
            "status": "error",
            "model": model_type,
            "duration": duration,
            "error": response.text
        }

def compare_models():
    """Compare both models"""
    
    print("🔍 ===== MODEL COMPARISON =====")
    
    # Test audio file (you can replace with your own)
    test_audio_path = "reference.wav"  # Use your test audio file
    
    if not Path(test_audio_path).exists():
        print(f"❌ Test audio file not found: {test_audio_path}")
        print("💡 Please provide a test audio file or update the path")
        return
    
    # Load test audio
    audio_base64 = load_test_audio(test_audio_path)
    print(f"✅ Loaded test audio: {len(audio_base64)} characters")
    
    # Test voice cloning
    print("\n🎤 ===== VOICE CLONING COMPARISON =====")
    
    chatterbox_vc = test_voice_cloning(audio_base64, "chatterbox")
    higgs_vc = test_voice_cloning(audio_base64, "higgs")
    
    print(f"\n📊 Voice Cloning Results:")
    print(f"ChatterboxTTS: {chatterbox_vc['duration']:.2f}s - {chatterbox_vc['status']}")
    print(f"Higgs Audio: {higgs_vc['duration']:.2f}s - {higgs_vc['status']}")
    
    # Test TTS generation
    print("\n🎵 ===== TTS GENERATION COMPARISON =====")
    
    test_text = "Hello, this is a test of the unified TTS system. We are comparing the performance and quality of different models."
    
    chatterbox_tts = test_tts_generation("test_voice_chatterbox", test_text, "chatterbox")
    higgs_tts = test_tts_generation("test_voice_higgs", test_text, "higgs")
    
    print(f"\n📊 TTS Generation Results:")
    print(f"ChatterboxTTS: {chatterbox_tts['duration']:.2f}s - {chatterbox_tts['status']}")
    print(f"Higgs Audio: {higgs_tts['duration']:.2f}s - {higgs_tts['status']}")
    
    # Summary
    print("\n📋 ===== SUMMARY =====")
    print(f"Voice Cloning Speed:")
    print(f"  ChatterboxTTS: {chatterbox_vc['duration']:.2f}s")
    print(f"  Higgs Audio: {higgs_vc['duration']:.2f}s")
    print(f"  Speed difference: {abs(chatterbox_vc['duration'] - higgs_vc['duration']):.2f}s")
    
    print(f"\nTTS Generation Speed:")
    print(f"  ChatterboxTTS: {chatterbox_tts['duration']:.2f}s")
    print(f"  Higgs Audio: {higgs_tts['duration']:.2f}s")
    print(f"  Speed difference: {abs(chatterbox_tts['duration'] - higgs_tts['duration']):.2f}s")
    
    # Recommendations
    print(f"\n💡 ===== RECOMMENDATIONS =====")
    if chatterbox_vc['duration'] < higgs_vc['duration']:
        print("✅ ChatterboxTTS is faster for voice cloning")
    else:
        print("✅ Higgs Audio is faster for voice cloning")
    
    if chatterbox_tts['duration'] < higgs_tts['duration']:
        print("✅ ChatterboxTTS is faster for TTS generation")
    else:
        print("✅ Higgs Audio is faster for TTS generation")
    
    print("\n🎯 Use Cases:")
    print("  ChatterboxTTS: Real-time applications, quick prototyping")
    print("  Higgs Audio: High-quality content, audiobooks, podcasts")

if __name__ == "__main__":
    compare_models() 
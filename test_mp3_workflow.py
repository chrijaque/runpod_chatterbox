#!/usr/bin/env python3
"""
Test script for MP3 workflow implementation.
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torchaudio

def test_mp3_conversion():
    """Test MP3 conversion utilities"""
    print("🧪 Testing MP3 conversion utilities...")
    
    try:
        # Import MP3 utilities
        from tts_handler import tensor_to_mp3_bytes, tensor_to_audiosegment
        
        # Create test audio tensor
        sample_rate = 22050
        duration = 2  # seconds
        samples = int(sample_rate * duration)
        
        # Generate test audio (sine wave)
        t = torch.linspace(0, duration, samples)
        audio_tensor = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        
        print(f"✅ Created test audio tensor: {audio_tensor.shape}")
        
        # Test tensor to MP3 conversion
        mp3_bytes = tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
        print(f"✅ MP3 conversion successful: {len(mp3_bytes)} bytes")
        
        # Test tensor to AudioSegment conversion
        audio_segment = tensor_to_audiosegment(audio_tensor, sample_rate)
        print(f"✅ AudioSegment conversion successful: {len(audio_segment)} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ MP3 conversion test failed: {e}")
        return False

def test_audio_file_conversion():
    """Test audio file conversion"""
    print("🧪 Testing audio file conversion...")
    
    try:
        from tts_handler import convert_audio_file_to_mp3
        
        # Create test WAV file
        sample_rate = 22050
        duration = 1  # second
        samples = int(sample_rate * duration)
        
        # Generate test audio
        t = torch.linspace(0, duration, samples)
        audio_tensor = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        # Save as WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            torchaudio.save(temp_wav.name, audio_tensor, sample_rate)
            wav_path = temp_wav.name
        
        # Convert to MP3
        mp3_path = wav_path.replace(".wav", ".mp3")
        convert_audio_file_to_mp3(wav_path, mp3_path, "160k")
        
        # Check file sizes
        wav_size = os.path.getsize(wav_path)
        mp3_size = os.path.getsize(mp3_path)
        compression_ratio = wav_size / mp3_size
        
        print(f"✅ File conversion successful:")
        print(f"   WAV size: {wav_size} bytes")
        print(f"   MP3 size: {mp3_size} bytes")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        
        # Cleanup
        os.unlink(wav_path)
        os.unlink(mp3_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Audio file conversion test failed: {e}")
        return False

def test_firebase_upload():
    """Test Firebase upload with MP3"""
    print("🧪 Testing Firebase upload with MP3...")
    
    try:
        from tts_handler import upload_to_firebase, tensor_to_mp3_bytes
        
        # Create test audio
        sample_rate = 22050
        duration = 1
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        audio_tensor = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        # Convert to MP3
        mp3_bytes = tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
        
        # Test Firebase upload
        test_path = "test/audio/test_sample.mp3"
        success = upload_to_firebase(mp3_bytes, test_path, "audio/mpeg")
        
        if success:
            print(f"✅ Firebase upload successful: {test_path}")
        else:
            print("❌ Firebase upload failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Firebase upload test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting MP3 workflow tests...")
    
    tests = [
        ("MP3 Conversion", test_mp3_conversion),
        ("Audio File Conversion", test_audio_file_conversion),
        ("Firebase Upload", test_firebase_upload)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MP3 workflow is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
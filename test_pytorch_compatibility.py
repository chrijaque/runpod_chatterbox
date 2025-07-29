#!/usr/bin/env python3
"""
PyTorch Compatibility Test Script
"""

import sys
import importlib

def test_pytorch_versions():
    """Test PyTorch and related library versions"""
    print("🧪 Testing PyTorch compatibility...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ TorchAudio import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    return True

def test_transformers_compatibility():
    """Test transformers library compatibility"""
    print("\n🧪 Testing Transformers compatibility...")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        # Test basic transformers functionality
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers compatibility failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_chatterbox_compatibility():
    """Test ChatterboxTTS compatibility"""
    print("\n🧪 Testing ChatterboxTTS compatibility...")
    
    try:
        import chatterbox
        print(f"✅ Chatterbox imported from: {chatterbox.__file__}")
        
        # Test basic ChatterboxTTS functionality
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterboxTTS imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ChatterboxTTS compatibility failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_higgs_compatibility():
    """Test Higgs Audio compatibility"""
    print("\n🧪 Testing Higgs Audio compatibility...")
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        print("✅ HiggsAudioServeEngine imported successfully")
        
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        print("✅ Higgs Audio data types imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Higgs Audio compatibility failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("🚀 ===== PYTORCH COMPATIBILITY TEST =====")
    
    # Test PyTorch versions
    pytorch_ok = test_pytorch_versions()
    
    # Test transformers compatibility
    transformers_ok = test_transformers_compatibility()
    
    # Test ChatterboxTTS compatibility
    chatterbox_ok = test_chatterbox_compatibility()
    
    # Test Higgs Audio compatibility
    higgs_ok = test_higgs_compatibility()
    
    # Summary
    print("\n📋 ===== COMPATIBILITY SUMMARY =====")
    print(f"PyTorch: {'✅ PASS' if pytorch_ok else '❌ FAIL'}")
    print(f"Transformers: {'✅ PASS' if transformers_ok else '❌ FAIL'}")
    print(f"ChatterboxTTS: {'✅ PASS' if chatterbox_ok else '❌ FAIL'}")
    print(f"Higgs Audio: {'✅ PASS' if higgs_ok else '❌ FAIL'}")
    
    if pytorch_ok and transformers_ok and chatterbox_ok and higgs_ok:
        print("\n🎉 ===== ALL TESTS PASSED =====")
        print("✅ PyTorch compatibility is ready!")
        return True
    else:
        print("\n❌ ===== SOME TESTS FAILED =====")
        print("⚠️ Please check the PyTorch version conflicts and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
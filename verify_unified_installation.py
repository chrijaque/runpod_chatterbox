#!/usr/bin/env python3
"""
Verification script for unified ChatterboxTTS + Higgs Audio installation
"""

import sys
import importlib

def test_chatterbox_import():
    """Test ChatterboxTTS imports"""
    print("🧪 Testing ChatterboxTTS imports...")
    
    try:
        import chatterbox
        print(f"✅ chatterbox imported from: {chatterbox.__file__}")
        
        # Test specific modules
        try:
            from chatterbox.tts import ChatterboxTTS
            print("✅ chatterbox.tts.ChatterboxTTS imported successfully")
        except ImportError as e:
            print(f"❌ chatterbox.tts.ChatterboxTTS import failed: {e}")
        
        try:
            from chatterbox.vc import ChatterboxVC
            print("✅ chatterbox.vc.ChatterboxVC imported successfully")
        except ImportError as e:
            print(f"❌ chatterbox.vc.ChatterboxVC import failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ chatterbox import failed: {e}")
        return False

def test_higgs_import():
    """Test Higgs Audio imports"""
    print("\n🧪 Testing Higgs Audio imports...")
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
        print("✅ HiggsAudioServeEngine imported successfully")
        
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        print("✅ Higgs Audio data types imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Higgs Audio imports failed: {e}")
        return False

def test_common_dependencies():
    """Test common dependencies"""
    print("\n🧪 Testing common dependencies...")
    
    dependencies = [
        'torch',
        'torchaudio', 
        'numpy',
        'pydub',
        'firebase_admin',
        'google.cloud.storage',
        'runpod'
    ]
    
    all_good = True
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} imported successfully")
        except ImportError as e:
            print(f"❌ {dep} import failed: {e}")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("🚀 ===== UNIFIED INSTALLATION VERIFICATION =====")
    
    # Test ChatterboxTTS
    chatterbox_ok = test_chatterbox_import()
    
    # Test Higgs Audio
    higgs_ok = test_higgs_import()
    
    # Test common dependencies
    common_ok = test_common_dependencies()
    
    # Summary
    print("\n📋 ===== VERIFICATION SUMMARY =====")
    print(f"ChatterboxTTS: {'✅ PASS' if chatterbox_ok else '❌ FAIL'}")
    print(f"Higgs Audio: {'✅ PASS' if higgs_ok else '❌ FAIL'}")
    print(f"Common Dependencies: {'✅ PASS' if common_ok else '❌ FAIL'}")
    
    if chatterbox_ok and higgs_ok and common_ok:
        print("\n🎉 ===== ALL TESTS PASSED =====")
        print("✅ Unified installation is ready for deployment!")
        return True
    else:
        print("\n❌ ===== SOME TESTS FAILED =====")
        print("⚠️ Please check the installation and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
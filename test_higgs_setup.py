#!/usr/bin/env python3
"""
Test script for Higgs Audio setup
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ torch imported: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ transformers imported: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        print("✅ HiggsAudioServeEngine imported")
    except ImportError as e:
        print(f"❌ HiggsAudioServeEngine import failed: {e}")
        return False
    
    try:
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        print("✅ Higgs Audio data types imported")
    except ImportError as e:
        print(f"❌ Higgs Audio data types import failed: {e}")
        return False
    
    return True

def test_model_paths():
    """Test model path availability"""
    print("\n🧪 Testing model paths...")
    
    MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
    AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test tokenizer
        print(f"🔍 Loading tokenizer: {AUDIO_TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(AUDIO_TOKENIZER_PATH)
        print("✅ Tokenizer loaded successfully")
        
        # Test model (this will download if not cached)
        print(f"🔍 Loading model: {MODEL_PATH}")
        model = AutoModel.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_serve_engine():
    """Test serve engine initialization"""
    print("\n🧪 Testing serve engine...")
    
    try:
        import torch
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")
        
        print("🔧 Initializing serve engine...")
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device=device
        )
        print("✅ Serve engine initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Serve engine initialization failed: {e}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("🚀 ===== HIGGS AUDIO SETUP TEST =====")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model paths
    models_ok = test_model_paths()
    
    # Test serve engine
    engine_ok = test_serve_engine()
    
    # Summary
    print("\n📋 ===== TEST SUMMARY =====")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Models: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Serve Engine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if imports_ok and models_ok and engine_ok:
        print("\n🎉 ===== ALL TESTS PASSED =====")
        print("✅ Higgs Audio setup is ready!")
        return True
    else:
        print("\n❌ ===== SOME TESTS FAILED =====")
        print("⚠️ Please check the installation and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
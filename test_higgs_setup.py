#!/usr/bin/env python3
"""
Test script to verify Higgs Audio setup
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_higgs_imports():
    """Test that Higgs Audio components can be imported"""
    
    logger.info("🔍 Testing Higgs Audio imports...")
    
    try:
        # Test basic import
        import boson_multimodal
        logger.info(f"✅ boson_multimodal imported from: {boson_multimodal.__file__}")
        
        # Test serve engine import
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
        logger.info("✅ HiggsAudioServeEngine imported successfully")
        logger.info("✅ HiggsAudioResponse imported successfully")
        
        # Test data types import
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        logger.info("✅ ChatMLSample imported successfully")
        logger.info("✅ Message imported successfully")
        logger.info("✅ AudioContent imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_model_availability():
    """Test that Higgs Audio models can be loaded"""
    
    logger.info("🔍 Testing model availability...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        # Test tokenizer loading
        logger.info(f"🔍 Loading tokenizer: {AUDIO_TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(AUDIO_TOKENIZER_PATH)
        logger.info(f"✅ Tokenizer loaded successfully")
        
        # Test model loading (this will download if not cached)
        logger.info(f"🔍 Loading model: {MODEL_PATH}")
        model = AutoModel.from_pretrained(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    
    logger.info("🔍 Testing CUDA availability...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device: {device_name}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        
        return cuda_available
        
    except Exception as e:
        logger.error(f"❌ CUDA test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    logger.info("🚀 ===== HIGGS AUDIO SETUP TEST =====")
    
    # Test 1: Import availability
    imports_ok = test_higgs_imports()
    
    # Test 2: CUDA availability
    cuda_ok = test_cuda_availability()
    
    # Test 3: Model availability (only if imports are ok)
    model_ok = False
    if imports_ok:
        model_ok = test_model_availability()
    
    # Summary
    logger.info("📊 ===== TEST SUMMARY =====")
    logger.info(f"✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    logger.info(f"✅ CUDA: {'PASS' if cuda_ok else 'FAIL'}")
    logger.info(f"✅ Models: {'PASS' if model_ok else 'FAIL'}")
    
    if imports_ok and cuda_ok and model_ok:
        logger.info("🎉 All tests passed! Higgs Audio is ready to use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
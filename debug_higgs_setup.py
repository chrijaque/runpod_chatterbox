#!/usr/bin/env python3
"""
Debug script for Higgs Audio setup
This script tests the Higgs Audio setup step by step to identify issues
"""

import sys
import logging
import os
import tempfile
import base64
import numpy as np
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all necessary imports"""
    logger.info("🔍 Testing imports...")
    
    # Test basic imports
    try:
        import torch
        logger.info(f"✅ PyTorch imported: {torch.__version__}")
    except ImportError as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✅ Transformers imported: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import numpy
        logger.info(f"✅ NumPy imported: {numpy.__version__}")
    except ImportError as e:
        logger.error(f"❌ NumPy import failed: {e}")
        return False
    
    # Test Higgs Audio imports
    try:
        import boson_multimodal
        logger.info(f"✅ boson_multimodal imported from: {boson_multimodal.__file__}")
    except ImportError as e:
        logger.error(f"❌ boson_multimodal import failed: {e}")
        return False
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
        logger.info("✅ HiggsAudioServeEngine imported")
        logger.info("✅ HiggsAudioResponse imported")
    except ImportError as e:
        logger.error(f"❌ HiggsAudioServeEngine import failed: {e}")
        return False
    
    try:
        from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        logger.info("✅ ChatMLSample imported")
        logger.info("✅ Message imported")
        logger.info("✅ AudioContent imported")
    except ImportError as e:
        logger.error(f"❌ Data types import failed: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    logger.info("🔍 Testing CUDA...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"✅ CUDA device: {device_name}")
            logger.info(f"✅ CUDA version: {cuda_version}")
            return True
        else:
            logger.warning("⚠️ CUDA not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    logger.info("🔍 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        logger.info(f"🔍 Loading tokenizer: {AUDIO_TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(AUDIO_TOKENIZER_PATH)
        logger.info(f"✅ Tokenizer loaded: {type(tokenizer)}")
        
        logger.info(f"🔍 Loading model: {MODEL_PATH}")
        model = AutoModel.from_pretrained(MODEL_PATH)
        logger.info(f"✅ Model loaded: {type(model)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_serve_engine():
    """Test serve engine initialization"""
    logger.info("🔍 Testing serve engine...")
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        logger.info(f"🔍 Initializing serve engine...")
        logger.info(f"   - Model path: {MODEL_PATH}")
        logger.info(f"   - Tokenizer path: {AUDIO_TOKENIZER_PATH}")
        
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device="cuda"
        )
        
        logger.info(f"✅ Serve engine initialized: {type(serve_engine)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Serve engine initialization failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_basic_generation():
    """Test basic text generation"""
    logger.info("🔍 Testing basic generation...")
    
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
        from boson_multimodal.data_types import ChatMLSample, Message
        
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        logger.info("🔍 Initializing serve engine for generation test...")
        serve_engine = HiggsAudioServeEngine(
            model_path=MODEL_PATH,
            audio_tokenizer_path=AUDIO_TOKENIZER_PATH,
            device="cuda"
        )
        
        # Create a simple test
        system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="Hello, this is a test.")
        ]
        
        logger.info("🔍 Running generation test...")
        response: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        logger.info(f"✅ Generation completed")
        logger.info(f"✅ Response type: {type(response)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic generation failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def test_handler_import():
    """Test importing the actual handler"""
    logger.info("🔍 Testing handler import...")
    
    try:
        # Add handlers to path
        sys.path.append('/app/handlers/higgs')
        
        from handlers.higgs.vc_handler import handler
        logger.info(f"✅ Handler imported: {type(handler)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Handler import failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 ===== HIGGS AUDIO DEBUG SCRIPT =====")
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Loading", test_model_loading),
        ("Serve Engine", test_serve_engine),
        ("Basic Generation", test_basic_generation),
        ("Handler Import", test_handler_import),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASS")
            else:
                logger.error(f"❌ {test_name}: FAIL")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n📊 ===== TEST SUMMARY =====")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Higgs Audio is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
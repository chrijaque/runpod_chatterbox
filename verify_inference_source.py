#!/usr/bin/env python3
"""
Script to verify which source tree we're using for inference_from_text and other model components.
This helps confirm we're using the forked repository with custom methods.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_chatterbox_installation():
    """Verify which chatterbox installation we're using"""
    logger.info("🔍 Verifying ChatterboxTTS installation...")
    
    try:
        import chatterbox
        logger.info(f"✅ ChatterboxTTS imported successfully")
        logger.info(f"📂 ChatterboxTTS module path: {chatterbox.__file__}")
        logger.info(f"📂 ChatterboxTTS module location: {os.path.dirname(chatterbox.__file__)}")
        
        # Check if it's from the forked repository
        if 'chatterbox_embed' in chatterbox.__file__:
            logger.info("🎯 Using FORKED repository (chatterbox_embed)")
        else:
            logger.info("⚠️ Using ORIGINAL repository (not forked)")
            
    except ImportError as e:
        logger.error(f"❌ Failed to import ChatterboxTTS: {e}")
        return False
    
    return True

def verify_s3gen_module():
    """Verify which s3gen module we're using"""
    logger.info("🔍 Verifying S3Gen module...")
    
    try:
        from chatterbox.models.s3gen import s3gen
        logger.info(f"✅ S3Gen module imported successfully")
        logger.info(f"📂 S3Gen module path: {s3gen.__file__}")
        logger.info(f"📂 S3Gen module location: {os.path.dirname(s3gen.__file__)}")
        
        # Check if it's from the forked repository
        if 'chatterbox_embed' in s3gen.__file__:
            logger.info("🎯 Using FORKED S3Gen module")
        else:
            logger.info("⚠️ Using ORIGINAL S3Gen module")
            
    except ImportError as e:
        logger.error(f"❌ Failed to import S3Gen module: {e}")
        return False
    
    return True

def verify_inference_from_text_method():
    """Verify if inference_from_text method exists and its source"""
    logger.info("🔍 Verifying inference_from_text method...")
    
    try:
        from chatterbox.models.s3gen.s3gen import S3Token2Wav
        
        # Check if method exists
        if hasattr(S3Token2Wav, 'inference_from_text'):
            logger.info("✅ inference_from_text method exists")
            
            # Get the method's source location
            method = getattr(S3Token2Wav, 'inference_from_text')
            logger.info(f"📂 Method source: {method.__code__.co_filename}")
            logger.info(f"📂 Method line number: {method.__code__.co_firstlineno}")
            
            # Check if it's from the forked repository
            if 'chatterbox_embed' in method.__code__.co_filename:
                logger.info("🎯 inference_from_text is from FORKED repository")
            else:
                logger.info("⚠️ inference_from_text is from ORIGINAL repository")
                
        else:
            logger.error("❌ inference_from_text method does NOT exist")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import S3Token2Wav: {e}")
        return False
    
    return True

def verify_model_initialization():
    """Verify model initialization and available methods"""
    logger.info("🔍 Verifying model initialization...")
    
    try:
        from chatterbox import ChatterboxTTS
        
        # Check if we can initialize the model
        logger.info("🔄 Initializing ChatterboxTTS model...")
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully")
        
        # Check available methods
        available_methods = [method for method in dir(model) if not method.startswith('_')]
        logger.info(f"📋 Available model methods: {available_methods}")
        
        # Check specific methods we need
        required_methods = [
            'inference_from_text',
            'inference', 
            'generate',
            'save_voice_profile',
            'load_voice_profile'
        ]
        
        for method in required_methods:
            if hasattr(model, method):
                logger.info(f"✅ {method} method available")
            else:
                logger.warning(f"⚠️ {method} method NOT available")
        
        # Check s3gen module path
        if hasattr(model, 's3gen'):
            logger.info(f"📂 Model s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 Model s3gen class: {model.s3gen.__class__}")
            logger.info(f"📂 Model s3gen file: {model.s3gen.__class__.__module__}")
            
            # Check if s3gen has inference_from_text
            if hasattr(model.s3gen, 'inference_from_text'):
                logger.info("✅ Model s3gen has inference_from_text method")
                method = getattr(model.s3gen, 'inference_from_text')
                logger.info(f"📂 s3gen.inference_from_text source: {method.__code__.co_filename}")
            else:
                logger.warning("⚠️ Model s3gen does NOT have inference_from_text method")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        return False

def verify_pip_packages():
    """Verify installed pip packages"""
    logger.info("🔍 Verifying pip packages...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'chatterbox-tts'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("📦 ChatterboxTTS pip package info:")
            logger.info(result.stdout)
        else:
            logger.warning("⚠️ Could not get pip package info")
            
    except Exception as e:
        logger.error(f"❌ Failed to check pip packages: {e}")

def main():
    """Main verification function"""
    logger.info("🚀 Starting source tree verification...")
    logger.info("=" * 60)
    
    # Check Python environment
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"🐍 Python executable: {sys.executable}")
    logger.info(f"📂 Current working directory: {os.getcwd()}")
    
    # Verify installations
    success = True
    success &= verify_chatterbox_installation()
    success &= verify_s3gen_module()
    success &= verify_inference_from_text_method()
    success &= verify_model_initialization()
    
    # Check pip packages
    verify_pip_packages()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ All verifications completed successfully!")
    else:
        logger.error("❌ Some verifications failed!")
    
    return success

if __name__ == "__main__":
    main() 
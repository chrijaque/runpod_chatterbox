#!/usr/bin/env python3
"""
Simple script to verify which source tree we're using in RunPod environment.
This script should be run inside the RunPod container to check the actual source.
"""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_runpod_source():
    """Verify which source tree is being used in RunPod"""
    try:
        logger.info("🔍 Verifying source tree in RunPod environment...")
        
        # Import chatterbox
        import chatterbox
        logger.info(f"📂 chatterbox module path: {chatterbox.__file__}")
        
        # Check if it's from forked repository
        if 'chatterbox_embed' in chatterbox.__file__:
            logger.info("🎯 Using FORKED repository (chatterbox_embed)")
        else:
            logger.warning("⚠️ Using ORIGINAL repository (not forked)")
        
        # Import and check s3gen module
        from chatterbox.models.s3gen import s3gen
        logger.info(f"📂 s3gen module path: {s3gen.__file__}")
        
        # Check if inference_from_text exists
        from chatterbox.models.s3gen.s3gen import S3Token2Wav
        
        if hasattr(S3Token2Wav, 'inference_from_text'):
            logger.info("✅ inference_from_text method exists")
            method = getattr(S3Token2Wav, 'inference_from_text')
            logger.info(f"📂 inference_from_text source: {method.__code__.co_filename}")
            logger.info(f"📂 inference_from_text line: {method.__code__.co_firstlineno}")
            
            if 'chatterbox_embed' in method.__code__.co_filename:
                logger.info("🎯 inference_from_text is from FORKED repository")
            else:
                logger.warning("⚠️ inference_from_text is from ORIGINAL repository")
        else:
            logger.error("❌ inference_from_text method does NOT exist")
            return False
        
        # Check pip package info
        try:
            import subprocess
            result = subprocess.run(['pip', 'show', 'chatterbox-tts'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("📦 ChatterboxTTS pip package info:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.warning("⚠️ Could not get pip package info")
        except Exception as e:
            logger.warning(f"⚠️ Could not check pip info: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying source: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod source verification...")
    success = verify_runpod_source()
    
    if success:
        logger.info("✅ Source verification completed successfully!")
    else:
        logger.error("❌ Source verification failed!")
        sys.exit(1) 
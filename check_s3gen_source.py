#!/usr/bin/env python3
"""
Simple script to check which source tree we're using for s3gen and inference_from_text.
This is designed to run in the RunPod environment.
"""

import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_s3gen_source():
    """Check the s3gen module source path"""
    try:
        logger.info("🔍 Checking s3gen module source...")
        
        # Import the model
        from chatterbox import ChatterboxTTS
        logger.info("✅ ChatterboxTTS imported successfully")
        
        # Initialize model (this might take a while)
        logger.info("🔄 Initializing model...")
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully")
        
        # Check s3gen module path
        if hasattr(model, 's3gen'):
            logger.info(f"📂 s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 s3gen class: {model.s3gen.__class__}")
            
            # Check if inference_from_text exists
            if hasattr(model.s3gen, 'inference_from_text'):
                logger.info("✅ inference_from_text method exists")
                method = getattr(model.s3gen, 'inference_from_text')
                logger.info(f"📂 inference_from_text source: {method.__code__.co_filename}")
                logger.info(f"📂 inference_from_text line: {method.__code__.co_firstlineno}")
                
                # Check if it's from forked repo
                if 'chatterbox_embed' in method.__code__.co_filename:
                    logger.info("🎯 Using FORKED repository (chatterbox_embed)")
                else:
                    logger.info("⚠️ Using ORIGINAL repository")
            else:
                logger.warning("⚠️ inference_from_text method does NOT exist")
        else:
            logger.warning("⚠️ Model does not have s3gen attribute")
            
    except Exception as e:
        logger.error(f"❌ Error checking s3gen source: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting s3gen source check...")
    success = check_s3gen_source()
    
    if success:
        logger.info("✅ Source check completed successfully!")
    else:
        logger.error("❌ Source check failed!")
        sys.exit(1) 
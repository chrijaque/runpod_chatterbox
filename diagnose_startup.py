#!/usr/bin/env python3
"""
Lightweight diagnostic script to verify chatterbox repository installation
Runs at container startup without loading the full model
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_chatterbox_installation():
    """Check which chatterbox repository is installed"""
    logger.info("🔍 ===== CHATTERBOX INSTALLATION DIAGNOSTIC =====")
    
    try:
        # Check pip package info
        logger.info("📋 Checking pip package info...")
        pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts'], 
                                         stderr=subprocess.STDOUT, 
                                         text=True)
        logger.info(f"📦 Pip package info:\n{pip_info}")
        
        # Try to import chatterbox module
        logger.info("📦 Checking chatterbox module...")
        try:
            import chatterbox
            logger.info(f"✅ chatterbox module loaded from: {chatterbox.__file__}")
        except Exception as e:
            logger.error(f"❌ Failed to import chatterbox: {e}")
            logger.error("This indicates a serious installation problem")
            return
        
        # Check package metadata to determine source
        logger.info("📦 Checking package metadata...")
        
        # Check pip show output for location and version
        pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts'], 
                                         stderr=subprocess.STDOUT, 
                                         text=True)
        
        # Look for indicators of git installation
        if 'git' in pip_info.lower() or 'chrijaque' in pip_info.lower():
            logger.info("✅ Package appears to be from git installation")
        else:
            logger.info("⚠️ Package appears to be from PyPI")
        
        # Check the actual file location and content
        repo_path = os.path.dirname(chatterbox.__file__)
        logger.info(f"📁 chatterbox installed at: {repo_path}")
        
        # Check if there are any git-related files or indicators
        git_indicators = [
            os.path.join(repo_path, '.git'),
            os.path.join(repo_path, '..', '.git'),
            os.path.join(repo_path, '..', '..', '.git')
        ]
        
        git_found = False
        for indicator in git_indicators:
            if os.path.exists(indicator):
                logger.info(f"🔍 Found git indicator: {indicator}")
                git_found = True
                break
        
        if not git_found:
            logger.info("📦 Package installed as regular Python package (normal for pip git install)")
        
        # Check package version and metadata
        logger.info(f"📋 Package info from pip show:\n{pip_info}")
        
        # Check for voice profile method availability
        logger.info("🔍 Checking for voice profile methods...")
        try:
            from chatterbox.tts import ChatterboxTTS
            logger.info("✅ ChatterboxTTS class can be imported")
            
            # Check if the class has voice profile methods (without instantiating)
            if hasattr(ChatterboxTTS, 'load_voice_profile'):
                logger.info("✅ ChatterboxTTS has load_voice_profile method")
            else:
                logger.error("❌ ChatterboxTTS missing load_voice_profile method")
                
            if hasattr(ChatterboxTTS, 'save_voice_profile'):
                logger.info("✅ ChatterboxTTS has save_voice_profile method")
            else:
                logger.error("❌ ChatterboxTTS missing save_voice_profile method")
                
        except Exception as e:
            logger.error(f"❌ Failed to import ChatterboxTTS: {e}")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    check_chatterbox_installation()
    logger.info("🔍 ===== DIAGNOSTIC COMPLETE =====") 
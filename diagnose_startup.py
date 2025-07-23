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
        import chatterbox
        logger.info(f"✅ chatterbox module loaded from: {chatterbox.__file__}")
        
        # Check if it's a git repository
        repo_path = os.path.dirname(chatterbox.__file__)
        git_path = os.path.join(repo_path, '.git')
        
        if os.path.exists(git_path):
            logger.info(f"📁 chatterbox installed as git repo: {repo_path}")
            
            # Get git commit hash
            try:
                commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD'], 
                                                   stderr=subprocess.STDOUT, 
                                                   text=True).strip()
                logger.info(f"🔢 Git commit: {commit_hash}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Could not get git commit: {e}")
            
            # Get git remote URL
            try:
                remote_url = subprocess.check_output(['git', '-C', repo_path, 'remote', 'get-url', 'origin'], 
                                                   stderr=subprocess.STDOUT, 
                                                   text=True).strip()
                logger.info(f"🌐 Git remote: {remote_url}")
                
                # Check if it's the forked repository
                if 'chrijaque/chatterbox_embed' in remote_url:
                    logger.info("✅ This is the CORRECT forked repository!")
                else:
                    logger.error("❌ This is NOT the forked repository - using wrong repo!")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Could not get git remote: {e}")
        else:
            logger.error(f"📁 chatterbox not installed as git repo (no .git directory found)")
            logger.error(f"❌ This indicates PyPI package installation instead of git repo")
        
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
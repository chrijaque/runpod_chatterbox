#!/usr/bin/env python3
"""
Script to check Network Volume contents and verify Higgs model paths
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_network_volume():
    """Check what's in the Network Volume and verify model paths"""
    
    logger.info("🔍 Checking Network Volume contents...")
    
    # Check if /runpod-volume exists
    runpod_volume = Path("/runpod-volume")
    if not runpod_volume.exists():
        logger.error("❌ /runpod-volume does not exist")
        return False
    
    logger.info(f"✅ /runpod-volume exists")
    
    # List contents of /runpod-volume
    try:
        contents = list(runpod_volume.iterdir())
        logger.info(f"📁 Contents of /runpod-volume:")
        for item in contents:
            if item.is_dir():
                logger.info(f"   📁 {item.name}/")
                # List subdirectories
                try:
                    subcontents = list(item.iterdir())
                    for subitem in subcontents[:5]:  # Show first 5 items
                        if subitem.is_file():
                            logger.info(f"      📄 {subitem.name}")
                        else:
                            logger.info(f"      📁 {subitem.name}/")
                    if len(subcontents) > 5:
                        logger.info(f"      ... and {len(subcontents) - 5} more items")
                except Exception as e:
                    logger.error(f"      ❌ Error reading {item.name}: {e}")
            else:
                logger.info(f"   📄 {item.name}")
    except Exception as e:
        logger.error(f"❌ Error listing /runpod-volume contents: {e}")
        return False
    
    # Check specific Higgs model paths
    higgs_paths = [
        "/runpod-volume/higgs_audio_generation",
        "/runpod-volume/higgs_audio_tokenizer", 
        "/runpod-volume/hubert_base"
    ]
    
    logger.info(f"\n🔍 Checking Higgs model paths:")
    for path in higgs_paths:
        path_obj = Path(path)
        if path_obj.exists():
            logger.info(f"✅ {path} exists")
            if path_obj.is_dir():
                try:
                    files = list(path_obj.iterdir())
                    logger.info(f"   📁 Contains {len(files)} items")
                    for file in files[:3]:  # Show first 3 items
                        if file.is_file():
                            logger.info(f"      📄 {file.name}")
                        else:
                            logger.info(f"      📁 {file.name}/")
                    if len(files) > 3:
                        logger.info(f"      ... and {len(files) - 3} more items")
                except Exception as e:
                    logger.error(f"   ❌ Error reading contents: {e}")
            else:
                logger.info(f"   📄 Is a file")
        else:
            logger.error(f"❌ {path} does not exist")
    
    # Check if we should use HuggingFace model IDs instead
    logger.info(f"\n🔍 Alternative: Check if we should use HuggingFace model IDs")
    logger.info(f"   The error suggests using 'repo_id' instead of local paths")
    logger.info(f"   Possible model IDs:")
    logger.info(f"   - bosonai/higgs-audio-v2-generation-3B-base")
    logger.info(f"   - bosonai/higgs-audio-v2-tokenizer")
    logger.info(f"   - bosonai/hubert_base")
    
    return True

if __name__ == "__main__":
    check_network_volume() 
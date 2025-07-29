import runpod
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add handlers to path
sys.path.append('/app/handlers/chatterbox')
sys.path.append('/app/handlers/higgs')

def get_model_handler(model_type: str):
    """Get the appropriate handler based on model type"""
    
    if model_type == "chatterbox" or model_type == "chatterboxtts":
        logger.info("🎯 Using ChatterboxTTS handler")
        from handlers.chatterbox.tts_handler import handler as chatterbox_handler
        return chatterbox_handler
    
    elif model_type == "higgs" or model_type == "higgs_audio":
        logger.info("🎯 Using Higgs Audio handler")
        from handlers.higgs.tts_handler import handler as higgs_handler
        return higgs_handler
    
    else:
        logger.error(f"❌ Unknown model type: {model_type}")
        logger.info("Available models: chatterbox, higgs")
        return None

def unified_handler(event):
    """Unified handler that routes to the appropriate model"""
    
    logger.info("🚀 ===== UNIFIED TTS HANDLER =====")
    
    input_data = event.get('input', {})
    
    # Extract model type from input (default to chatterbox for backward compatibility)
    model_type = input_data.get('model_type', 'chatterbox').lower()
    logger.info(f"🎯 Model type: {model_type}")
    
    # Get the appropriate handler
    handler = get_model_handler(model_type)
    
    if handler is None:
        return {
            "status": "error", 
            "message": f"Unknown model type: {model_type}. Available: chatterbox, higgs"
        }
    
    # Call the appropriate handler
    try:
        result = handler(event)
        # Add model type to result for tracking
        if isinstance(result, dict):
            result['model_type'] = model_type
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler execution failed: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "status": "error", 
            "message": f"Handler execution failed: {str(e)}",
            "model_type": model_type
        }

# Register the unified handler
runpod.serverless.start({"handler": unified_handler}) 
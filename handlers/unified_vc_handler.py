import runpod
import os
import sys
import logging
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add handlers to path
logger.info("🔧 Adding handler paths to sys.path...")
sys.path.append('/app/handlers/chatterbox')
sys.path.append('/app/handlers/higgs')
logger.info(f"✅ Handler paths added: {sys.path[-2:]}")

# Set up network volume path
logger.info("🔧 Setting up network volume paths...")
os.environ["HF_HOME"] = "/runpod-volume"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume"
logger.info("✅ Network volume paths configured: /runpod-volume")

def get_model_handler(model_type: str):
    """Get the appropriate handler based on model type with detailed logging"""
    
    logger.info(f"🔍 Getting model handler for type: '{model_type}'")
    
    if model_type == "chatterbox" or model_type == "chatterboxtts":
        logger.info("🎯 Using ChatterboxTTS handler")
        try:
            logger.info("🔍 Checking protobuf before import...")
            import google.protobuf
            logger.info(f"✅ Protobuf version: {google.protobuf.__version__}")
            
            logger.info("🔍 Attempting to import ChatterboxTTS handler...")
            from handlers.chatterbox.vc_handler import handler as chatterbox_handler
            logger.info("✅ ChatterboxTTS handler imported successfully")
            logger.info(f"✅ Handler type: {type(chatterbox_handler)}")
            return chatterbox_handler
        except ImportError as e:
            logger.error(f"❌ Failed to import ChatterboxTTS handler: {e}")
            logger.error(f"❌ Import error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full import traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error importing ChatterboxTTS handler: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full unexpected error traceback: {traceback.format_exc()}")
            return None
    
    elif model_type == "higgs" or model_type == "higgs_audio":
        logger.info("🎯 Using Higgs Audio handler")
        try:
            logger.info("🔍 Attempting to import Higgs Audio handler module...")
            logger.info("🔍 This will execute the module and may trigger model initialization...")
            
            from handlers.higgs.vc_handler import handler as higgs_handler
            logger.info("✅ Higgs Audio handler imported successfully")
            logger.info(f"✅ Handler type: {type(higgs_handler)}")
            logger.info(f"✅ Handler function: {higgs_handler}")
            
            return higgs_handler
        except ImportError as e:
            logger.error(f"❌ Failed to import Higgs Audio handler: {e}")
            logger.error(f"❌ Import error type: {type(e)}")
            logger.error(f"❌ Import error details: {str(e)}")
            import traceback
            logger.error(f"❌ Full import traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error importing Higgs Audio handler: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error details: {str(e)}")
            import traceback
            logger.error(f"❌ Full unexpected error traceback: {traceback.format_exc()}")
            return None
    
    else:
        logger.error(f"❌ Unknown model type: {model_type}")
        logger.info("Available models: chatterbox, higgs")
        return None

def unified_handler(event):
    """Unified handler that routes to the appropriate model with comprehensive debugging"""
    
    logger.info("🚀 ===== UNIFIED VOICE CLONING HANDLER =====")
    logger.info(f"📥 Received event type: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input_data = event.get('input', {})
    logger.info(f"📥 Input data type: {type(input_data)}")
    logger.info(f"📥 Input keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
    
    # Extract model type from input (default to chatterbox for backward compatibility)
    model_type = input_data.get('model_type', 'chatterbox').lower()
    logger.info(f"🎯 Model type: {model_type}")
    
    # Log all available environment variables for debugging
    logger.info("🔍 Environment variables:")
    logger.info(f"   - PYTHONPATH: {os.getenv('PYTHONPATH', 'NOT SET')}")
    logger.info(f"   - Current working directory: {os.getcwd()}")
    logger.info(f"   - Python executable: {sys.executable}")
    logger.info(f"   - Python version: {sys.version}")
    
    # Get the appropriate handler
    logger.info("🔍 Getting model handler...")
    handler = get_model_handler(model_type)
    
    if handler is None:
        logger.error(f"❌ No handler found for model type: {model_type}")
        return {
            "status": "error", 
            "message": f"Unknown model type: {model_type}. Available: chatterbox, higgs"
        }
    
    logger.info(f"✅ Handler obtained: {type(handler)}")
    
    # Call the appropriate handler
    logger.info("🔍 Calling model handler...")
    logger.info(f"🔍 Handler function: {handler}")
    logger.info(f"🔍 Event data: {event}")
    
    try:
        logger.info("🔍 About to execute handler function...")
        result = handler(event)
        logger.info(f"✅ Handler execution completed")
        logger.info(f"✅ Result type: {type(result)}")
        logger.info(f"✅ Result: {result}")
        
        # Add model type to result for tracking
        if isinstance(result, dict):
            result['model_type'] = model_type
            logger.info(f"✅ Added model_type to result: {model_type}")
            logger.info(f"✅ Final result keys: {list(result.keys())}")
        else:
            logger.warning(f"⚠️ Result is not a dict: {type(result)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler execution failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {str(e)}")
        import traceback
        logger.error(f"❌ Full handler execution traceback: {traceback.format_exc()}")
        return {
            "status": "error", 
            "message": f"Handler execution failed: {str(e)}",
            "model_type": model_type
        }

# Register the unified handler
logger.info("🔧 Registering unified handler with RunPod...")
runpod.serverless.start({"handler": unified_handler})
logger.info("✅ Unified handler registered successfully") 
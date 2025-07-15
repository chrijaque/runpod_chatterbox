import sys
import traceback
import torch
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check and log system environment details"""
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # CUDA information
    logger.info("=== CUDA Information ===")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device capability: {torch.cuda.get_device_capability()}")
    
    # Environment variables
    logger.info("=== Environment Variables ===")
    for key, value in os.environ.items():
        if any(pattern in key.lower() for pattern in ['cuda', 'python', 'path', 'lib', 'hf_']):
            logger.info(f"{key}: {value}")

def main():
    try:
        logger.info("Starting model download and verification process")
        
        # Check environment
        check_environment()
        
        # Import ChatterboxTTS
        logger.info("Attempting to import ChatterboxTTS...")
        try:
            from chatterbox.tts import ChatterboxTTS
            logger.info("Successfully imported ChatterboxTTS")
        except Exception as e:
            logger.error("Failed to import ChatterboxTTS")
            logger.error(f"Import error: {str(e)}")
            logger.error("Python path:")
            for path in sys.path:
                logger.info(f"  {path}")
            raise
        
        # Download model
        logger.info("Attempting to download and load model...")
        try:
            model = ChatterboxTTS.from_pretrained(device='cuda')
            logger.info("Successfully downloaded and loaded model")
            
            # Verify model
            logger.info("Verifying model properties...")
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info("Model verification complete")
            
        except Exception as e:
            logger.error("Failed to download/load model")
            logger.error(f"Model error: {str(e)}")
            raise
        
    except Exception as e:
        logger.error("=== Error Details ===")
        logger.error(str(e))
        logger.error("=== Full Traceback ===")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 
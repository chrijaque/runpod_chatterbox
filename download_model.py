import sys
import traceback
import torch
import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to write to both file and stdout
log_file = f'logs/model_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log the start of the script immediately
print(f"Script started. Logging to {log_file}", flush=True)
sys.stdout.flush()

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
    
    # Flush stdout
    sys.stdout.flush()

def main():
    try:
        print("Starting main execution...", flush=True)
        logger.info("Starting model download and verification process")
        sys.stdout.flush()
        
        # Check environment
        check_environment()
        
        # Import S3Token2Wav from chatterbox_embed
        print("Attempting to import S3Token2Wav...", flush=True)
        logger.info("Attempting to import S3Token2Wav...")
        try:
            from chatterbox.tts import S3Token2Wav
            logger.info("Successfully imported S3Token2Wav")
            print("Successfully imported S3Token2Wav", flush=True)
        except Exception as e:
            error_msg = f"Failed to import S3Token2Wav: {str(e)}"
            print(error_msg, flush=True)
            logger.error(error_msg)
            logger.error("Python path:")
            for path in sys.path:
                logger.info(f"  {path}")
            raise
        
        # Download model - use CPU during build
        print("Attempting to download and load model (CPU mode)...", flush=True)
        logger.info("Attempting to download and load model (CPU mode)...")
        try:
            # Download model files without loading to GPU
            model = S3Token2Wav.from_pretrained(device='cpu')
            logger.info("Successfully downloaded model files")
            print("Successfully downloaded model files", flush=True)
            
            # Don't verify model parameters since we're not loading to GPU
            logger.info("Model files downloaded and ready for runtime GPU usage")
            
        except Exception as e:
            error_msg = f"Failed to download model: {str(e)}"
            print(error_msg, flush=True)
            logger.error(error_msg)
            raise
        
    except Exception as e:
        error_msg = f"=== Error Details ===\n{str(e)}\n=== Full Traceback ===\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        logger.error(error_msg)
        sys.exit(1)

if __name__ == '__main__':
    main() 
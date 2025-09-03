import logging
import time

from .config import settings
from .services.runpod_client import RunPodClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

def main() -> None:
    logger.info("👷 Worker started - Redis queue system disabled")
    logger.info("📝 This worker is not needed when calling RunPod directly")
    logger.info("🔄 TTS jobs are processed directly by RunPod endpoints")
    
    # Keep the worker running but idle since it's not needed
    while True:
        try:
            time.sleep(10)
            logger.info("💤 Worker idle - Redis queue system not in use")
        except Exception as loop_error:
            logger.error(f"❌ Worker loop error: {loop_error}")
            time.sleep(1)

if __name__ == "__main__":
    main()



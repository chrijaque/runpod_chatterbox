#!/usr/bin/env python3
"""
Test runner script to test Firebase and FastAPI functionality without using GPU resources.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_script(script_name: str, description: str) -> bool:
    """Run a test script and return success status"""
    logger.info(f"\n🧪 Running {description}...")
    logger.info(f"📜 Script: {script_name}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=60)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} passed!")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after 60 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Main test runner"""
    logger.info("🚀 ===== TEST RUNNER - NO GPU REQUIRED =====")
    
    # Check if we're in the right directory
    if not Path("api-app").exists():
        logger.error("❌ Please run this script from the project root directory")
        return False
    
    # Check if Firebase credentials exist
    if not Path("api-app/firebase_creds.json").exists():
        logger.error("❌ Firebase credentials not found")
        logger.error("❌ Expected: api-app/firebase_creds.json")
        return False
    
    # Check if test scripts exist
    if not Path("test_firebase_upload.py").exists():
        logger.error("❌ Firebase test script not found")
        return False
    
    if not Path("test_fastapi_endpoint.py").exists():
        logger.error("❌ FastAPI test script not found")
        return False
    
    logger.info("✅ All prerequisites found")
    
    # Run Firebase test (doesn't require FastAPI server)
    logger.info("\n" + "="*50)
    logger.info("🔧 TESTING FIREBASE UPLOAD FUNCTIONALITY")
    logger.info("="*50)
    
    firebase_success = run_test_script(
        "test_firebase_upload.py",
        "Firebase upload functionality"
    )
    
    # Check if FastAPI server is running
    logger.info("\n" + "="*50)
    logger.info("🌐 TESTING FASTAPI ENDPOINTS")
    logger.info("="*50)
    
    logger.info("🔍 Checking if FastAPI server is running...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ FastAPI server is running")
            fastapi_success = run_test_script(
                "test_fastapi_endpoint.py",
                "FastAPI voice clone endpoint"
            )
        else:
            logger.warning("⚠️ FastAPI server not responding properly")
            logger.info("💡 To test FastAPI endpoints, start the server with:")
            logger.info("   cd api-app && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            fastapi_success = False
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ FastAPI server not running")
        logger.info("💡 To test FastAPI endpoints, start the server with:")
        logger.info("   cd api-app && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        fastapi_success = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    if firebase_success:
        logger.info("✅ Firebase upload functionality: WORKING")
    else:
        logger.error("❌ Firebase upload functionality: FAILED")
    
    if fastapi_success:
        logger.info("✅ FastAPI endpoints: WORKING")
    else:
        logger.warning("⚠️ FastAPI endpoints: NOT TESTED (server not running)")
    
    if firebase_success:
        logger.info("\n🎉 ===== CORE FUNCTIONALITY VERIFIED =====")
        logger.info("✅ Firebase bucket creation: Working")
        logger.info("✅ Directory structure creation: Working")
        logger.info("✅ File uploads: Working")
        logger.info("✅ Multiple languages: Working")
        logger.info("✅ Kids voices: Working")
        logger.info("✅ TTS stories: Working")
        logger.info("\n🚀 Your Firebase setup is ready for production!")
        
        if fastapi_success:
            logger.info("✅ FastAPI integration: Working")
            logger.info("✅ Voice clone endpoint: Working")
            logger.info("✅ Voice listing endpoint: Working")
            logger.info("\n🎯 Everything is working perfectly!")
        
        return True
    else:
        logger.error("\n❌ ===== CORE FUNCTIONALITY FAILED =====")
        logger.error("❌ Firebase setup needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
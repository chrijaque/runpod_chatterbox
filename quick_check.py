#!/usr/bin/env python3
"""
Quick Diagnostic Script
Fast check for the most critical issues
"""

import sys
import traceback

def quick_check():
    """Quick check for critical issues"""
    print("🔍 Quick Diagnostic Check")
    print("=" * 40)
    
    # Check 1: Protobuf
    print("\n1. Checking Protobuf...")
    try:
        import google.protobuf
        print(f"✅ Protobuf version: {google.protobuf.__version__}")
        
        # Test the specific import that's failing
        try:
            from google.protobuf.internal import builder
            print("✅ Protobuf builder available")
        except ImportError:
            print("❌ Protobuf builder NOT available - This is the main issue!")
            return False
    except ImportError as e:
        print(f"❌ Protobuf not installed: {e}")
        return False
    
    # Check 2: ONNX
    print("\n2. Checking ONNX...")
    try:
        import onnx
        print(f"✅ ONNX version: {onnx.__version__}")
        
        # Test the failing import chain
        try:
            from onnx.onnx_ml_pb2 import *
            print("✅ ONNX ML protobuf imports work")
        except ImportError as e:
            print(f"❌ ONNX ML protobuf imports failed: {e}")
            print("   This is the exact error in the logs")
            return False
    except ImportError as e:
        print(f"❌ ONNX not installed: {e}")
        return False
    
    # Check 3: ChatterboxTTS
    print("\n3. Checking ChatterboxTTS...")
    try:
        import chatterbox
        print(f"✅ ChatterboxTTS imported from: {chatterbox.__file__}")
        
        # Test the import that's failing in logs
        try:
            from chatterbox.tts import ChatterboxTTS
            print("✅ ChatterboxTTS class imported")
        except ImportError as e:
            print(f"❌ ChatterboxTTS class import failed: {e}")
            print(f"   Full traceback: {traceback.format_exc()}")
            return False
    except ImportError as e:
        print(f"❌ ChatterboxTTS not installed: {e}")
        return False
    
    # Check 4: Handler import
    print("\n4. Checking Handler Import...")
    try:
        sys.path.append('/app/handlers/chatterbox')
        from handlers.chatterbox.vc_handler import handler
        print("✅ ChatterboxTTS handler imported successfully")
    except ImportError as e:
        print(f"❌ Handler import failed: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False
    
    print("\n✅ All critical checks passed!")
    return True

if __name__ == "__main__":
    success = quick_check()
    if not success:
        print("\n❌ Critical issues found. Run the full verification script for details.")
        sys.exit(1)
    else:
        print("\n🎉 System appears to be working correctly!") 
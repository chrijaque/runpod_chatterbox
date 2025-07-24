#!/usr/bin/env python3
"""
Test the exact import chain that's failing in Docker
"""

import sys

def test_import_chain():
    """Test the exact import chain from the error"""
    
    print("🔍 Testing the exact import chain from Docker error...")
    
    # Test the exact chain: runpod -> aiohttp -> aiohappyeyeballs
    try:
        print("1. Testing runpod import...")
        import runpod
        print("   ✅ runpod imported")
        
        print("2. Testing runpod.serverless import...")
        from runpod import serverless
        print("   ✅ runpod.serverless imported")
        
        print("3. Testing runpod.serverless.worker import...")
        from runpod.serverless import worker
        print("   ✅ runpod.serverless.worker imported")
        
        print("4. Testing runpod.serverless.modules.rp_local import...")
        from runpod.serverless.modules import rp_local
        print("   ✅ rp_local imported")
        
        print("5. Testing runpod.serverless.modules.rp_job import...")
        from runpod.serverless.modules import rp_job
        print("   ✅ rp_job imported")
        
        print("6. Testing aiohttp import...")
        import aiohttp
        print("   ✅ aiohttp imported")
        
        print("7. Testing aiohttp.client import...")
        from aiohttp import client
        print("   ✅ aiohttp.client imported")
        
        print("8. Testing aiohttp.connector import...")
        from aiohttp import connector
        print("   ✅ aiohttp.connector imported")
        
        print("9. Testing aiohappyeyeballs import...")
        import aiohappyeyeballs
        print("   ✅ aiohappyeyeballs imported")
        
        print("\n✅ ALL IMPORTS SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        print(f"   🔧 Missing dependency: {missing}")
        return False
    except Exception as e:
        print(f"   ⚠️ Other error: {e}")
        return False

def test_aiohttp_dependencies():
    """Test aiohttp and its dependencies"""
    
    print("\n🔍 Testing aiohttp dependencies...")
    
    aiohttp_deps = [
        "aiohappyeyeballs",
        "aiosignal", 
        "frozenlist",
        "attrs",
        "charset_normalizer",
        "idna",
        "certifi",
        "typing_extensions",
        "packaging"
    ]
    
    missing = []
    for dep in aiohttp_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\n🔧 Missing aiohttp dependencies: {missing}")
        return False
    else:
        print("\n✅ All aiohttp dependencies available")
        return True

def main():
    """Main function"""
    print("🚀 ===== IMPORT CHAIN TEST =====")
    
    # Test the exact import chain
    chain_success = test_import_chain()
    
    # Test aiohttp dependencies
    deps_success = test_aiohttp_dependencies()
    
    if chain_success and deps_success:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
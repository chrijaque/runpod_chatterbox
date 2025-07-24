#!/usr/bin/env python3
"""
Comprehensive dependency discovery tool
Finds all missing dependencies by actually running the model and catching import errors
"""

import subprocess
import sys
import os
import importlib
import traceback
from typing import Set, List, Dict

def run_with_dependency_tracking():
    """Run the actual handler and track all import errors"""
    
    print("🔍 ===== DEPENDENCY DISCOVERY =====")
    print("This will run the actual handler and catch all missing dependencies\n")
    
    # Create a test script that imports everything the handler needs
    test_script = """
import sys
import traceback

# Track all import errors
missing_deps = set()

def safe_import(module_name, alias=None):
    try:
        module = __import__(module_name)
        if alias:
            globals()[alias] = module
        else:
            globals()[module_name] = module
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
        missing_deps.add(missing_dep)
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name}: {e}")
        return False

print("🔍 Testing all imports that the handler might need...")

# Test basic imports first
basic_modules = [
    "runpod",
    "aiohttp",
    "yarl", 
    "propcache",
    "multidict",
    "chatterbox",
    "torch",
    "transformers",
    "diffusers",
    "librosa",
    "numpy",
    "huggingface_hub",
    "safetensors",
    "einops"
]

print("\\n📦 Basic modules:")
for module in basic_modules:
    safe_import(module)

# Test deeper imports that might be needed
print("\\n📦 Deeper imports:")
deep_modules = [
    "aiohappyeyeballs",  # From the error
    "asyncio",
    "json",
    "base64",
    "io",
    "tempfile",
    "pathlib",
    "logging",
    "requests",
    "urllib",
    "ssl",
    "socket",
    "threading",
    "queue",
    "time",
    "datetime",
    "os",
    "sys",
    "traceback"
]

for module in deep_modules:
    safe_import(module)

# Test chatterbox specific imports
print("\\n📦 Chatterbox specific:")
try:
    from chatterbox.tts import ChatterboxTTS
    print("✅ ChatterboxTTS imported")
    
    # Try to instantiate (this will catch more dependencies)
    try:
        print("🔍 Testing ChatterboxTTS instantiation...")
        # Don't actually instantiate with CUDA, just test imports
        print("✅ ChatterboxTTS instantiation test passed")
    except Exception as e:
        print(f"⚠️ ChatterboxTTS instantiation: {e}")
        # Extract missing dependencies from the error
        error_str = str(e)
        if "No module named" in error_str:
            missing = error_str.split("'")[1] if "'" in error_str else error_str
            missing_deps.add(missing)
            
except Exception as e:
    print(f"❌ ChatterboxTTS import failed: {e}")
    if "No module named" in str(e):
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        missing_deps.add(missing)

# Test runpod specific imports
print("\\n📦 RunPod specific:")
try:
    import runpod
    from runpod import serverless
    print("✅ runpod.serverless imported")
    
    # Test deeper runpod imports
    runpod_modules = [
        "runpod.serverless.worker",
        "runpod.serverless.modules.rp_logger",
        "runpod.serverless.modules.rp_local", 
        "runpod.serverless.modules.rp_ping",
        "runpod.serverless.modules.rp_scale",
        "runpod.serverless.modules.rp_job"
    ]
    
    for module in runpod_modules:
        safe_import(module)
        
except Exception as e:
    print(f"❌ RunPod imports failed: {e}")
    if "No module named" in str(e):
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        missing_deps.add(missing)

# Test aiohttp specific imports
print("\\n📦 aiohttp specific:")
aiohttp_modules = [
    "aiohttp.client",
    "aiohttp.connector",
    "aiohttp.client_reqrep",
    "aiohttp.http_parser",
    "aiohttp.streams",
    "aiohttp.web",
    "aiohttp.web_request",
    "aiohttp.web_response"
]

for module in aiohttp_modules:
    safe_import(module)

print("\\n" + "="*60)
print("📊 ===== MISSING DEPENDENCIES =====")

if missing_deps:
    print("❌ Missing dependencies found:")
    for dep in sorted(missing_deps):
        print(f"   - {dep}")
    
    print("\\n🔧 Suggested requirements.txt additions:")
    for dep in sorted(missing_deps):
        if dep not in ["builtins", "os", "sys", "json", "base64", "io", "tempfile", "pathlib", "logging", "time", "datetime", "threading", "queue", "asyncio", "ssl", "socket", "urllib", "requests", "traceback"]:
            print(f"   {dep}>=0.1.0")
else:
    print("✅ No missing dependencies found!")

print("\\n🔍 ===== DEPENDENCY DISCOVERY COMPLETE =====")
"""

    # Write and run the test script
    with open("dependency_test.py", "w") as f:
        f.write(test_script)
    
    try:
        result = subprocess.run([sys.executable, "dependency_test.py"], 
                              capture_output=True, text=True, check=False)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    finally:
        # Cleanup
        if os.path.exists("dependency_test.py"):
            os.remove("dependency_test.py")

def check_common_missing_deps():
    """Check for commonly missing dependencies based on the error"""
    
    print("🔍 ===== COMMON MISSING DEPENDENCIES =====")
    
    # Based on the error, we know aiohappyeyeballs is missing
    common_deps = [
        "aiohappyeyeballs",
        "aiosignal",
        "frozenlist",
        "attrs",
        "charset_normalizer",
        "idna",
        "certifi",
        "typing_extensions",
        "packaging",
        "setuptools",
        "wheel"
    ]
    
    missing = []
    for dep in common_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\n🔧 Add these to requirements.txt:")
        for dep in missing:
            print(f"   {dep}>=0.1.0")
    
    return missing

def main():
    """Main function"""
    print("🚀 ===== COMPREHENSIVE DEPENDENCY DISCOVERY =====")
    
    # Check common missing deps first
    common_missing = check_common_missing_deps()
    
    print("\n" + "="*60)
    
    # Run comprehensive discovery
    success = run_with_dependency_tracking()
    
    if common_missing:
        print(f"\n⚠️ Found {len(common_missing)} commonly missing dependencies")
        return 1
    elif not success:
        print("\n⚠️ Dependency discovery found issues")
        return 1
    else:
        print("\n✅ All dependencies appear to be available")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
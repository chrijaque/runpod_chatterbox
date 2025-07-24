#!/usr/bin/env python3
"""
Test Docker environment locally
Simulates the Docker build process to catch issues before building
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, capture_output=True):
    """Run a command and return result"""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, cwd=cwd, check=False)
        if capture_output:
            if result.stdout:
                print(f"📤 STDOUT: {result.stdout}")
            if result.stderr:
                print(f"📤 STDERR: {result.stderr}")
        print(f"📊 Exit code: {result.returncode}")
        return result
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return None

def create_test_environment():
    """Create a test environment similar to Docker"""
    print("🔧 Creating test environment...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="chatterbox_test_")
    print(f"📁 Test directory: {temp_dir}")
    
    # Copy requirements.txt
    shutil.copy("requirements.txt", os.path.join(temp_dir, "requirements.txt"))
    
    return temp_dir

def test_pip_install_process(temp_dir):
    """Test the pip installation process from Dockerfile"""
    print("\n🔧 ===== TESTING PIP INSTALLATION PROCESS =====")
    
    # Step 1: Test repository access
    print("\n📡 Step 1: Testing repository access...")
    result = run_command(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", 
                         "https://github.com/chrijaque/chatterbox_embed"])
    if result and result.returncode == 0:
        print("✅ Repository access successful")
    else:
        print("❌ Repository access failed")
        return False
    
    # Step 2: Install forked repository
    print("\n📦 Step 2: Installing forked repository...")
    result = run_command([
        "pip", "install", "--no-cache-dir", "--force-reinstall",
        "git+https://github.com/chrijaque/chatterbox_embed.git@master#egg=chatterbox-tts"
    ], cwd=temp_dir)
    
    if result and result.returncode == 0:
        print("✅ Forked repository installation successful")
    else:
        print("❌ Forked repository installation failed")
        return False
    
    # Step 3: Verify forked repository
    print("\n🔍 Step 3: Verifying forked repository...")
    result = run_command(["pip", "show", "chatterbox-tts"], cwd=temp_dir)
    if result and result.returncode == 0:
        if "chrijaque" in result.stdout.lower():
            print("✅ Forked repository confirmed")
        else:
            print("❌ Not using forked repository")
            return False
    else:
        print("❌ Could not verify repository")
        return False
    
    # Step 4: Install other requirements
    print("\n📦 Step 4: Installing other requirements...")
    result = run_command(["pip", "install", "-r", "requirements.txt"], cwd=temp_dir)
    
    if result and result.returncode == 0:
        print("✅ Other requirements installation successful")
    else:
        print("❌ Other requirements installation failed")
        return False
    
    return True

def test_runtime_imports(temp_dir):
    """Test that all runtime imports work"""
    print("\n🔧 ===== TESTING RUNTIME IMPORTS =====")
    
    # Create a test script
    test_script = """
import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} import successful")
        return True
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        return False

# Test critical imports
modules_to_test = [
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
    "numpy"
]

all_success = True
for module in modules_to_test:
    if not test_import(module):
        all_success = False

# Test chatterbox specific functionality
try:
    from chatterbox.tts import ChatterboxTTS
    print("✅ ChatterboxTTS import successful")
    
    has_load = hasattr(ChatterboxTTS, 'load_voice_profile')
    has_save = hasattr(ChatterboxTTS, 'save_voice_profile')
    
    print(f"✅ load_voice_profile method: {'✅' if has_load else '❌'}")
    print(f"✅ save_voice_profile method: {'✅' if has_save else '❌'}")
    
    if not has_load or not has_save:
        all_success = False
        
except Exception as e:
    print(f"❌ ChatterboxTTS test failed: {e}")
    all_success = False

sys.exit(0 if all_success else 1)
"""
    
    test_file = os.path.join(temp_dir, "test_imports.py")
    with open(test_file, "w") as f:
        f.write(test_script)
    
    # Run the test
    result = run_command([sys.executable, "test_imports.py"], cwd=temp_dir)
    
    return result and result.returncode == 0

def main():
    """Main test function"""
    print("🚀 ===== DOCKER ENVIRONMENT TEST =====")
    print("This simulates the Docker build process locally\n")
    
    # Create test environment
    temp_dir = create_test_environment()
    
    try:
        # Test pip installation process
        if not test_pip_install_process(temp_dir):
            print("\n❌ PIP INSTALLATION PROCESS FAILED")
            return 1
        
        # Test runtime imports
        if not test_runtime_imports(temp_dir):
            print("\n❌ RUNTIME IMPORTS FAILED")
            return 1
        
        print("\n✅ ALL TESTS PASSED - Docker build should work!")
        return 0
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
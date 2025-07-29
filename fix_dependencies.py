#!/usr/bin/env python3
"""
Dependency Fix Script
Helps resolve common dependency conflicts
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_current_versions():
    """Check current versions of problematic packages"""
    print("📋 Current Package Versions")
    print("=" * 40)
    
    packages = ['protobuf', 'onnx', 'torch', 'torchvision', 'torchaudio']
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Extract version
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        print(f"✅ {package}: {version}")
                        break
            else:
                print(f"❌ {package}: Not installed")
        except Exception as e:
            print(f"⚠️ {package}: Error checking ({e})")

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n🔧 Suggested Fixes")
    print("=" * 40)
    
    print("\n1. Fix Protobuf/ONNX Conflict:")
    print("   pip uninstall protobuf onnx -y")
    print("   pip install protobuf==3.20.3")
    print("   pip install onnx==1.14.0")
    
    print("\n2. Alternative: Use Original Base Image")
    print("   Change Dockerfile base image to:")
    print("   FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")
    
    print("\n3. Force Reinstall ChatterboxTTS:")
    print("   pip uninstall chatterbox-tts -y")
    print("   cd /workspace/chatterbox_embed")
    print("   pip install -e .")
    
    print("\n4. Check PyTorch Compatibility:")
    print("   python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'")
    print("   python -c 'import torchvision; print(f\"TorchVision: {torchvision.__version__}\")'")

def apply_fix_1():
    """Apply fix 1: Reinstall protobuf and onnx"""
    print("\n🔧 Applying Fix 1: Protobuf/ONNX Reinstall")
    print("=" * 50)
    
    steps = [
        ("pip uninstall protobuf onnx -y", "Uninstalling conflicting packages"),
        ("pip install protobuf==3.20.3", "Installing compatible protobuf"),
        ("pip install onnx==1.14.0", "Installing compatible onnx"),
        ("python -c 'import google.protobuf; print(f\"Protobuf: {google.protobuf.__version__}\")'", "Verifying protobuf"),
        ("python -c 'import onnx; print(f\"ONNX: {onnx.__version__}\")'", "Verifying onnx")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"❌ Failed at step: {desc}")
            return False
    
    print("\n✅ Fix 1 applied successfully!")
    return True

def apply_fix_2():
    """Apply fix 2: Reinstall ChatterboxTTS"""
    print("\n🔧 Applying Fix 2: ChatterboxTTS Reinstall")
    print("=" * 50)
    
    steps = [
        ("pip uninstall chatterbox-tts -y", "Uninstalling ChatterboxTTS"),
        ("cd /workspace/chatterbox_embed && pip install -e .", "Reinstalling ChatterboxTTS in editable mode"),
        ("python -c 'import chatterbox; print(f\"ChatterboxTTS: {chatterbox.__file__}\")'", "Verifying ChatterboxTTS")
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"❌ Failed at step: {desc}")
            return False
    
    print("\n✅ Fix 2 applied successfully!")
    return True

def test_imports():
    """Test critical imports after fixes"""
    print("\n🧪 Testing Critical Imports")
    print("=" * 40)
    
    tests = [
        ("import google.protobuf", "Protobuf"),
        ("from google.protobuf.internal import builder", "Protobuf builder"),
        ("import onnx", "ONNX"),
        ("from onnx.onnx_ml_pb2 import *", "ONNX ML protobuf"),
        ("import chatterbox", "ChatterboxTTS"),
        ("from chatterbox.tts import ChatterboxTTS", "ChatterboxTTS class")
    ]
    
    all_passed = True
    for test_cmd, desc in tests:
        try:
            exec(test_cmd)
            print(f"✅ {desc}: OK")
        except Exception as e:
            print(f"❌ {desc}: FAILED - {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main function"""
    print("🔧 Dependency Fix Script")
    print("=" * 40)
    
    # Check current state
    check_current_versions()
    
    # Show suggestions
    suggest_fixes()
    
    # Ask user what to do
    print("\n" + "=" * 40)
    print("Choose an option:")
    print("1. Apply Fix 1 (Protobuf/ONNX reinstall)")
    print("2. Apply Fix 2 (ChatterboxTTS reinstall)")
    print("3. Apply both fixes")
    print("4. Just test imports")
    print("5. Exit")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            apply_fix_1()
        elif choice == "2":
            apply_fix_2()
        elif choice == "3":
            apply_fix_1() and apply_fix_2()
        elif choice == "4":
            test_imports()
        elif choice == "5":
            print("Exiting...")
            return
        else:
            print("Invalid choice")
            return
        
        # Test imports after fixes
        if choice in ["1", "2", "3"]:
            print("\n🧪 Testing imports after fixes...")
            if test_imports():
                print("\n🎉 All tests passed! The system should work now.")
            else:
                print("\n❌ Some tests failed. Additional fixes may be needed.")
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main() 
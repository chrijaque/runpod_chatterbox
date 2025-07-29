#!/usr/bin/env python3
"""
Unified Installation Verification Script
Checks all dependencies, imports, and functionality for ChatterboxTTS + Higgs Audio
"""

import sys
import os
import subprocess
import importlib
import traceback
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section"""
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    print(f"✅ Platform: {sys.platform}")
    print(f"✅ Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")

def check_system_dependencies():
    """Check system dependencies"""
    print_section("System Dependencies")
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            # Extract version
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
        else:
            print("❌ FFmpeg is not working properly")
    except FileNotFoundError:
        print("❌ FFmpeg is not installed")
    
    # Check CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers are available")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
                    break
        else:
            print("❌ NVIDIA drivers are not working properly")
    except FileNotFoundError:
        print("❌ NVIDIA drivers are not available")

def check_python_packages():
    """Check Python package versions"""
    print_section("Python Package Versions")
    
    packages = [
        'torch', 'torchaudio', 'torchvision',
        'numpy', 'scipy', 'librosa', 'soundfile',
        'transformers', 'accelerate', 'datasets', 'tokenizers',
        'firebase_admin', 'google.cloud.storage',
        'pydub', 'nltk', 'requests',
        'protobuf', 'onnx', 's3tokenizer'
    ]
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: Not installed ({e})")
        except Exception as e:
            print(f"⚠️ {package}: Error checking version ({e})")

def check_protobuf_compatibility():
    """Check protobuf compatibility"""
    print_section("Protobuf Compatibility")
    
    try:
        import google.protobuf
        print(f"✅ Protobuf version: {google.protobuf.__version__}")
        
        # Check if builder is available
        try:
            from google.protobuf.internal import builder
            print("✅ Protobuf builder module is available")
        except ImportError:
            print("❌ Protobuf builder module is NOT available")
            print("   This is likely the cause of ONNX import issues")
        
        # Check protobuf internal structure
        try:
            import google.protobuf.internal
            print("✅ Protobuf internal module is available")
            print(f"   Available in internal: {dir(google.protobuf.internal)[:10]}...")
        except ImportError as e:
            print(f"❌ Protobuf internal module error: {e}")
            
    except ImportError as e:
        print(f"❌ Protobuf not installed: {e}")

def check_onnx_compatibility():
    """Check ONNX compatibility"""
    print_section("ONNX Compatibility")
    
    try:
        import onnx
        print(f"✅ ONNX version: {onnx.__version__}")
        
        # Test ONNX import chain
        try:
            from onnx.onnx_pb import *
            print("✅ ONNX protobuf imports work")
        except ImportError as e:
            print(f"❌ ONNX protobuf imports failed: {e}")
            
        # Test the specific import that's failing
        try:
            from onnx.onnx_ml_pb2 import *
            print("✅ ONNX ML protobuf imports work")
        except ImportError as e:
            print(f"❌ ONNX ML protobuf imports failed: {e}")
            print("   This is the exact error we're seeing in the logs")
            
    except ImportError as e:
        print(f"❌ ONNX not installed: {e}")

def check_chatterbox_installation():
    """Check ChatterboxTTS installation"""
    print_section("ChatterboxTTS Installation")
    
    try:
        import chatterbox
        print(f"✅ ChatterboxTTS imported from: {chatterbox.__file__}")
        
        # Check if it's editable install
        if 'site-packages' not in chatterbox.__file__:
            print("✅ ChatterboxTTS is installed in editable mode")
        else:
            print("⚠️ ChatterboxTTS is installed in site-packages")
        
        # Test key imports
        try:
            from chatterbox.tts import ChatterboxTTS
            print("✅ ChatterboxTTS class imported successfully")
        except ImportError as e:
            print(f"❌ ChatterboxTTS class import failed: {e}")
            
        try:
            from chatterbox.vc import ChatterboxVC
            print("✅ ChatterboxVC class imported successfully")
        except ImportError as e:
            print(f"❌ ChatterboxVC class import failed: {e}")
            
    except ImportError as e:
        print(f"❌ ChatterboxTTS not installed: {e}")

def check_higgs_installation():
    """Check Higgs Audio installation"""
    print_section("Higgs Audio Installation")
    
    try:
        import boson_multimodal
        print(f"✅ Higgs Audio imported from: {boson_multimodal.__file__}")
        
        # Test key imports
        try:
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            print("✅ HiggsAudioServeEngine imported successfully")
        except ImportError as e:
            print(f"❌ HiggsAudioServeEngine import failed: {e}")
            
        try:
            from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
            print("✅ Higgs Audio data types imported successfully")
        except ImportError as e:
            print(f"❌ Higgs Audio data types import failed: {e}")
            
    except ImportError as e:
        print(f"❌ Higgs Audio not installed: {e}")

def check_handler_imports():
    """Check if handlers can be imported"""
    print_section("Handler Imports")
    
    # Add handler paths to sys.path
    sys.path.append('/app/handlers/chatterbox')
    sys.path.append('/app/handlers/higgs')
    
    # Test ChatterboxTTS handler
    try:
        from handlers.chatterbox.vc_handler import handler as chatterbox_vc_handler
        print("✅ ChatterboxTTS VC handler imported successfully")
    except ImportError as e:
        print(f"❌ ChatterboxTTS VC handler import failed: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
    
    try:
        from handlers.chatterbox.tts_handler import handler as chatterbox_tts_handler
        print("✅ ChatterboxTTS TTS handler imported successfully")
    except ImportError as e:
        print(f"❌ ChatterboxTTS TTS handler import failed: {e}")
    
    # Test Higgs handler
    try:
        from handlers.higgs.vc_handler import handler as higgs_vc_handler
        print("✅ Higgs Audio VC handler imported successfully")
    except ImportError as e:
        print(f"❌ Higgs Audio VC handler import failed: {e}")
    
    try:
        from handlers.higgs.tts_handler import handler as higgs_tts_handler
        print("✅ Higgs Audio TTS handler imported successfully")
    except ImportError as e:
        print(f"❌ Higgs Audio TTS handler import failed: {e}")

def check_unified_handler():
    """Check unified handler functionality"""
    print_section("Unified Handler")
    
    try:
        import unified_vc_handler
        print("✅ Unified VC handler imported successfully")
        
        # Test the get_model_handler function
        if hasattr(unified_vc_handler, 'get_model_handler'):
            print("✅ get_model_handler function exists")
            
            # Test with chatterbox
            try:
                handler = unified_vc_handler.get_model_handler('chatterbox')
                if handler:
                    print("✅ ChatterboxTTS handler routing works")
                else:
                    print("❌ ChatterboxTTS handler routing failed")
            except Exception as e:
                print(f"❌ ChatterboxTTS handler routing error: {e}")
                
            # Test with higgs
            try:
                handler = unified_vc_handler.get_model_handler('higgs')
                if handler:
                    print("✅ Higgs Audio handler routing works")
                else:
                    print("❌ Higgs Audio handler routing failed")
            except Exception as e:
                print(f"❌ Higgs Audio handler routing error: {e}")
        else:
            print("❌ get_model_handler function not found")
            
    except ImportError as e:
        print(f"❌ Unified handler import failed: {e}")

def check_directories():
    """Check required directories"""
    print_section("Directory Structure")
    
    directories = [
        '/app/handlers/chatterbox',
        '/app/handlers/higgs',
        '/voice_profiles',
        '/voice_samples',
        '/temp_voice',
        '/tts_generated'
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory} exists")
            # List contents
            try:
                contents = list(path.iterdir())
                print(f"   Contents: {len(contents)} items")
            except PermissionError:
                print("   Contents: Permission denied")
        else:
            print(f"❌ {directory} does not exist")

def check_environment():
    """Check environment variables"""
    print_section("Environment Variables")
    
    env_vars = [
        'PYTHONPATH',
        'PYTHONUNBUFFERED',
        'HF_HOME',
        'RUNPOD_SECRET_Firebase',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
            if var in ['RUNPOD_SECRET_Firebase', 'GOOGLE_APPLICATION_CREDENTIALS']:
                print(f"   Value: [REDACTED]")
            else:
                print(f"   Value: {value}")
        else:
            print(f"❌ {var} is not set")

def run_comprehensive_check():
    """Run all checks"""
    print_header("Unified Installation Verification")
    print("This script checks all dependencies and functionality for the unified setup.")
    
    check_python_version()
    check_system_dependencies()
    check_python_packages()
    check_protobuf_compatibility()
    check_onnx_compatibility()
    check_chatterbox_installation()
    check_higgs_installation()
    check_handler_imports()
    check_unified_handler()
    check_directories()
    check_environment()
    
    print_header("Verification Complete")
    print("Review the output above to identify any issues.")
    print("Common issues and solutions:")
    print("1. Protobuf/ONNX conflicts: Try different protobuf versions")
    print("2. Missing dependencies: Install required packages")
    print("3. Import errors: Check module installation and paths")
    print("4. Permission issues: Check directory permissions")

if __name__ == "__main__":
    run_comprehensive_check() 
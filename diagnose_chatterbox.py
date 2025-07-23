#!/usr/bin/env python3
"""
Diagnostic script to identify chatterbox installation issues
Run this in the RunPod container to debug installation problems
"""

import sys
import os
import subprocess

def diagnose_chatterbox():
    print("🔍 Chatterbox Installation Diagnostic")
    print("=" * 50)
    
    # Test 1: Check Python environment
    print("🐍 Python Environment:")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python path: {sys.path[:3]}...")
    
    # Test 2: Check pip installation
    print(f"\n📦 Pip Installation:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        chatterbox_packages = [line for line in result.stdout.split('\n') 
                             if 'chatterbox' in line.lower()]
        print(f"   Found chatterbox packages: {chatterbox_packages}")
    except Exception as e:
        print(f"   ❌ Error checking pip: {e}")
    
    # Test 3: Check chatterbox import
    print(f"\n📦 Chatterbox Import:")
    try:
        import chatterbox
        print(f"   ✅ chatterbox imported successfully")
        print(f"   📂 Module path: {chatterbox.__file__}")
        print(f"   📁 Module directory: {os.path.dirname(chatterbox.__file__)}")
        
        # Check if it's a git repository
        repo_path = os.path.dirname(chatterbox.__file__)
        git_path = os.path.join(repo_path, '.git')
        
        if os.path.exists(git_path):
            print(f"   🔍 Found .git directory: {git_path}")
            try:
                commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD'], 
                                                    text=True).strip()
                print(f"   🔢 Git commit: {commit_hash}")
                
                remote_url = subprocess.check_output(['git', '-C', repo_path, 'remote', 'get-url', 'origin'], 
                                                   text=True).strip()
                print(f"   🌐 Git remote: {remote_url}")
                
                if 'chrijaque/chatterbox_embed' in remote_url:
                    print(f"   ✅ This is the CORRECT forked repository!")
                else:
                    print(f"   ❌ This is NOT the forked repository!")
                    
            except Exception as e:
                print(f"   ⚠️ Could not get git info: {e}")
        else:
            print(f"   ❌ No .git directory found - this is a PyPI installation")
            
    except ImportError as e:
        print(f"   ❌ Failed to import chatterbox: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error importing chatterbox: {e}")
        return False
    
    # Test 4: Check ChatterboxTTS class
    print(f"\n🧠 ChatterboxTTS Class:")
    try:
        from chatterbox.tts import ChatterboxTTS
        print(f"   ✅ ChatterboxTTS imported successfully")
        
        # Check available methods
        model = ChatterboxTTS.from_pretrained(device='cpu')
        print(f"   ✅ Model loaded successfully")
        
        # Check for voice profile methods
        has_save_profile = hasattr(model, 'save_voice_profile')
        has_load_profile = hasattr(model, 'load_voice_profile')
        has_save_clone = hasattr(model, 'save_voice_clone')
        has_load_clone = hasattr(model, 'load_voice_clone')
        
        print(f"   🔍 Voice Profile Methods:")
        print(f"      save_voice_profile: {'✅' if has_save_profile else '❌'}")
        print(f"      load_voice_profile: {'✅' if has_load_profile else '❌'}")
        print(f"      save_voice_clone: {'✅' if has_save_clone else '❌'}")
        print(f"      load_voice_clone: {'✅' if has_load_clone else '❌'}")
        
    except Exception as e:
        print(f"   ❌ Error with ChatterboxTTS: {e}")
        return False
    
    # Test 5: Check pip show details
    print(f"\n📋 Pip Package Details:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'chatterbox-tts'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   📦 Package info:\n{result.stdout}")
        else:
            print(f"   ❌ Package not found: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error getting package info: {e}")
    
    # Test 6: Check for dependency conflicts
    print(f"\n🔍 Dependency Analysis:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        all_packages = result.stdout.split('\n')
        
        # Look for packages that might conflict
        conflict_packages = []
        for line in all_packages:
            if any(pkg in line.lower() for pkg in ['chatterbox', 'resemble', 'perth']):
                conflict_packages.append(line.strip())
        
        if conflict_packages:
            print(f"   🔍 Potentially conflicting packages:")
            for pkg in conflict_packages:
                print(f"      {pkg}")
        else:
            print(f"   ✅ No obvious conflicts found")
            
    except Exception as e:
        print(f"   ❌ Error analyzing dependencies: {e}")
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY:")
    print(f"=" * 30)
    
    is_forked = 'chrijaque/chatterbox_embed' in str(repo_path) if 'repo_path' in locals() else False
    has_profile_methods = has_save_profile and has_load_profile if 'has_save_profile' in locals() else False
    
    print(f"   Repository: {'✅ Forked' if is_forked else '❌ Original'}")
    print(f"   Voice Profile Support: {'✅ Yes' if has_profile_methods else '❌ No'}")
    print(f"   Installation Type: {'✅ Git' if os.path.exists(git_path) else '❌ PyPI'}")
    
    if is_forked and has_profile_methods:
        print(f"   🎉 SUCCESS: Everything looks good!")
        return True
    else:
        print(f"   ❌ ISSUE: Installation problem detected")
        return False

if __name__ == "__main__":
    success = diagnose_chatterbox()
    sys.exit(0 if success else 1) 
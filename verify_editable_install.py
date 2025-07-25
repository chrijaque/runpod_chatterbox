#!/usr/bin/env python3
"""
Verification script to check if the editable install is working correctly.
This should be run inside the RunPod container to verify the installation.
"""

import os
import sys
import subprocess

def main():
    print("🔍 ===== EDITABLE INSTALL VERIFICATION =====")
    
    # 1. Check pip package info
    print("\n📦 Pip package info:")
    try:
        result = subprocess.run(['pip', 'show', 'chatterbox-tts'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting pip info: {e}")
    
    # 2. Check if it's an editable install
    print("\n📂 Editable install check:")
    try:
        import chatterbox
        print(f"📂 chatterbox.__file__: {chatterbox.__file__}")
        is_editable = 'site-packages' not in chatterbox.__file__
        print(f"📂 Is editable install: {is_editable}")
        
        if is_editable:
            print("✅ Using editable install (should have latest code)")
        else:
            print("⚠️ Using site-packages install (may have old wheel)")
            
    except ImportError as e:
        print(f"❌ Error importing chatterbox: {e}")
    
    # 3. Check s3gen module
    print("\n🔧 S3Gen module check:")
    try:
        import chatterbox.models.s3gen.s3gen as s3gen
        print(f"📂 s3gen.__file__: {s3gen.__file__}")
        
        # Check if it's from the forked repository
        if 'chatterbox_embed' in s3gen.__file__:
            print("🎯 s3gen module is from FORKED repository")
        else:
            print("⚠️ s3gen module is from ORIGINAL repository")
            
        # Check if inference_from_text exists
        has_inference_from_text = hasattr(s3gen.S3Token2Wav, 'inference_from_text')
        print(f"📂 Has inference_from_text method: {has_inference_from_text}")
        
        if has_inference_from_text:
            print("✅ inference_from_text method exists!")
        else:
            print("❌ inference_from_text method does NOT exist")
            
    except ImportError as e:
        print(f"❌ Error importing s3gen: {e}")
    
    # 4. Check available methods on S3Token2Wav
    print("\n📋 Available S3Token2Wav methods:")
    try:
        import chatterbox.models.s3gen.s3gen as s3gen
        methods = [method for method in dir(s3gen.S3Token2Wav) if not method.startswith('_')]
        print(f"📋 Total methods: {len(methods)}")
        
        # Check for key methods
        key_methods = ['inference_from_text', 'inference', 'generate', 'embed_ref']
        for method in key_methods:
            has_method = method in methods
            status = "✅" if has_method else "❌"
            print(f"{status} {method}: {has_method}")
            
    except ImportError as e:
        print(f"❌ Error checking methods: {e}")
    
    # 5. Check git repository info
    print("\n🌐 Git repository check:")
    try:
        repo_path = os.path.dirname(chatterbox.__file__)
        git_path = os.path.join(repo_path, '.git')
        
        if os.path.exists(git_path):
            print(f"📁 Found git repository at: {repo_path}")
            try:
                commit_hash = subprocess.run(['git', '-C', repo_path, 'rev-parse', 'HEAD'], 
                                           capture_output=True, text=True, check=True)
                print(f"🔢 Git commit: {commit_hash.stdout.strip()}")
                
                remote_url = subprocess.run(['git', '-C', repo_path, 'remote', 'get-url', 'origin'], 
                                          capture_output=True, text=True, check=True)
                print(f"🌐 Git remote: {remote_url.stdout.strip()}")
                
                if 'chrijaque/chatterbox_embed' in remote_url.stdout:
                    print("✅ Git confirms forked repository")
                else:
                    print("⚠️ Git shows original repository")
                    
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Could not get git info: {e}")
        else:
            print("📁 No git repository found (PyPI package installation)")
            
    except Exception as e:
        print(f"⚠️ Could not check git info: {e}")
    
    print("\n🔍 ===== END VERIFICATION =====")

if __name__ == "__main__":
    main() 
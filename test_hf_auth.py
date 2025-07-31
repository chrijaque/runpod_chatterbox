#!/usr/bin/env python3
"""
Test script to verify HuggingFace authentication
"""

import os
import sys
from huggingface_hub import snapshot_download

def test_hf_auth():
    """Test HuggingFace authentication with different methods"""
    
    print("🔍 Testing HuggingFace Authentication...")
    
    # Method 1: Check environment variable
    hf_token_env = os.environ.get('HF_TOKEN')
    print(f"🔑 HF_TOKEN from environment: {'✅ Set' if hf_token_env else '❌ Not set'}")
    if hf_token_env:
        print(f"   Length: {len(hf_token_env)}")
        print(f"   Preview: {hf_token_env[:10]}...")
    
    # Method 2: Check HUGGING_FACE_HUB_TOKEN
    hf_hub_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    print(f"🔑 HUGGING_FACE_HUB_TOKEN: {'✅ Set' if hf_hub_token else '❌ Not set'}")
    
    # Method 3: Check if huggingface_hub can authenticate
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get user info
        user_info = api.whoami()
        print(f"✅ HuggingFace authentication successful!")
        print(f"   User: {user_info.get('name', 'Unknown')}")
        print(f"   Email: {user_info.get('email', 'Unknown')}")
        
        # Test model access
        print("\n🔍 Testing model access...")
        try:
            # Try to access the Higgs model
            model_info = api.model_info("bosonai/higgs-audio-v2-generation-3B-base")
            print(f"✅ Model access successful!")
            print(f"   Model: {model_info.modelId}")
            print(f"   Private: {model_info.private}")
            print(f"   Downloads: {model_info.downloads}")
            
            # Test download
            print("\n🔍 Testing model download...")
            test_dir = "/tmp/test_hf_download"
            snapshot_download(
                repo_id="bosonai/higgs-audio-v2-generation-3B-base",
                local_dir=test_dir,
                max_workers=1
            )
            print(f"✅ Model download successful!")
            print(f"   Downloaded to: {test_dir}")
            
        except Exception as e:
            print(f"❌ Model access failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ HuggingFace authentication failed: {str(e)}")
        
        # Try with explicit token
        if hf_token_env:
            print(f"\n🔍 Trying with explicit token...")
            try:
                api = HfApi(token=hf_token_env)
                user_info = api.whoami()
                print(f"✅ Authentication with explicit token successful!")
                print(f"   User: {user_info.get('name', 'Unknown')}")
            except Exception as e2:
                print(f"❌ Authentication with explicit token failed: {str(e2)}")
    
    print("\n📋 Environment variables:")
    for key, value in os.environ.items():
        if 'HF' in key or 'HUGGING' in key:
            print(f"   {key}: {'✅ Set' if value else '❌ Not set'}")

if __name__ == "__main__":
    test_hf_auth() 
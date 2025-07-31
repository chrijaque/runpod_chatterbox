#!/usr/bin/env python3
"""
Test script to check if Higgs models are public
"""

import requests
from huggingface_hub import HfApi

def test_public_models():
    """Test if Higgs models are publicly accessible"""
    
    print("🔍 Testing Higgs model accessibility...")
    
    models = [
        "bosonai/higgs-audio-v2-generation-3B-base",
        "bosonai/higgs-audio-v2-tokenizer", 
        "bosonai/hubert_base"
    ]
    
    for model in models:
        print(f"\n🔍 Testing model: {model}")
        
        # Method 1: Direct HTTP request
        try:
            url = f"https://huggingface.co/api/models/{model}"
            response = requests.get(url)
            print(f"   HTTP Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Public: {not data.get('private', True)}")
                print(f"   📊 Downloads: {data.get('downloads', 0)}")
            else:
                print(f"   ❌ HTTP Error: {response.text}")
        except Exception as e:
            print(f"   ❌ HTTP Request failed: {str(e)}")
        
        # Method 2: HuggingFace Hub API
        try:
            api = HfApi()
            model_info = api.model_info(model)
            print(f"   ✅ Hub API: {model_info.modelId}")
            print(f"   📊 Private: {model_info.private}")
            print(f"   📊 Downloads: {model_info.downloads}")
        except Exception as e:
            print(f"   ❌ Hub API failed: {str(e)}")

if __name__ == "__main__":
    test_public_models() 
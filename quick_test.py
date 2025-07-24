#!/usr/bin/env python3
"""
Quick test to check current environment dependencies
"""

import sys

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

print("🔍 Quick dependency test...")

modules = [
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

all_good = True
for module in modules:
    if not test_import(module):
        all_good = False

print(f"\n{'✅ All good!' if all_good else '❌ Issues found'}")
sys.exit(0 if all_good else 1) 
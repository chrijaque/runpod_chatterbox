#!/usr/bin/env python3
"""
Check for locally saved TTS files
"""

import os
from pathlib import Path

def check_local_tts_files():
    """Check for TTS files in the local directory"""
    print("🔍 Checking for local TTS files...")
    
    local_dir = Path("./tts_generated")
    
    if not local_dir.exists():
        print(f"❌ Local TTS directory not found: {local_dir}")
        print("💡 TTS files will be saved here when generated")
        return
    
    files = list(local_dir.glob("*.wav"))
    
    if not files:
        print(f"📂 Local TTS directory exists but no files found: {local_dir}")
        print("💡 TTS files will appear here after generation")
        return
    
    print(f"✅ Found {len(files)} local TTS files:")
    
    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        file_size = file.stat().st_size
        file_time = file.stat().st_mtime
        import datetime
        file_date = datetime.datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"  📄 {file.name}")
        print(f"     📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"     🕒 Created: {file_date}")
        print()

def create_sample_local_file():
    """Create a sample local directory structure"""
    print("🔧 Creating sample local TTS directory...")
    
    local_dir = Path("./tts_generated")
    local_dir.mkdir(exist_ok=True)
    
    # Create a sample file to show the structure
    sample_file = local_dir / "sample_tts_file.wav"
    
    if not sample_file.exists():
        # Create a minimal WAV file for demonstration
        import wave
        import struct
        
        with wave.open(str(sample_file), 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Sample rate
            
            # Write some silence (1 second)
            silence = struct.pack('<h', 0) * 22050
            wav_file.writeframes(silence)
        
        print(f"✅ Created sample file: {sample_file}")
    else:
        print(f"✅ Sample file already exists: {sample_file}")

if __name__ == "__main__":
    print("🧪 Local TTS File Checker")
    print("=" * 40)
    
    # Check for existing files
    check_local_tts_files()
    
    print("\n" + "=" * 40)
    print("🎯 Next Steps:")
    print("1. Generate a TTS file in RunPod")
    print("2. Check this script again to see the local file")
    print("3. The file will be saved in ./tts_generated/")
    
    # Create sample directory if it doesn't exist
    print("\n🔧 Setting up local directory...")
    create_sample_local_file() 
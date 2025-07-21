from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from pathlib import Path
import base64
import json
import os
from typing import List, Dict, Any, Optional

# Import builtins explicitly to satisfy strict linter
import builtins
print = builtins.print
list = builtins.list
max = builtins.max
str = builtins.str
len = builtins.len
open = builtins.open
Exception = builtins.Exception

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Local directory paths (same as in rp_handler.py)
VOICE_CLONES_DIR = Path("./voice_clones")
VOICE_SAMPLES_DIR = Path("./voice_samples")
TEMP_VOICE_DIR = Path("./temp_voice")

# Create directories if they don't exist
VOICE_CLONES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

def get_voice_library() -> List[Dict[str, Any]]:
    """Get list of all created voices with their sample files"""
    voices: List[Dict[str, Any]] = []
    
    try:
        # Check if directories exist
        if not VOICE_CLONES_DIR.exists() or not VOICE_SAMPLES_DIR.exists():
            print("Voice directories don't exist yet")
            return voices
        
        # Get all .npy files (voice embeddings)
        embedding_files = list(VOICE_CLONES_DIR.glob("*.npy"))
        
        for embedding_file in embedding_files:
            # Extract voice_id from filename (remove .npy extension)
            voice_id: str = embedding_file.stem
            
            # Find corresponding sample files
            sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
            
            if sample_files:
                # Get the most recent sample file
                latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
                
                # Extract name from voice_id (remove voice_ prefix)
                display_name: str = voice_id.replace("voice_", "").replace("_", " ").title()
                
                voice_info: Dict[str, Any] = {
                    "voice_id": voice_id,
                    "name": display_name,
                    "sample_file": str(latest_sample),
                    "embedding_file": str(embedding_file),
                    "created_date": latest_sample.stat().st_mtime
                }
                voices.append(voice_info)
                
        # Sort by creation date (newest first)
        voices.sort(key=lambda x: x["created_date"], reverse=True)
        
        print(f"Found {len(voices)} voices in library")
        
    except Exception as e:
        print(f"Error getting voice library: {e}")
    
    return voices

@app.route('/api/voices', methods=['GET'])
def list_voices() -> Dict[str, Any]:
    """Get voice library"""
    try:
        voices = get_voice_library()
        return jsonify({
            "status": "success",
            "voices": voices,
            "total_voices": len(voices)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/voices/<voice_id>/sample', methods=['GET'])
def get_voice_sample(voice_id: str) -> Any:
    """Get voice sample audio file"""
    try:
        # Find the sample file
        sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
        if not sample_files:
            return jsonify({
                "status": "error",
                "message": f"No sample found for voice_id: {voice_id}"
            }), 404
        
        # Get the most recent sample
        latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
        
        # Return the audio file directly
        return send_file(latest_sample, mimetype='audio/wav')
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/voices/<voice_id>/sample/base64', methods=['GET'])
def get_voice_sample_base64(voice_id: str) -> Dict[str, Any]:
    """Get voice sample as base64 (for compatibility)"""
    try:
        # Find the sample file
        sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
        if not sample_files:
            return jsonify({
                "status": "error",
                "message": f"No sample found for voice_id: {voice_id}"
            }), 404
        
        # Get the most recent sample
        latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
        
        # Read and encode the audio file
        with open(latest_sample, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "voice_id": voice_id,
            "audio_base64": audio_data,
            "sample_file": str(latest_sample)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "voice-library-api"})

if __name__ == '__main__':
    print("🎵 Voice Library API Server Starting...")
    print(f"📂 Voice Clones Directory: {VOICE_CLONES_DIR.absolute()}")
    print(f"📂 Voice Samples Directory: {VOICE_SAMPLES_DIR.absolute()}")
    print(f"📂 Temp Voice Directory: {TEMP_VOICE_DIR.absolute()}")
    print("🌐 API will be available at: http://localhost:5001")
    app.run(debug=True, port=5001) 
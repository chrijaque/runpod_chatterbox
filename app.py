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

# Local directory paths (same as in rp_handler.py but for local development)
VOICE_CLONES_DIR = Path("./voice_clones")
VOICE_SAMPLES_DIR = Path("./voice_samples")
TEMP_VOICE_DIR = Path("./temp_voice")

# Create directories if they don't exist (local development only)
print(f"🔍 Checking local directories...")
VOICE_CLONES_DIR.mkdir(exist_ok=True)
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
TEMP_VOICE_DIR.mkdir(exist_ok=True)

print(f"✅ Local directories ready:")
print(f"  VOICE_CLONES_DIR: {VOICE_CLONES_DIR.absolute()}")
print(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR.absolute()}")
print(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR.absolute()}")

def get_voice_library() -> List[Dict[str, Any]]:
    """Get list of all created voices with their sample files"""
    voices: List[Dict[str, Any]] = []
    
    try:
        # Check if directories exist
        if not VOICE_CLONES_DIR.exists() or not VOICE_SAMPLES_DIR.exists():
            print("Voice directories don't exist yet")
            return voices
        
        # Get all voice files (.npy embeddings or .json metadata)
        embedding_files = list(VOICE_CLONES_DIR.glob("*.npy"))
        metadata_files = list(VOICE_CLONES_DIR.glob("*.json"))
        
        # Combine and deduplicate voice IDs
        voice_ids = set()
        for file in embedding_files:
            voice_ids.add(file.stem)
        for file in metadata_files:
            voice_ids.add(file.stem)
        
        for voice_id in voice_ids:
            # Find corresponding sample files
            sample_files = list(VOICE_SAMPLES_DIR.glob(f"{voice_id}_sample_*.wav"))
            
            if sample_files:
                # Get the most recent sample file
                latest_sample = max(sample_files, key=lambda f: f.stat().st_mtime)
                
                # Extract name from voice_id (remove voice_ prefix)
                display_name: str = voice_id.replace("voice_", "").replace("_", " ").title()
                
                # Check if we have embedding or metadata file
                embedding_file = VOICE_CLONES_DIR / f"{voice_id}.npy"
                metadata_file = VOICE_CLONES_DIR / f"{voice_id}.json"
                
                embedding_path = str(embedding_file) if embedding_file.exists() else str(metadata_file) if metadata_file.exists() else ""
                
                voice_info: Dict[str, Any] = {
                    "voice_id": voice_id,
                    "name": display_name,
                    "sample_file": str(latest_sample),
                    "embedding_file": embedding_path,
                    "created_date": latest_sample.stat().st_mtime,
                    "has_embedding": embedding_file.exists(),
                    "has_metadata": metadata_file.exists()
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

@app.route('/api/voices/save', methods=['POST'])
def save_voice_locally() -> Dict[str, Any]:
    """Save voice files locally from RunPod generation"""
    try:
        voice_id = request.form.get('voice_id')
        voice_name = request.form.get('voice_name')
        audio_file = request.files.get('audio_file')
        template_message = request.form.get('template_message', '')
        
        if not all([voice_id, voice_name, audio_file]):
            return jsonify({
                "status": "error",
                "message": "Missing required fields: voice_id, voice_name, audio_file"
            }), 400
            
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save audio sample
        sample_filename = VOICE_SAMPLES_DIR / f"{voice_id}_sample_{timestamp}.wav"
        audio_file.save(sample_filename)
        
        # Create a placeholder embedding file (since we can't get the actual embedding from RunPod)
        embedding_filename = VOICE_CLONES_DIR / f"{voice_id}.npy"
        if not embedding_filename.exists():
            # Create a metadata file instead of trying to recreate the embedding
            metadata = {
                "voice_id": voice_id,
                "voice_name": voice_name,
                "template_message": template_message,
                "created_timestamp": timestamp,
                "source": "runpod_generation"
            }
            
            import json
            with open(embedding_filename.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved voice files locally:")
        print(f"   Audio: {sample_filename}")
        print(f"   Metadata: {embedding_filename.with_suffix('.json')}")
        
        return jsonify({
            "status": "success",
            "message": f"Voice files saved locally for {voice_name}",
            "voice_id": voice_id,
            "sample_file": str(sample_filename),
            "metadata_file": str(embedding_filename.with_suffix('.json'))
        })
        
    except Exception as e:
        print(f"Error saving voice files locally: {e}")
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
# RunPod Chatterbox Voice Cloning

A voice cloning application using Chatterbox TTS with persistent voice embeddings, featuring a local voice library.

## Architecture

- **Frontend**: Next.js/React app for voice recording, library management, and TTS generation
- **Local API**: Flask server for voice library, TTS library, and audio file serving  
- **Voice Cloning Handler**: RunPod serverless handler for creating voice embeddings
- **TTS Generation Handler**: RunPod serverless handler for text-to-speech generation
- **Local Storage**: Voice embeddings (`.npy`), audio samples (`.wav`), and TTS generations (`.wav`)

### Handler Separation

- **Voice Cloning** (`rp_handler.py`): Creates voice embeddings from audio samples
- **TTS Generation** (`tts_handler.py`): Uses saved embeddings to generate speech
- **Separate Endpoints**: Each handler has its own RunPod endpoint for easier debugging

## Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### 2. Start Local API Server

The local API serves the voice library and audio files:

```bash
python app.py
```

This will start the Flask server at `http://localhost:5001` and create the local directories:
- `./voice_clones/` - Voice embedding files (`.npy`)
- `./voice_samples/` - Generated audio samples (`.wav`) 
- `./temp_voice/` - Temporary audio files

### 3. Configure RunPod

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_RUNPOD_API_KEY=your_runpod_api_key
NEXT_PUBLIC_RUNPOD_ENDPOINT_ID=your_voice_clone_endpoint_id
NEXT_PUBLIC_TTS_ENDPOINT_ID=your_tts_endpoint_id
```

**Note**: You'll need to deploy both handlers separately on RunPod:
- Voice Cloning Handler: Use `rp_handler.py` and `Dockerfile`
- TTS Generation Handler: Use `tts_handler.py` and `Dockerfile.tts`

### 4. Start Frontend

```bash
cd frontend
npm run dev
```

Visit `http://localhost:3000`

## Usage

### Creating Voice Clones

1. Enter a name for the voice clone
2. Record or upload audio sample
3. Click "Clone Voice"
4. RunPod generates the voice and saves files locally

### Voice Library

- **Automatic**: Library updates after successful voice creation
- **Manual**: Click "Refresh" to reload from local files
- **Playback**: Click "Play Sample" to hear voice samples

## API Endpoints

### Local API (Flask - Port 5001)

```
GET  /api/voices                    # List voice library
GET  /api/voices/{id}/sample        # Get audio file
GET  /api/voices/{id}/sample/base64 # Get base64 audio
GET  /health                        # Health check
```

### RunPod APIs

```
# Voice Cloning Handler
POST /run  # Generate voice clone

# TTS Generation Handler  
POST /run  # Generate TTS with saved voice
```

## File Structure

```
voice_clones/           # Voice embeddings
├── voice_emma.npy
└── voice_christian.npy

voice_samples/          # Audio samples  
├── voice_emma_sample_20240120_143022.wav
└── voice_christian_sample_20240120_143155.wav

tts_generated/          # TTS generations
├── tts_voice_emma_20240120_143022.wav
└── tts_voice_christian_20240120_143155.wav

temp_voice/             # Temporary files
```

## Development

The application separates concerns:

- **RunPod**: Heavy TTS processing with CUDA
- **Local API**: Fast file serving and library management
- **Frontend**: User interface and interaction

This design minimizes RunPod API calls and provides instant library access.

# Voice Library API - Modular FastAPI Structure

A production-ready voice cloning and TTS API built with FastAPI, featuring Firebase integration and modular architecture.

## 🏗️ Project Structure

```
api-app/
├── app/
│   ├── main.py               ← FastAPI application initialization
│   ├── api/
│   │   ├── voices.py         ← Voice management endpoints
│   │   ├── tts.py            ← TTS generation endpoints
│   │   └── health.py         ← Health and debug endpoints
│   ├── services/
│   │   ├── firebase.py       ← Firebase Storage operations
│   │   └── runpod_client.py  ← RunPod API client
│   ├── models/
│   │   └── schemas.py        ← Pydantic models and schemas
│   └── config.py             ← Application configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start_fastapi.py
├── test_fastapi.py
└── README.md
```

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Voice Cloning**: Create and manage voice profiles using RunPod
- **TTS Generation**: Generate speech from text using cloned voices
- **Firebase Integration**: Cloud storage for voice files and TTS generations
- **Production Ready**: Docker support, health checks, and comprehensive logging
- **API Documentation**: Auto-generated OpenAPI documentation

## 📋 API Endpoints

### Voice Management
- `GET /api/voices` - List all voices
- `POST /api/voices/clone` - Create voice clone
- `POST /api/voices/save` - Save voice files locally
- `GET /api/voices/{voice_id}/sample` - Get voice sample audio
- `GET /api/voices/{voice_id}/sample/base64` - Get voice sample as base64
- `GET /api/voices/{voice_id}/profile` - Get voice profile

### TTS Generation
- `GET /api/tts/generations` - List TTS generations
- `POST /api/tts/generate` - Generate TTS
- `POST /api/tts/save` - Save TTS generation
- `GET /api/tts/generations/{file_id}/audio` - Get TTS audio file

### System
- `GET /health` - Health check
- `GET /api/debug/directories` - Debug directory status
- `GET /docs` - API documentation

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker (optional)
- Firebase credentials (`firebase_creds.json`)

### Local Development

1. **Clone and setup**:
   ```bash
   cd api-app
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   # For voice library display (optional - uses firebase_local_only.json automatically)
   export FIREBASE_STORAGE_BUCKET="your-project-id.firebasestorage.app"
   
   # For RunPod API access (required for voice cloning/TTS)
   export RUNPOD_API_KEY="your-runpod-api-key"
   export VC_CB_ENDPOINT_ID="your-chatterbox-voice-clone-endpoint"
   export TTS_CB_ENDPOINT_ID="your-chatterbox-tts-endpoint"
   ```

3. **Start the server**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Optional: Setup Firebase for voice library** (to see existing voices):
   ```bash
   chmod +x setup_local_dev.sh
   ./setup_local_dev.sh
   # Follow the instructions to add Firebase credentials to .env
   ```

**Note**: 
- **Voice cloning/TTS**: Uses RunPod's own `RUNPOD_SECRET_Firebase` secrets (no local setup needed)
- **Voice library display**: Uses `firebase_local_only.json` file automatically to show existing voices

### Docker Deployment

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**:
   ```bash
   docker build -t voice-library-api .
   docker run -p 8000:8000 voice-library-api
   ```

## 🧪 Testing

Run the test suite:
```bash
python test_fastapi.py
```

## 📁 Directory Structure

The API manages these directories:
- `voice_profiles/` - Voice profile files (.npy, .json)
- `voice_samples/` - Voice sample audio files
- `temp_voice/` - Temporary voice processing files
- `tts_generated/` - Generated TTS audio files

## 🔧 Configuration

Key configuration options in `app/config.py`:
- `FIREBASE_STORAGE_BUCKET` - Firebase Storage bucket name
- `API_HOST` - API server host (default: 0.0.0.0)
- `API_PORT` - API server port (default: 8000)
- `CORS_ORIGINS` - Allowed CORS origins
- `MAX_AUDIO_FILE_SIZE` - Maximum audio file size (50MB)
- `MAX_PROFILE_FILE_SIZE` - Maximum profile file size (10MB)

## 🔐 Environment Variables

Required environment variables:
- `FIREBASE_STORAGE_BUCKET` - Firebase Storage bucket
- `RUNPOD_API_KEY` - RunPod API key
- `VC_CB_ENDPOINT_ID` - ChatterboxTTS Voice cloning endpoint ID
- `TTS_CB_ENDPOINT_ID` - ChatterboxTTS TTS generation endpoint ID


Optional:
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: False)
- `LOCAL_STORAGE_ENABLED` - Enable local storage (default: True)
- `FIREBASE_STORAGE_ENABLED` - Enable Firebase storage (default: True)

## 📚 API Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔄 Workflow

### Voice Cloning
1. Upload audio file → `POST /api/voices/clone`
2. RunPod processes the audio
3. Save results → `POST /api/voices/save`
4. Files stored locally and in Firebase

### TTS Generation
1. Select voice and text → `POST /api/tts/generate`
2. RunPod generates TTS
3. Save audio → `POST /api/tts/save`
4. Audio stored locally and in Firebase

## 🚀 Production Deployment

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Manual Docker
```bash
docker build -t voice-library-api .
docker run -d \
  -p 8000:8000 \
  -v ./voice_profiles:/app/voice_profiles \
  -v ./voice_samples:/app/voice_samples \
  -v ./firebase_creds.json:/app/firebase_creds.json:ro \
  -e FIREBASE_STORAGE_BUCKET=your-bucket \
  -e RUNPOD_API_KEY=your-key \
  voice-library-api
```

## 🔍 Monitoring

- **Health Check**: `GET /health`
- **Directory Status**: `GET /api/debug/directories`
- **Logs**: Check application logs for detailed information

## 🤝 Contributing

1. Follow the modular structure
2. Add tests for new features
3. Update documentation
4. Use type hints and Pydantic models

## 📄 License

This project is part of the Voice Library API system. 
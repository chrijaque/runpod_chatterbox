# Voice Library API - RunPod Chatterbox Integration

A production-ready voice cloning and TTS API built with FastAPI, featuring Firebase integration and RunPod ML inference.

## 🏗️ Project Structure

```
runpod_chatterbox/
├── api-app/                    ← **NEW: Modular FastAPI Application**
│   ├── app/
│   │   ├── main.py            ← FastAPI application initialization
│   │   ├── api/
│   │   │   ├── voices.py      ← Voice management endpoints
│   │   │   ├── tts.py         ← TTS generation endpoints
│   │   │   └── health.py      ← Health and debug endpoints
│   │   ├── services/
│   │   │   ├── firebase.py    ← Firebase Storage operations
│   │   │   └── runpod_client.py ← RunPod API client
│   │   ├── models/
│   │   │   └── schemas.py     ← Pydantic models and schemas
│   │   └── config.py          ← Application configuration
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
├── frontend/                   ← Next.js Frontend Application
├── vc_handler.py              ← RunPod Voice Cloning Handler
├── tts_handler.py             ← RunPod TTS Handler
├── firebase_creds.json        ← Firebase Credentials
└── README.md                  ← This file
```

## 🚀 Quick Start

### 1. Start the FastAPI Server

```bash
cd api-app
python start_fastapi.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at: http://localhost:3000

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

## 🔧 Configuration

### Environment Variables

Required for the FastAPI server (`api-app/`):
```bash
export FIREBASE_STORAGE_BUCKET="your-project-id.appspot.com"
export RUNPOD_API_KEY="your-runpod-api-key"
export RUNPOD_ENDPOINT_ID="your-voice-clone-endpoint"
export TTS_ENDPOINT_ID="your-tts-endpoint"
```

### Firebase Setup

1. Place your `firebase_creds.json` in the root directory
2. Update the bucket name in environment variables
3. The API will automatically upload files to Firebase Storage

## 🧪 Testing

### Test the FastAPI Server
```bash
cd api-app
python test_fastapi.py
```

### Test RunPod Handlers
```bash
# Test voice cloning
python vc_handler.py

# Test TTS generation
python tts_handler.py
```

## 🐳 Docker Deployment

### FastAPI Server
```bash
cd api-app
docker-compose up --build
```

### RunPod Handlers
```bash
# Build voice cloning handler
docker build -f Dockerfile.vc -t voice-clone-handler .

# Build TTS handler
docker build -f Dockerfile.tts -t tts-handler .
```

## 📁 Directory Structure

The application manages these directories:
- `voice_profiles/` - Voice profile files (.npy, .json)
- `voice_samples/` - Voice sample audio files
- `temp_voice/` - Temporary voice processing files
- `tts_generated/` - Generated TTS audio files

## 🔄 Workflow

### Voice Cloning
1. Upload audio file → Frontend → FastAPI → RunPod
2. RunPod processes the audio and returns voice profile
3. Save results locally and to Firebase
4. Voice available for TTS generation

### TTS Generation
1. Select voice and text → Frontend → FastAPI → RunPod
2. RunPod generates TTS audio
3. Save audio locally and to Firebase
4. Audio available for playback and download

## 🚀 Production Deployment

### FastAPI Server (Recommended)
```bash
cd api-app
docker-compose up -d
```

### RunPod Handlers
Deploy to RunPod.io with the provided Dockerfiles and configuration files.

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs
- **Frontend**: Next.js application with TypeScript
- **RunPod Handlers**: Python handlers for ML inference
- **Firebase**: Cloud storage for file persistence

## 🤝 Contributing

1. Follow the modular structure in `api-app/`
2. Add tests for new features
3. Update documentation
4. Use type hints and Pydantic models

## 📄 License

This project is part of the Voice Library API system.

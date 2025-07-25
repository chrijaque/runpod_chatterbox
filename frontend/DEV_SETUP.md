# 🚀 Development Setup Guide

## **Port Configuration**

To avoid port conflicts, we use the following setup:

- **FastAPI Server**: `http://localhost:8000`
- **Frontend (Next.js)**: `http://localhost:3000`

## **Quick Start**

### **1. Start FastAPI Server**
```bash
cd api-app
python start_fastapi.py
```

### **2. Start Frontend (in a new terminal)**
```bash
cd frontend
npm run dev
```

### **3. Access the Applications**
- **Frontend**: http://localhost:3000
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## **Environment Configuration**

The frontend is configured to connect to the FastAPI server via the `.env.local` file:

```env
NEXT_PUBLIC_FASTAPI_URL=http://localhost:8000
NEXT_PUBLIC_RUNPOD_API_KEY=your_runpod_api_key
NEXT_PUBLIC_RUNPOD_ENDPOINT_ID=your_voice_clone_endpoint_id
NEXT_PUBLIC_TTS_ENDPOINT_ID=your_tts_endpoint_id
```

## **Testing the Integration**

### **Manual Testing**
1. Open http://localhost:3000
2. Test voice cloning functionality
3. Test TTS generation
4. Verify files are saved to Firebase (not locally)

## **Troubleshooting**

### **Port Already in Use**
If you get port conflicts:
- **Port 3000**: Usually another Next.js app or development server
- **Port 8000**: Usually the FastAPI server (should be running)

### **Check Running Processes**
```bash
# Check what's running on port 3000
lsof -i :3000

# Check what's running on port 8000
lsof -i :8000
```

### **Kill Conflicting Processes**
```bash
# Kill process on port 3000 (if needed)
kill -9 $(lsof -t -i:3000)

# Kill process on port 8000 (if needed)
kill -9 $(lsof -t -i:8000)
```

## **Development Workflow**

1. **Start FastAPI server first** (port 8000)
2. **Start frontend second** (port 3000)
3. **Test integration** through the frontend UI
4. **Check Firebase storage** for uploaded files
5. **Monitor logs** in both terminal windows

## **Firebase Integration**

The setup uses Firebase for all storage:
- **Voice files**: `gs://bucket/audio/voices/{language}/`
- **Story files**: `gs://bucket/audio/stories/{language}/{type}/`
- **No local storage**: Everything goes to Firebase

## **API Endpoints**

### **Voice Management**
- `GET /api/voices/languages` - List available languages
- `GET /api/voices/by-language/{language}` - List voices by language
- `POST /api/voices/clone` - Create voice clone
- `GET /api/voices/{voice_id}/firebase-urls` - Get Firebase URLs

### **TTS Generation**
- `POST /api/tts/generate` - Generate TTS
- `GET /api/tts/stories/languages` - List story languages
- `GET /api/tts/stories/{language}` - List stories by language

### **Health & Debug**
- `GET /health` - Health check
- `GET /api/debug/directories` - Debug Firebase storage

## **Next Steps**

1. **Test voice cloning** through the frontend
2. **Test TTS generation** through the frontend
3. **Verify Firebase storage** organization
4. **Check shared access** between apps
5. **Deploy to production** when ready 
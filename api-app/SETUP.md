# API Setup Guide

## 🔧 Environment Configuration

### 1. Create Environment File

Create a `.env` file in the `api-app/` directory with the following variables:

```bash
# RunPod Configuration
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_voice_clone_endpoint_id_here
TTS_ENDPOINT_ID=your_tts_endpoint_id_here

# Firebase Configuration
FIREBASE_STORAGE_BUCKET=godnathistorie-a25fa.appspot.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
LOCAL_STORAGE_ENABLED=True
FIREBASE_STORAGE_ENABLED=True
```

### 2. Get Your RunPod Credentials

1. **RunPod API Key**: 
   - Go to https://runpod.io/console/user/settings
   - Copy your API key

2. **Voice Clone Endpoint ID**:
   - Deploy `vc_handler.py` to RunPod
   - Copy the endpoint ID from the RunPod console

3. **TTS Endpoint ID**:
   - Deploy `tts_handler.py` to RunPod  
   - Copy the endpoint ID from the RunPod console

### 3. Firebase Setup

1. **Firebase Credentials**: 
   - The `firebase_creds.json` file should already be in the `api-app/` directory
   - This contains your Firebase service account credentials

2. **Firebase Bucket**:
   - Your bucket name is: `godnathistorie-a25fa.appspot.com`
   - This is already configured in the `.env` file

## 🚀 Installation

### 1. Install Dependencies

```bash
cd api-app
pip install -r requirements.txt
```

### 2. Test Configuration

```bash
python test_fastapi.py
```

This will show you what's missing and what's working.

### 3. Start the Server

```bash
python start_fastapi.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### 1. Build and Run

```bash
cd api-app
docker-compose up --build
```

### 2. Environment Variables for Docker

You can also set environment variables directly in `docker-compose.yml`:

```yaml
environment:
  - RUNPOD_API_KEY=your_key_here
  - RUNPOD_ENDPOINT_ID=your_endpoint_here
  - TTS_ENDPOINT_ID=your_tts_endpoint_here
  - FIREBASE_STORAGE_BUCKET=godnathistorie-a25fa.appspot.com
```

## 🔍 Troubleshooting

### Common Issues

1. **"Missing configuration" error**:
   - Make sure your `.env` file exists in the `api-app/` directory
   - Check that all required variables are set

2. **Firebase connection failed**:
   - Ensure `firebase_creds.json` is in the `api-app/` directory
   - Verify the bucket name is correct

3. **RunPod API errors**:
   - Check your API key is correct
   - Verify endpoint IDs are valid
   - Ensure endpoints are running on RunPod

### Test Commands

```bash
# Test configuration
python test_fastapi.py

# Test health endpoint
curl http://localhost:8000/health

# Test voice library
curl http://localhost:8000/api/voices

# Test API documentation
curl http://localhost:8000/docs
```

## 📋 Required Files

Make sure these files exist in `api-app/`:

- ✅ `.env` - Environment variables
- ✅ `firebase_creds.json` - Firebase credentials  
- ✅ `requirements.txt` - Python dependencies
- ✅ `app/` - Application code
- ✅ `voice_profiles/` - Voice profile directory
- ✅ `voice_samples/` - Voice sample directory

## 🎯 Next Steps

1. **Set up your RunPod endpoints** using the handlers in the root directory
2. **Update your frontend** to use port 8000 instead of 5001
3. **Test the full workflow** with voice cloning and TTS generation
4. **Deploy to production** using Docker Compose 
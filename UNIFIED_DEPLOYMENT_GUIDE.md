# Unified Deployment Guide: ChatterboxTTS + Higgs Audio

## 🎯 Overview

This guide covers deploying the unified system that supports both ChatterboxTTS and Higgs Audio models in a single Docker image with frontend model selection.

## 📋 Prerequisites

- RunPod account with GPU access
- Docker registry (Docker Hub, GitHub Container Registry, etc.)
- Firebase project configured
- Environment variables set up

## 🚀 Deployment Steps

### 1. Build Docker Images

```bash
# Make build script executable
chmod +x build_unified.sh

# Build unified images
./build_unified.sh
```

### 2. Push to Registry

```bash
# Tag images for your registry
docker tag runpod-chatterbox-unified-vc your-registry/runpod-chatterbox-unified-vc:latest
docker tag runpod-chatterbox-unified-tts your-registry/runpod-chatterbox-unified-tts:latest

# Push to registry
docker push your-registry/runpod-chatterbox-unified-vc:latest
docker push your-registry/runpod-chatterbox-unified-tts:latest
```

### 3. Configure RunPod Endpoints

#### Voice Cloning Endpoint
- **Image**: `your-registry/runpod-chatterbox-unified-vc:latest`
- **Handler**: `/app/unified_vc_handler.py`
- **Environment Variables**:
  - `RUNPOD_SECRET_Firebase`: Firebase credentials JSON
  - `FIREBASE_STORAGE_BUCKET`: Your Firebase bucket name

#### TTS Endpoint
- **Image**: `your-registry/runpod-chatterbox-unified-tts:latest`
- **Handler**: `/app/unified_tts_handler.py`
- **Environment Variables**:
  - `RUNPOD_SECRET_Firebase`: Firebase credentials JSON
  - `FIREBASE_STORAGE_BUCKET`: Your Firebase bucket name

### 4. Update API Configuration

Update your `.env` file in `api-app/`:

```env
# RunPod Configuration
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_voice_endpoint_id
TTS_ENDPOINT_ID=your_tts_endpoint_id

# Firebase Configuration
FIREBASE_CREDENTIALS_FILE=firebase_creds.json
FIREBASE_STORAGE_BUCKET=your-bucket-name.firebasestorage.app
```

### 5. Deploy Frontend

```bash
# Build frontend
cd frontend
npm run build

# Deploy to your hosting platform (Vercel, Netlify, etc.)
```

## 🧪 Testing

### 1. Test API Endpoints

```bash
# Start the API server
cd api-app
python -m uvicorn app.main:app --reload

# Run tests
python ../test_unified_setup.py
```

### 2. Test Frontend

1. Navigate to your deployed frontend
2. Try voice cloning with both models:
   - Select "ChatterboxTTS" and create a voice
   - Select "Higgs Audio" and create a voice
3. Try TTS generation with both models
4. Verify files are saved to Firebase with correct metadata

## 🔧 Model Comparison

### ChatterboxTTS
- **Pros**: Fast, efficient, smaller model size
- **Best for**: Real-time applications, quick voice cloning
- **Use case**: When speed is priority over expressiveness

### Higgs Audio
- **Pros**: More expressive, better long-form capabilities
- **Best for**: High-quality TTS, audiobooks, podcasts
- **Use case**: When quality and expressiveness are priority

## 📊 Performance Monitoring

### API Response Format

Both models return responses with `model_type` field:

```json
{
  "status": "success",
  "voice_id": "test_voice",
  "profile_path": "audio/voices/en/profiles/test_voice_20250101_120000.npy",
  "sample_path": "audio/voices/en/samples/test_voice_sample_20250101_120000.mp3",
  "model_type": "chatterbox"  // or "higgs"
}
```

### Firebase Metadata

Files are tagged with model information:

```json
{
  "voice_id": "test_voice",
  "model": "chatterbox",  // or "higgs_audio_v2"
  "created_date": "1704067200",
  "language": "en",
  "is_kids_voice": "False"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not switching**: Check `model_type` parameter in API calls
2. **Handler not found**: Verify Docker image includes unified handlers
3. **Import errors**: Ensure all dependencies in `requirements/unified.txt`
4. **Firebase upload fails**: Check credentials and bucket permissions

### Debug Commands

```bash
# Check Docker image contents
docker run --rm your-registry/runpod-chatterbox-unified-vc ls -la /app/

# Test handler directly
docker run --rm your-registry/runpod-chatterbox-unified-vc python /app/unified_vc_handler.py

# Check logs
docker logs <container_id>
```

## 📈 Scaling Considerations

### Resource Requirements

- **ChatterboxTTS**: ~8GB GPU memory
- **Higgs Audio**: ~24GB GPU memory
- **Unified Image**: ~30GB (includes both models)

### Cost Optimization

1. **Separate Endpoints**: Use different endpoints for different models
2. **Auto-scaling**: Scale based on model type demand
3. **Caching**: Cache voice profiles to reduce regeneration

## 🔄 Migration from Separate Models

If you have existing separate ChatterboxTTS and Higgs Audio deployments:

1. **Backup existing data**: Export voice profiles and TTS generations
2. **Deploy unified system**: Follow steps above
3. **Test thoroughly**: Ensure all functionality works
4. **Update frontend**: Deploy new frontend with model toggle
5. **Monitor performance**: Compare results between models

## 📚 Additional Resources

- [ChatterboxTTS Documentation](https://github.com/your-chatterbox-repo)
- [Higgs Audio Documentation](https://github.com/chrijaque/higgs-audio)
- [RunPod Documentation](https://docs.runpod.io/)
- [Firebase Storage Documentation](https://firebase.google.com/docs/storage)

## 🎉 Success Metrics

- ✅ Both models working in single Docker image
- ✅ Frontend toggle switching between models
- ✅ Files saved to Firebase with correct metadata
- ✅ API responses include model type
- ✅ No performance degradation vs separate deployments 
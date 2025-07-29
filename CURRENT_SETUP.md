# Current Setup Summary

## 🎯 Cleaned Up Structure

After removing Higgs and unified files, here's what we have:

### **📁 Dockerfiles**
- `dockerfiles/lightweight/` - Production Dockerfiles for network volume deployment
- `dockerfiles/chatterbox/` - Original ChatterboxTTS Dockerfiles (kept for reference)

### **📁 Requirements**
- `requirements/lightweight.txt` - Production requirements for lightweight images
- `requirements/chatterbox.txt` - Original ChatterboxTTS requirements
- `requirements/chatterbox_only.txt` - ChatterboxTTS-only requirements

### **📁 Handlers**
- `handlers/chatterbox/` - ChatterboxTTS handlers
- `handlers/higgs/` - Higgs Audio handlers (using network volume models)
- `handlers/unified_*.py` - Unified handlers that route to both models

## 🚀 Production Deployment

### **Current Approach:**
- ✅ **Optimized Docker images** (~4-6GB with pre-loaded ChatterboxTTS)
- ✅ **Network volume** for Higgs Audio models (`dtabrd8bbd:/runpod-volume`)
- ✅ **Pre-loaded ChatterboxTTS** models (no runtime initialization)
- ✅ **Unified handlers** that support both ChatterboxTTS and Higgs Audio
- ✅ **GitHub Container Registry** for image storage

### **Model Loading Strategy:**
- **ChatterboxTTS:** Pre-loaded during Docker build (no runtime initialization)
- **Higgs Audio:** Loaded from network volume at runtime
  - Generation: `/runpod-volume/higgs_audio_generation`
  - Tokenizer: `/runpod-volume/higgs_audio_tokenizer`
  - HuBERT: `/runpod-volume/hubert_base`

## 🛠️ Build Commands

```bash
# Build lightweight images
chmod +x build_lightweight.sh
./build_lightweight.sh

# Or build manually
docker build -f dockerfiles/lightweight/Dockerfile.vc -t runpod-chatterbox-lightweight-vc .
docker build -f dockerfiles/lightweight/Dockerfile.tts -t runpod-chatterbox-lightweight-tts .
```

## 🚀 Deploy Commands

```bash
# Deploy with network volume
runpod endpoint create \
  --name unified-vc-handler \
  --image ghcr.io/YOUR_USERNAME/runpod-chatterbox-lightweight-vc:latest \
  --volume dtabrd8bbd:/runpod-volume

runpod endpoint create \
  --name unified-tts-handler \
  --image ghcr.io/YOUR_USERNAME/runpod-chatterbox-lightweight-tts:latest \
  --volume dtabrd8bbd:/runpod-volume
```

## ✅ Benefits of This Setup

1. **Optimized Performance** - No runtime model initialization for ChatterboxTTS
2. **Cost Effective** - Avoids repeated GPU initialization costs
3. **Network Volume** - Higgs Audio models persist across deployments
4. **Unified Handlers** - Single endpoint for both models
5. **Clean Structure** - Removed unused files
6. **Production Ready** - Optimized for cost and performance

## 🎯 Next Steps

1. Build the lightweight images
2. Push to GitHub Container Registry
3. Deploy with your network volume
4. Test both ChatterboxTTS and Higgs Audio models 
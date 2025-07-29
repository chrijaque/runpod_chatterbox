# Network Volume Deployment Guide

This guide explains how to deploy the unified TTS/VC system using a network volume for models.

## 🎯 Benefits

- **Faster Builds**: No model download during Docker build
- **Smaller Images**: Lightweight Docker images (~2-3GB vs 8-10GB)
- **Faster Deployments**: No model download delays
- **Cost Efficient**: Network volumes are cheaper than larger images
- **Easy Updates**: Update models without rebuilding Docker image

## 📋 Prerequisites

1. RunPod account with network volume support
2. Access to RunPod CLI or web interface

## 🚀 Step-by-Step Deployment

### Step 1: Use Existing Network Volume

You already have a network volume set up:
- **Volume ID**: `hlm3wqzffe`
- **Volume Name**: "Higgs Audio"

This volume will be used to store the Higgs Audio models.

### Step 2: Build and Push to GitHub Container Registry

```bash
# Set up GitHub token (one-time setup)
export GITHUB_TOKEN=your_github_personal_access_token

# Use the GitHub setup script
chmod +x setup_github_registry.sh
./setup_github_registry.sh YOUR_GITHUB_USERNAME

# Or do it manually:
# 1. Login to GitHub Container Registry
docker login ghcr.io -u YOUR_GITHUB_USERNAME -p YOUR_GITHUB_TOKEN

# 2. Build images
chmod +x build_lightweight.sh
./build_lightweight.sh

# 3. Tag for GitHub Container Registry
docker tag runpod-chatterbox-lightweight-vc ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest
docker tag runpod-chatterbox-lightweight-tts ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest

# 4. Push to GitHub Container Registry
docker push ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest
docker push ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest
```

### Step 3: Push Images to Registry

```bash
# Tag and push to your registry
docker tag runpod-chatterbox-unified-vc your-registry/runpod-chatterbox-unified-vc:latest
docker tag runpod-chatterbox-unified-tts your-registry/runpod-chatterbox-unified-tts:latest

docker push your-registry/runpod-chatterbox-unified-vc:latest
docker push your-registry/runpod-chatterbox-unified-tts:latest
```

### Step 4: Set Up Network Volume with Models

Create a temporary pod to download models to the network volume:

```bash
# Create a temporary pod for model download
runpod pod create \
  --name "model-downloader" \
  --image "ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest" \
  --volume "hlm3wqzffe:/models" \
  --env "DOWNLOAD_MODELS=true"

# Execute model download script
runpod pod exec model-downloader -- python /app/setup_network_volume.py

# Delete temporary pod
runpod pod delete model-downloader
```

### Step 5: Deploy Handlers with Network Volume

#### VC Handler Deployment

```bash
runpod endpoint create \
  --name "unified-vc-handler" \
  --image "ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest" \
  --volume "dtabrd8bbd:/runpod-volume" \
  --env "HANDLER_PATH=/app/unified_vc_handler.py" \
  --env "RUNPOD_SECRET_Firebase=your-firebase-credentials" \
  --env "FIREBASE_STORAGE_BUCKET=your-bucket-name"
```

#### TTS Handler Deployment

```bash
runpod endpoint create \
  --name "unified-tts-handler" \
  --image "ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest" \
  --volume "dtabrd8bbd:/runpod-volume" \
  --env "HANDLER_PATH=/app/unified_tts_handler.py" \
  --env "RUNPOD_SECRET_Firebase=your-firebase-credentials" \
  --env "FIREBASE_STORAGE_BUCKET=your-bucket-name"
```

## 🔧 Configuration

### Environment Variables

- `RUNPOD_SECRET_Firebase`: Firebase service account credentials (JSON)
- `FIREBASE_STORAGE_BUCKET`: Firebase storage bucket name
- `HANDLER_PATH`: Path to the unified handler script

### Volume Mounts

- `hlm3wqzffe:/models`: Network volume containing Higgs Audio models

## 📊 Performance Comparison

| Metric | With Network Volume | Without Network Volume |
|--------|-------------------|----------------------|
| **Build Time** | ~5 minutes | ~15-20 minutes |
| **Image Size** | ~2-3GB | ~8-10GB |
| **Deployment Time** | ~30 seconds | ~2-3 minutes |
| **Startup Time** | ~10 seconds | ~1-2 minutes |
| **Storage Cost** | $0.10/GB/month | $0.10/GB/month |

## 🔄 Model Updates

To update models in the network volume:

```bash
# Create temporary pod with network volume
runpod pod create \
  --name "model-updater" \
  --image "ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest" \
  --volume "hlm3wqzffe:/models"

# Run update script
runpod pod exec model-updater -- python /app/setup_network_volume.py

# Delete temporary pod
runpod pod delete model-updater
```

## 🛠️ Troubleshooting

### Models Not Found

If handlers can't find models:

1. Check network volume is mounted: `ls /models`
2. Verify models exist: `ls /models/hub/`
3. Re-run setup script: `python /app/setup_network_volume.py`

### Permission Issues

If there are permission issues:

```bash
# Check volume permissions
ls -la /models

# Fix permissions if needed
chmod -R 755 /models
```

### Space Issues

If network volume is full:

```bash
# Check space usage
df -h /models

# Clean up if needed
rm -rf /models/hub/*/snapshots/*/tmp
```

## 📝 Notes

- Network volumes persist across pod restarts
- Models are shared between all pods using the same volume
- Volume costs are separate from pod costs
- Consider using different volumes for different model versions 
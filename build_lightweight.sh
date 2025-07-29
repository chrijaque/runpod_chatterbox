#!/bin/bash

# Lightweight Docker Build Script for Network Volume Deployment
# This script builds lightweight Docker images that use network volumes for models

set -e

echo "🚀 Building Lightweight Docker Images for Network Volume Deployment"
echo "📦 Images will be ~2-3GB instead of 8-10GB"
echo "🔗 Models will be mounted from network volume at runtime"
echo ""

# Build VC Handler Image
echo "🔧 Building VC Handler Image..."
docker build -f dockerfiles/lightweight/Dockerfile.vc -t runpod-chatterbox-lightweight-vc .
echo "✅ VC Handler Image built successfully"

# Build TTS Handler Image
echo "🔧 Building TTS Handler Image..."
docker build -f dockerfiles/lightweight/Dockerfile.tts -t runpod-chatterbox-lightweight-tts .
echo "✅ TTS Handler Image built successfully"

# Show image sizes
echo ""
echo "📊 Image Sizes:"
echo "VC Handler:"
docker images runpod-chatterbox-lightweight-vc --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""
echo "TTS Handler:"
docker images runpod-chatterbox-lightweight-tts --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "🎯 Next Steps:"
echo "1. Login to GitHub Container Registry:"
echo "   docker login ghcr.io -u YOUR_GITHUB_USERNAME -p YOUR_GITHUB_TOKEN"
echo ""
echo "2. Push images to GitHub Container Registry:"
echo "   docker tag runpod-chatterbox-lightweight-vc ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest"
echo "   docker tag runpod-chatterbox-lightweight-tts ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest"
echo "   docker push ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest"
echo "   docker push ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest"
echo ""
echo "3. Deploy handlers with network volume:"
echo "   runpod endpoint create --name unified-vc-handler --image ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-vc:latest --volume dtabrd8bbd:/runpod-volume"
echo "   runpod endpoint create --name unified-tts-handler --image ghcr.io/YOUR_GITHUB_USERNAME/runpod-chatterbox-lightweight-tts:latest --volume dtabrd8bbd:/runpod-volume"
echo ""
echo "✅ Lightweight build completed successfully!" 
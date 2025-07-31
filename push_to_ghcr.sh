#!/bin/bash

# Push Docker images to GitHub Container Registry
# Usage: ./push_to_ghcr.sh

set -e

echo "🚀 Pushing Docker images to GitHub Container Registry..."

# Login to GitHub Container Registry using RunPod secrets
echo "🔑 Logging in to GitHub Container Registry..."
echo $DOCKER_PASSWORD | docker login ghcr.io -u $DOCKER_USERNAME --password-stdin

# Push Higgs Audio VC Handler
echo "📤 Pushing Higgs Audio VC Handler..."
docker push ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest

# Push Higgs Audio TTS Handler
echo "📤 Pushing Higgs Audio TTS Handler..."
docker push ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest

echo "✅ All images pushed successfully!"
echo "📋 Images available at:"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest"
echo ""
echo "🎯 Next step: Deploy to RunPod using these image URLs"
echo "   RunPod will use DOCKER_USERNAME and DOCKER_PASSWORD secrets for authentication" 
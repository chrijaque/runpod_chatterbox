#!/bin/bash

# Local build script for Higgs Audio Docker images
# Usage: ./build_local.sh

set -e

echo "🔧 Building Higgs Audio Docker images locally..."

# Check if HUGGINGFACE_ACCESS_TOKEN is set
if [ -z "$HUGGINGFACE_ACCESS_TOKEN" ]; then
    echo "❌ Error: HUGGINGFACE_ACCESS_TOKEN environment variable is not set"
    echo "Please set your HuggingFace token:"
    echo "export HUGGINGFACE_ACCESS_TOKEN=your_token_here"
    exit 1
fi

echo "✅ HUGGINGFACE_ACCESS_TOKEN is set"

# Build Higgs Audio VC Handler
echo "🔧 Building Higgs Audio VC Handler..."
docker build \
    --build-arg HF_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
    -f dockerfiles/higgs/Dockerfile.vc \
    -t ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest \
    .

echo "✅ Higgs Audio VC Handler built successfully"

# Build Higgs Audio TTS Handler
echo "🔧 Building Higgs Audio TTS Handler..."
docker build \
    --build-arg HF_TOKEN="$HUGGINGFACE_ACCESS_TOKEN" \
    -f dockerfiles/higgs/Dockerfile.tts \
    -t ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest \
    .

echo "✅ Higgs Audio TTS Handler built successfully"

echo "🎉 All Higgs Audio Docker images built successfully!"
echo "📋 Images created:"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest"
echo ""
echo "🚀 Next step: Push to GitHub Container Registry"
echo "   docker push ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest"
echo "   docker push ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest" 
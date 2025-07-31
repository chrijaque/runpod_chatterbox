#!/bin/bash

# Build script for Higgs Audio with Network Volume
# This creates lightweight images that mount models from Network Volume

set -e

echo "🔧 Building Higgs Audio Docker images with Network Volume..."

# Check if we're on a system with Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found locally"
    echo "🔧 Using GitHub Actions to build images..."
    echo ""
    echo "📤 Push to GitHub to trigger build:"
    echo "   git add ."
    echo "   git commit -m 'Use Network Volume for Higgs Audio models'"
    echo "   git push"
    echo ""
    echo "🎯 Then deploy to RunPod using:"
    echo "   - runpod.higgs.vc.toml"
    echo "   - runpod.higgs.tts.toml"
    exit 0
fi

echo "✅ Docker found, building locally..."

# Build Higgs Audio VC Handler
echo "🔧 Building Higgs Audio VC Handler..."
docker build \
    -f dockerfiles/higgs/Dockerfile.vc \
    -t ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest \
    .

echo "✅ Higgs Audio VC Handler built successfully"

# Build Higgs Audio TTS Handler
echo "🔧 Building Higgs Audio TTS Handler..."
docker build \
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
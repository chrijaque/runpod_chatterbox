#!/bin/bash

# Unified Build Script for ChatterboxTTS + Higgs Audio
# This script builds Docker images that support both models

set -e

echo "🚀 ===== UNIFIED BUILD SCRIPT ====="
echo "Building Docker images for ChatterboxTTS + Higgs Audio..."

# Check if we're in the right directory
if [ ! -f "dockerfiles/unified/Dockerfile.vc" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Build Voice Cloning image
echo "🔨 Building unified voice cloning image..."
docker build -f dockerfiles/unified/Dockerfile.vc -t runpod-chatterbox-unified-vc .

if [ $? -eq 0 ]; then
    echo "✅ Voice cloning image built successfully"
else
    echo "❌ Failed to build voice cloning image"
    exit 1
fi

# Build TTS image
echo "🔨 Building unified TTS image..."
docker build -f dockerfiles/unified/Dockerfile.tts -t runpod-chatterbox-unified-tts .

if [ $? -eq 0 ]; then
    echo "✅ TTS image built successfully"
else
    echo "❌ Failed to build TTS image"
    exit 1
fi

echo ""
echo "🎉 ===== BUILD COMPLETED ====="
echo ""
echo "📦 Available images:"
echo "   - runpod-chatterbox-unified-vc (Voice Cloning)"
echo "   - runpod-chatterbox-unified-tts (TTS Generation)"
echo ""
echo "🧪 ===== VERIFICATION ====="
echo "To verify the installation, run:"
echo "   docker run --rm runpod-chatterbox-unified-vc python /app/verify_unified_installation.py"
echo "   docker run --rm runpod-chatterbox-unified-tts python /app/verify_unified_installation.py"
echo ""
echo "🔧 To deploy to RunPod:"
echo "   1. Tag and push to your registry"
echo "   2. Update RunPod endpoint configurations"
echo "   3. Set model_type parameter in API calls"
echo ""
echo "🎯 Model types supported:"
echo "   - chatterbox (ChatterboxTTS)"
echo "   - higgs (Higgs Audio)"
echo ""
echo "💡 The unified handlers will automatically route to the correct model"
echo "   based on the model_type parameter in the API request." 
#!/bin/bash

# Complete workflow: Build → Push → Deploy Higgs Audio
# Usage: ./deploy_higgs_workflow.sh

set -e

echo "🚀 Higgs Audio Deployment Workflow"
echo "=================================="

# Check if required environment variables are set
if [ -z "$HUGGINGFACE_ACCESS_TOKEN" ]; then
    echo "❌ Error: HUGGINGFACE_ACCESS_TOKEN environment variable is not set"
    echo "Please set your HuggingFace token:"
    echo "export HUGGINGFACE_ACCESS_TOKEN=your_token_here"
    exit 1
fi

if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
    echo "❌ Error: DOCKER_USERNAME and DOCKER_PASSWORD environment variables are not set"
    echo "Please set your Docker credentials:"
    echo "export DOCKER_USERNAME=your_username"
    echo "export DOCKER_PASSWORD=your_password"
    exit 1
fi

echo "✅ All required environment variables are set"

# Step 1: Build Docker images locally
echo ""
echo "🔧 Step 1: Building Docker images locally..."
chmod +x build_local.sh
./build_local.sh

# Step 2: Push to GitHub Container Registry
echo ""
echo "📤 Step 2: Pushing to GitHub Container Registry..."
chmod +x push_to_ghcr.sh
./push_to_ghcr.sh

# Step 3: Deploy to RunPod
echo ""
echo "🎯 Step 3: Deploy to RunPod"
echo "============================"
echo "The Docker images are now available at:"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-vc-latest"
echo "  - ghcr.io/chrijaque/runpod_chatterbox:higgs-tts-latest"
echo ""
echo "Next steps:"
echo "1. Go to RunPod dashboard"
echo "2. Create new endpoints using the configuration files:"
echo "   - runpod.higgs.vc.toml (for Voice Cloning)"
echo "   - runpod.higgs.tts.toml (for TTS Generation)"
echo "3. RunPod will use DOCKER_USERNAME and DOCKER_PASSWORD secrets for authentication"
echo ""
echo "✅ Workflow completed successfully!" 
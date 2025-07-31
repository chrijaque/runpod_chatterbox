#!/bin/bash

# Test Docker build locally
echo "🔧 Testing Docker build locally..."

# Set HF_TOKEN for testing
export HF_TOKEN="your_token_here"

echo "🔑 HF_TOKEN set: ${HF_TOKEN:0:10}..."

# Test the Docker build
docker build \
    --build-arg HF_TOKEN="$HF_TOKEN" \
    -f dockerfiles/higgs/Dockerfile.vc \
    -t test-higgs-vc \
    .

echo "✅ Docker build test completed" 
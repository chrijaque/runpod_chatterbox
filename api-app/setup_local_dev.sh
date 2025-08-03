#!/bin/bash

# Setup script for local development
# This script helps set up environment variables for local development

echo "🔧 Setting up local development environment..."

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✅ .env file already exists"
    echo "📝 Current .env contents:"
cat .env | grep -E "(FIREBASE_SERVICE_ACCOUNT|FIREBASE_STORAGE_BUCKET)" || echo "   No Firebase credentials found"
else
    echo "📝 Creating .env file..."
    touch .env
fi

echo ""
echo "📋 To enable Firebase functionality locally, add these to your .env file:"
echo ""
echo "# Firebase Configuration"
echo "FIREBASE_SERVICE_ACCOUNT='{\"type\":\"service_account\",\"project_id\":\"your-project-id\",...}'"
echo "FIREBASE_STORAGE_BUCKET=your-project-id.firebasestorage.app"
echo ""
echo "# API Configuration"
echo "API_HOST=0.0.0.0"
echo "API_PORT=8000"
echo "DEBUG=False"
echo "LOCAL_STORAGE_ENABLED=True"
echo "FIREBASE_STORAGE_ENABLED=True"
echo ""
echo "# RunPod Configuration"
echo "RUNPOD_API_KEY=your-runpod-api-key"
echo "VC_CB_ENDPOINT_ID=your-chatterbox-voice-clone-endpoint"
echo "TTS_CB_ENDPOINT_ID=your-chatterbox-tts-endpoint"
echo ""

echo "✅ Setup complete!"
echo "🚀 You can now start the API server with:"
echo "   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "💡 Note: The API server works without Firebase credentials, but you won't see"
echo "   existing voices in the library. Add Firebase credentials to .env to see them." 
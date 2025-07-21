#!/bin/bash

echo "🎵 Chatterbox Voice Cloning - Local Setup"
echo "========================================"
echo ""
echo "This application requires 2 services:"
echo "1. Local API Server (Flask) - Port 5001"
echo "2. Frontend (Next.js) - Port 3000"
echo ""

# Check if requirements are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Please install dependencies first:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Please install:"
    echo "   cd frontend && npm install"
    exit 1
fi

echo "✅ Dependencies check passed"
echo ""

# Function to start local API
start_api() {
    echo "🚀 Starting Local API Server (Flask)..."
    python app.py &
    API_PID=$!
    echo "   API Server PID: $API_PID"
    echo "   Available at: http://localhost:5001"
    echo ""
}

# Function to start frontend
start_frontend() {
    echo "🌐 Starting Frontend (Next.js)..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    echo "   Available at: http://localhost:3000"
    cd ..
    echo ""
}

# Start both services
start_api
sleep 3  # Give API time to start
start_frontend

echo "✅ Both services are starting!"
echo ""
echo "📋 Next Steps:"
echo "1. Visit http://localhost:3000 in your browser"
echo "2. Configure RunPod credentials in frontend/.env.local"
echo "3. Create your first voice clone!"
echo ""
echo "🛑 To stop services: Ctrl+C or kill $API_PID $FRONTEND_PID"
echo ""

# Wait for services
wait 
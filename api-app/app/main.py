from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

from .config import settings
from .middleware.security import verify_hmac, verify_firebase_auth, verify_app_check
from .api import voices, tts, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Library API",
    description="Production-ready voice cloning and TTS API with Firebase integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (no enforcement yet; security will be added per-route later)
app.include_router(voices.router, prefix="/api/voices", tags=["voices"])
app.include_router(tts.router, prefix="/api/tts", tags=["tts"])
app.include_router(health.router, prefix="/api/health", tags=["health"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🎵 Voice Library API Server Starting...")
    logger.info(f"🔥 R2: {'Enabled' if settings.FIREBASE_STORAGE_ENABLED else 'Disabled'}")
    logger.info(f"🌐 API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"📚 API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("🛑 Voice Library API Server Shutting Down...")

if __name__ == "__main__":
    import uvicorn
    logger.info("🎵 Voice Library API Server Starting...")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT) 
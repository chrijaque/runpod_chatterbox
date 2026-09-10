from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from .engine import engine
from .models import english_only_model, normalize_model_type
from .paths import DATA_DIR, ensure_data_dirs
from .storage import (
    get_voice,
    list_tts,
    list_voices,
    profile_path,
    recorded_path,
    sample_path,
    save_tts_meta,
    save_voice_meta,
    tts_audio_path,
    voice_id_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local Chatterbox",
    description="On-machine voice cloning and TTS for the runpod_chatterbox frontend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VoiceCloneRequest(BaseModel):
    name: str
    audio_data: str
    audio_format: str = "wav"
    language: str = "en"
    is_kids_voice: bool = False
    model_type: str = "chatterbox"
    user_id: Optional[str] = None


class TTSGenerateRequest(BaseModel):
    voice_id: str
    text: str
    profile_base64: Optional[str] = None
    language: str = "en"
    story_type: str = "user"
    is_kids_voice: bool = False
    model_type: str = "chatterbox"
    user_id: Optional[str] = None
    story_id: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


def public_url(request: Request, path: str) -> str:
    return str(request.base_url).rstrip("/") + path


def decode_audio(audio_data: str, dest) -> None:
    payload = audio_data.split(",", 1)[-1] if audio_data.startswith("data:") else audio_data
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio_data") from exc
    if len(raw) < 1000:
        raise HTTPException(status_code=400, detail="Audio file is too small")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": detail, "detail": detail},
    )


@app.on_event("startup")
def startup() -> None:
    ensure_data_dirs()
    logger.info("Local data directory: %s", DATA_DIR)
    engine.load()


@app.get("/health")
@app.get("/api/health/health")
def health():
    return {
        "status": "healthy",
        "service": "local-chatterbox",
        "device": engine.device,
        "data_dir": str(DATA_DIR),
        "loaded_families": sorted(engine.models.keys()),
    }


@app.post("/api/voices/clone")
def clone_voice(payload: VoiceCloneRequest, http_request: Request):
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Voice name is required")

    try:
        model_type = normalize_model_type(payload.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    language = "en" if english_only_model(model_type) else (payload.language or "en").strip().lower()
    voice_id = voice_id_for(payload.name)
    recorded = recorded_path(voice_id, payload.audio_format)
    profile = profile_path(voice_id)
    sample = sample_path(voice_id)
    decode_audio(payload.audio_data, recorded)

    from chatterbox.factory import sample_text_for_language

    sample_text = sample_text_for_language(language, payload.name.strip())
    started = time.time()
    try:
        engine.save_profile(recorded, profile, model_type)
        wav, sample_rate = engine.synthesize(sample_text, profile, model_type, language)
        engine.save_wav(wav, sample_rate, sample)
    except Exception as exc:
        logger.exception("Voice clone failed")
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {exc}") from exc

    meta = save_voice_meta(
        voice_id,
        name=payload.name.strip(),
        language=language,
        is_kids_voice=payload.is_kids_voice,
        sample_rate=sample_rate,
        template_message=sample_text,
        model_type=model_type,
    )
    audio_b64 = base64.b64encode(sample.read_bytes()).decode("ascii")
    sample_url = public_url(http_request, f"/api/voices/{voice_id}/sample")
    logger.info("Cloned %s in %.1fs", voice_id, time.time() - started)
    return {
        "status": "success",
        "voice_id": voice_id,
        "audio_base64": audio_b64,
        "sample_audio_path": sample_url,
        "profile_path": str(profile),
        "generation_time": time.time() - started,
        "metadata": {
            "voice_id": voice_id,
            "voice_name": payload.name.strip(),
            "embedding_path": str(profile),
            "embedding_exists": True,
            "has_embedding_support": True,
            "generation_method": "embedding-based",
            "sample_file": sample_url,
            "template_message": sample_text,
            "sample_rate": sample_rate,
            "audio_shape": list(wav.shape),
            "model_type": model_type,
            "language": language,
        },
        "created_date": meta["created_date"],
    }


@app.get("/api/voices")
@app.get("/api/voices/by-language/{language}")
def voices_by_language(
    request: Request,
    language: str = "en",
    is_kids_voice: bool = Query(False),
    model_type: Optional[str] = Query(None),
):
    resolved_model = None
    if model_type:
        try:
            resolved_model = normalize_model_type(model_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    voices = []
    for meta in list_voices(language, is_kids_voice, resolved_model):
        voice_id = meta["voice_id"]
        voices.append(
            {
                "voice_id": voice_id,
                "name": meta.get("name", voice_id),
                "sample_file": public_url(request, f"/api/voices/{voice_id}/sample"),
                "embedding_file": public_url(request, f"/api/voices/{voice_id}/profile"),
                "created_date": meta.get("created_date", 0),
                "language": meta.get("language", language),
                "is_kids_voice": meta.get("is_kids_voice", False),
                "model_type": meta.get("model_type") or "chatterbox",
            }
        )
    return {
        "status": "success",
        "voices": voices,
        "language": language,
        "is_kids_voice": is_kids_voice,
        "total": len(voices),
        "total_voices": len(voices),
    }


@app.get("/api/voices/{voice_id}/sample")
def get_sample(voice_id: str):
    path = sample_path(voice_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}.wav")


@app.get("/api/voices/{voice_id}/profile")
def get_profile(voice_id: str, language: str = "en", is_kids_voice: bool = False):
    path = profile_path(voice_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return {
        "status": "success",
        "voice_id": voice_id,
        "profile_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


@app.post("/api/tts/generate")
def generate_tts(payload: TTSGenerateRequest, http_request: Request):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        model_type = normalize_model_type(payload.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    language = "en" if english_only_model(model_type) else (payload.language or "en").strip().lower()
    voice = get_voice(payload.voice_id)
    stored_model = (voice or {}).get("model_type") or "chatterbox"
    if voice and stored_model != model_type:
        raise HTTPException(
            status_code=400,
            detail=f"Voice {payload.voice_id} was cloned as {stored_model} and cannot run on {model_type}. Re-clone it for this model.",
        )
    profile = profile_path(payload.voice_id)
    if not profile.exists() and payload.profile_base64:
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_bytes(base64.b64decode(payload.profile_base64.split(",", 1)[-1]))
    if not profile.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found")

    generation_id = f"tts_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    dest = tts_audio_path(generation_id)
    started = time.time()
    try:
        wav, sample_rate = engine.synthesize(payload.text.strip(), profile, model_type, language)
        engine.save_wav(wav, sample_rate, dest)
    except Exception as exc:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    audio_url = public_url(http_request, f"/api/tts/generations/{generation_id}/audio")
    meta = save_tts_meta(
        generation_id,
        voice_id=payload.voice_id,
        voice_name=(voice or {}).get("name", payload.voice_id),
        language=language,
        story_type=payload.story_type,
        text=payload.text.strip(),
        file_size=dest.stat().st_size,
    )
    elapsed = time.time() - started
    logger.info("TTS %s in %.1fs", generation_id, elapsed)
    return {
        "status": "success",
        "generation_id": generation_id,
        "voice_id": payload.voice_id,
        "audio_path": audio_url,
        "metadata": {
            "voice_id": payload.voice_id,
            "voice_name": meta["voice_name"],
            "text_input": payload.text.strip(),
            "generation_time": elapsed,
            "sample_rate": sample_rate,
            "audio_shape": list(wav.shape),
            "tts_file": audio_url,
            "timestamp": meta["timestamp"],
            "response_type": "local",
            "format": "wav",
            "audio_path": audio_url,
        },
    }


@app.get("/api/tts/stories/{language}")
def tts_stories(language: str, request: Request, story_type: str = Query("user")):
    stories = []
    for meta in list_tts(language, story_type):
        generation_id = meta["generation_id"]
        stories.append(
            {
                **meta,
                "audio_file": public_url(request, f"/api/tts/generations/{generation_id}/audio"),
            }
        )
    return {"status": "success", "stories": stories, "total": len(stories)}


@app.get("/api/tts/generations")
def tts_generations(request: Request, language: str = "en", story_type: str = "user"):
    payload = tts_stories(language, request, story_type)
    return {
        "status": "success",
        "total_generations": payload["total"],
        "generations": payload["stories"],
    }


@app.get("/api/tts/generations/{generation_id}/audio")
def tts_audio(generation_id: str):
    path = tts_audio_path(generation_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="TTS audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{generation_id}.wav")

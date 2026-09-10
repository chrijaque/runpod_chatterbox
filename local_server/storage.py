from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from .paths import TTS_DIR, VOICES_DIR, ensure_data_dirs


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", name.lower().replace(" ", "_"))
    return cleaned or "voice"


def voice_id_for(name: str) -> str:
    base = f"voice_{_slug(name)}"
    if not (VOICES_DIR / base).exists():
        return base
    return f"{base}_{int(time.time())}"


def voice_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id


def profile_path(voice_id: str) -> Path:
    return voice_dir(voice_id) / "profile.npy"


def sample_path(voice_id: str) -> Path:
    return voice_dir(voice_id) / "sample.wav"


def recorded_path(voice_id: str, audio_format: str) -> Path:
    ext = (audio_format or "wav").lstrip(".").lower()
    if ext not in {"wav", "mp3", "m4a", "webm", "ogg", "flac"}:
        ext = "wav"
    return voice_dir(voice_id) / f"recorded.{ext}"


def meta_path(voice_id: str) -> Path:
    return voice_dir(voice_id) / "meta.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_voice_meta(
    voice_id: str,
    *,
    name: str,
    language: str,
    is_kids_voice: bool,
    sample_rate: int,
    template_message: str,
    model_type: str = "chatterbox",
) -> dict[str, Any]:
    ensure_data_dirs()
    voice_dir(voice_id).mkdir(parents=True, exist_ok=True)
    meta = {
        "voice_id": voice_id,
        "name": name,
        "language": language,
        "is_kids_voice": is_kids_voice,
        "model_type": model_type,
        "created_date": time.time(),
        "sample_rate": sample_rate,
        "template_message": template_message,
    }
    write_json(meta_path(voice_id), meta)
    return meta


def list_voices(language: str, is_kids_voice: bool, model_type: Optional[str] = None) -> list[dict[str, Any]]:
    ensure_data_dirs()
    voices: list[dict[str, Any]] = []
    for directory in sorted(VOICES_DIR.iterdir()) if VOICES_DIR.exists() else []:
        if not directory.is_dir():
            continue
        meta_file = directory / "meta.json"
        if not meta_file.exists() or not (directory / "profile.npy").exists():
            continue
        meta = read_json(meta_file)
        if meta.get("language", "en") != language:
            continue
        if bool(meta.get("is_kids_voice", False)) != bool(is_kids_voice):
            continue
        stored_model = meta.get("model_type") or "chatterbox"
        if model_type and stored_model != model_type:
            continue
        voices.append(meta)
    voices.sort(key=lambda item: item.get("created_date", 0), reverse=True)
    return voices


def get_voice(voice_id: str) -> Optional[dict[str, Any]]:
    path = meta_path(voice_id)
    if not path.exists():
        return None
    return read_json(path)


def save_tts_meta(
    generation_id: str,
    *,
    voice_id: str,
    voice_name: str,
    language: str,
    story_type: str,
    text: str,
    file_size: int,
) -> dict[str, Any]:
    ensure_data_dirs()
    meta = {
        "generation_id": generation_id,
        "file_id": generation_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "language": language,
        "story_type": story_type,
        "text": text,
        "created_date": time.time(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "file_size": file_size,
    }
    write_json(TTS_DIR / f"{generation_id}.json", meta)
    return meta


def tts_audio_path(generation_id: str) -> Path:
    return TTS_DIR / f"{generation_id}.wav"


def list_tts(language: str, story_type: str) -> list[dict[str, Any]]:
    ensure_data_dirs()
    items: list[dict[str, Any]] = []
    for path in sorted(TTS_DIR.glob("*.json")):
        meta = read_json(path)
        if meta.get("language", "en") != language:
            continue
        if meta.get("story_type", "user") != story_type:
            continue
        if not tts_audio_path(meta["generation_id"]).exists():
            continue
        items.append(meta)
    items.sort(key=lambda item: item.get("created_date", 0), reverse=True)
    return items

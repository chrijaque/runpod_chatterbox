from __future__ import annotations

from typing import Literal, Optional

from fastapi import HTTPException

from .config import settings

ModelType = Literal["chatterbox", "chatterbox-turbo", "chatterbox-mtl"]

_ALIASES = {
    "chatterbox": "chatterbox",
    "original": "chatterbox",
    "legacy": "chatterbox",
    "chatterbox-turbo": "chatterbox-turbo",
    "turbo": "chatterbox-turbo",
    "chatterbox-mtl": "chatterbox-mtl",
    "mtl": "chatterbox-mtl",
}


def normalize_model_type(raw: Optional[str]) -> ModelType:
    value = (raw or "chatterbox").strip().lower()
    if value not in _ALIASES:
        raise HTTPException(
            status_code=400,
            detail="Unknown model_type. Use chatterbox, chatterbox-turbo, or chatterbox-mtl.",
        )
    return _ALIASES[value]  # type: ignore[return-value]


def tts_endpoint_id_for_model(model_type: str) -> str:
    resolved = normalize_model_type(model_type)
    if resolved == "chatterbox-turbo":
        endpoint_id = settings.TTS_TURBO_ENDPOINT_ID
        env_name = "TTS_TURBO_ENDPOINT_ID"
    elif resolved == "chatterbox-mtl":
        endpoint_id = settings.TTS_MTL_ENDPOINT_ID
        env_name = "TTS_MTL_ENDPOINT_ID"
    else:
        endpoint_id = settings.TTS_CB_ENDPOINT_ID
        env_name = "TTS_CB_ENDPOINT_ID"
    if not endpoint_id:
        raise HTTPException(
            status_code=500,
            detail=f"{env_name} is not configured for model_type={resolved}",
        )
    return endpoint_id


def vc_endpoint_id_for_model(model_type: str) -> str:
    resolved = normalize_model_type(model_type)
    if resolved == "chatterbox-turbo":
        endpoint_id = settings.VC_TURBO_ENDPOINT_ID
        env_name = "VC_TURBO_ENDPOINT_ID"
    elif resolved == "chatterbox-mtl":
        endpoint_id = settings.VC_MTL_ENDPOINT_ID
        env_name = "VC_MTL_ENDPOINT_ID"
    else:
        endpoint_id = settings.VC_CB_ENDPOINT_ID
        env_name = "VC_CB_ENDPOINT_ID"
    if not endpoint_id:
        raise HTTPException(
            status_code=500,
            detail=f"{env_name} is not configured for model_type={resolved}",
        )
    return endpoint_id

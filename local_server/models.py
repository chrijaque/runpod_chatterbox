from __future__ import annotations

MODEL_TYPE_ORIGINAL = "chatterbox"
MODEL_TYPE_TURBO = "chatterbox-turbo"
MODEL_TYPE_MTL = "chatterbox-mtl"

FAMILY_ORIGINAL = "original"
FAMILY_TURBO = "turbo"
FAMILY_MTL = "mtl"

_ALIASES = {
    "chatterbox": MODEL_TYPE_ORIGINAL,
    "original": MODEL_TYPE_ORIGINAL,
    "legacy": MODEL_TYPE_ORIGINAL,
    "chatterbox-turbo": MODEL_TYPE_TURBO,
    "turbo": MODEL_TYPE_TURBO,
    "chatterbox-mtl": MODEL_TYPE_MTL,
    "mtl": MODEL_TYPE_MTL,
}

_MODEL_TYPE_TO_FAMILY = {
    MODEL_TYPE_ORIGINAL: FAMILY_ORIGINAL,
    MODEL_TYPE_TURBO: FAMILY_TURBO,
    MODEL_TYPE_MTL: FAMILY_MTL,
}


def normalize_model_type(raw: str | None) -> str:
    value = (raw or MODEL_TYPE_ORIGINAL).strip().lower()
    if value not in _ALIASES:
        raise ValueError(
            f"Unknown model_type '{raw}'. Use chatterbox, chatterbox-turbo, or chatterbox-mtl."
        )
    return _ALIASES[value]


def family_for_model_type(raw: str | None) -> str:
    return _MODEL_TYPE_TO_FAMILY[normalize_model_type(raw)]


def english_only_model(raw: str | None) -> bool:
    return normalize_model_type(raw) in (MODEL_TYPE_ORIGINAL, MODEL_TYPE_TURBO)

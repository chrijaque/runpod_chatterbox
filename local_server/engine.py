from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
import torchaudio

from .models import family_for_model_type, normalize_model_type
from .paths import ensure_chatterbox_on_path

logger = logging.getLogger(__name__)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalChatterbox:
    def __init__(self) -> None:
        self.device = detect_device()
        self.models: dict[str, object] = {}
        self.lock = threading.RLock()

    @property
    def model(self):
        return next(iter(self.models.values()), None)

    def load(self) -> None:
        ensure_chatterbox_on_path()
        import os

        os.environ.setdefault("CHATTERBOX_PROD_MODE", "true")
        os.environ.setdefault("CHATTERBOX_EXPERIMENT_MODE", "false")
        logger.info("Local Chatterbox ready on %s (models load on first use)", self.device)

    def get_model(self, model_type: str | None):
        family = family_for_model_type(model_type)
        with self.lock:
            cached = self.models.get(family)
            if cached is not None:
                return cached

            ensure_chatterbox_on_path()
            import os
            import perth

            os.environ.setdefault("CHATTERBOX_PROD_MODE", "true")
            os.environ.setdefault("CHATTERBOX_EXPERIMENT_MODE", "false")
            if getattr(perth, "PerthImplicitWatermarker", None) is None:
                logger.warning("PerthImplicitWatermarker unavailable; using DummyWatermarker")
                perth.PerthImplicitWatermarker = perth.DummyWatermarker

            from chatterbox.factory import load_tts_model

            logger.info("Loading Chatterbox family=%s on %s", family, self.device)
            model = load_tts_model(self.device, family=family)
            self.models[family] = model
            logger.info("Chatterbox family=%s ready on %s", family, self.device)
            return model

    def save_profile(self, audio_path: Path, profile_path: Path, model_type: str | None) -> None:
        model = self.get_model(model_type)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            model.save_voice_profile(str(audio_path), str(profile_path))

    def synthesize(
        self,
        text: str,
        profile_path: Path,
        model_type: str | None,
        language: str = "en",
    ) -> tuple["torch.Tensor", int]:
        resolved_type = normalize_model_type(model_type)
        model = self.get_model(resolved_type)
        with self.lock:
            if resolved_type == "chatterbox":
                model.conds = None
                model._cached_conditionals = None
                model._cached_voice_profile_path = None
                wav = model.generate(text, voice_profile_path=str(profile_path))
            else:
                model.load_voice_profile(str(profile_path))
                generate_kwargs = {}
                if resolved_type == "chatterbox-mtl":
                    generate_kwargs["language_id"] = language
                wav = model.generate(text, **generate_kwargs)
            return wav, int(model.sr)

    def save_wav(self, wav: "torch.Tensor", sample_rate: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(dest), wav.cpu(), sample_rate)


engine = LocalChatterbox()

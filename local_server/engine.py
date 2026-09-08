from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
import torchaudio

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
        self.model = None
        self.lock = threading.Lock()

    def load(self) -> None:
        ensure_chatterbox_on_path()
        import os

        os.environ.setdefault("CHATTERBOX_PROD_MODE", "true")
        os.environ.setdefault("CHATTERBOX_EXPERIMENT_MODE", "false")

        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            logger.warning("PerthImplicitWatermarker unavailable; using DummyWatermarker")
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

        from chatterbox.tts import ChatterboxTTS

        logger.info("Loading ChatterboxTTS on %s (first run downloads Hugging Face weights)", self.device)
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        logger.info("ChatterboxTTS ready on %s", self.device)

    def save_profile(self, audio_path: Path, profile_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.model.save_voice_profile(str(audio_path), str(profile_path))

    def synthesize(self, text: str, profile_path: Path) -> tuple["torch.Tensor", int]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        with self.lock:
            self.model.conds = None
            self.model._cached_conditionals = None
            self.model._cached_voice_profile_path = None
            wav = self.model.generate(text, voice_profile_path=str(profile_path))
            return wav, int(self.model.sr)

    def save_wav(self, wav: "torch.Tensor", sample_rate: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(dest), wav.cpu(), sample_rate)


engine = LocalChatterbox()

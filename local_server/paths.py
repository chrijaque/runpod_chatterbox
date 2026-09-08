from pathlib import Path
import os
import sys

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
PROJECTS_ROOT = REPO_ROOT.parent
CHATTERBOX_EMBED_SRC = PROJECTS_ROOT / "chatterbox_embed" / "src"
DATA_DIR = Path(os.getenv("LOCAL_CHATTERBOX_DATA", str(REPO_ROOT / "local_data"))).resolve()
VOICES_DIR = DATA_DIR / "voices"
TTS_DIR = DATA_DIR / "tts"


def ensure_data_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    TTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_chatterbox_on_path() -> None:
    src = str(CHATTERBOX_EMBED_SRC)
    if CHATTERBOX_EMBED_SRC.exists() and src not in sys.path:
        sys.path.insert(0, src)

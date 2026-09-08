#!/usr/bin/env python3
"""Start the local Chatterbox API used by the voice-cloning frontend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_server.paths import CHATTERBOX_EMBED_SRC, DATA_DIR, ensure_chatterbox_on_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Chatterbox API (no RunPod)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ensure_chatterbox_on_path()
    try:
        import chatterbox  # noqa: F401
    except ImportError:
        print("Could not import chatterbox.")
        print("From /Users/christianhammer/Projects/chatterbox_embed run:")
        print("  python3.11 -m venv .venv && source .venv/bin/activate")
        print("  pip install -e .")
        print("  pip install fastapi 'uvicorn[standard]' python-multipart")
        print(f"Expected package path: {CHATTERBOX_EMBED_SRC}")
        sys.exit(1)

    print("=" * 64)
    print("Local Chatterbox API")
    print(f"  API:      http://localhost:{args.port}")
    print(f"  Docs:     http://localhost:{args.port}/docs")
    print(f"  Data:     {DATA_DIR}")
    print("Frontend (other terminal):")
    print("  cd frontend && npm install && npm run dev -- -p 3001")
    print("  open http://localhost:3001")
    print("=" * 64)

    import uvicorn

    uvicorn.run(
        "local_server.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

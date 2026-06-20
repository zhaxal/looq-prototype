#!/usr/bin/env python3
"""Download the two zoo models this app needs (YuNet + head-pose) into models/.

The Pi setup script restores these from git, so you normally don't need this. Use it
to refresh the archives or fetch a face resolution that isn't committed:

    python scripts/download_models.py
"""
import shutil
import sys
from pathlib import Path

# Allow running as `python scripts/download_models.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention.config import load_dotenv, MODELS_DIR

load_dotenv()

import depthai as dai  # noqa: E402 — must come after load_dotenv

DOWNLOADS = [
    ("luxonis/yunet:320x240",              "yunet-320x240.rvc2.tar.xz"),
    ("luxonis/yunet:640x480",              "yunet-640x480.rvc2.tar.xz"),
    ("luxonis/head-pose-estimation:60x60", "head-pose-60x60.rvc2.tar.xz"),
]


def download(slug: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip]  {dest.name}  ({dest.stat().st_size // 1024} KB)")
        return
    print(f"[fetch] {slug} …")
    desc = dai.NNModelDescription(model=slug, platform="RVC2")
    tmp  = dai.getModelFromZoo(desc, useCached=True, progressFormat="pretty")
    shutil.copy(tmp, dest)
    print(f"        → {dest.name}  ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    for slug, fname in DOWNLOADS:
        download(slug, MODELS_DIR / fname)
    print("\nDone.")

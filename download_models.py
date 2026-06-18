#!/usr/bin/env python3
"""Download zoo models on the laptop (where zoo access works), then rsync to Pi.

Run on the laptop:
    python download_models.py

Then copy to Pi (adjust hostname/path):
    rsync -av models/ pi@raspberrypi.local:~/looq-prototype/models/

face_attention.py uses models/ automatically when present; no network needed on Pi.
"""
import os
import shutil
from pathlib import Path

_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import depthai as dai

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

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
    for slug, fname in DOWNLOADS:
        download(slug, MODELS_DIR / fname)
    print("\nDone. Rsync to Pi:")
    print("  rsync -av models/ pi@raspberrypi.local:~/looq-prototype/models/")

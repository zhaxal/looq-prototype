#!/usr/bin/env python3
"""Download zoo models once; saves to models/ so the pipeline runs offline.

Usage:
    DEPTHAI_HUB_API_KEY=your_key python download_models.py
    # or put the key in .env and just run:
    python download_models.py
"""
import os
import shutil
from pathlib import Path

# Load .env (same logic as face_attention.py so one file covers both)
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

# (slug, output filename)
# Slug format: "namespace/name:variant" — the variant selects the resolution.
# Platform "RVC2" targets OAK-D-Lite / MyriadX VPU.
DOWNLOADS = [
    ("luxonis/yunet:320x240",              "yunet-320x240.rvc2.tar.xz"),
    ("luxonis/yunet:640x480",              "yunet-640x480.rvc2.tar.xz"),
    ("luxonis/head-pose-estimation:60x60", "head-pose-60x60.rvc2.tar.xz"),
]


def download(slug: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip]  {dest.name}  (already present, {dest.stat().st_size // 1024} KB)")
        return
    print(f"[fetch] {slug} → {dest.name}")
    # VERIFY: NNModelDescription constructor accepts the full slug as `model`.
    # `platform` selects the VPU target; RVC2 = OAK-D-Lite / Myriad X.
    desc = dai.NNModelDescription(model=slug, platform="RVC2")
    tmp = dai.getModelFromZoo(desc, useCached=True, progressFormat="pretty")
    shutil.copy(tmp, dest)
    print(f"        saved  {dest.stat().st_size // 1024} KB")


if __name__ == "__main__":
    if not os.environ.get("DEPTHAI_HUB_API_KEY"):
        print("WARNING: DEPTHAI_HUB_API_KEY not set — public models may still work "
              "but private ones will fail. Set it in .env or the environment.")
    for slug, fname in DOWNLOADS:
        download(slug, MODELS_DIR / fname)
    print("\nAll done. Run face_attention.py — it will use local archives.")

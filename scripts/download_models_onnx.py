#!/usr/bin/env python3
"""Fetch host-runnable ONNX models for the GPU server into models/onnx/.

    python scripts/download_models_onnx.py

Notes on sources (VERIFY URLs — they drift):
  * YuNet           — OpenCV Zoo, direct ONNX download.
  * Emotion         — HSEmotion enet_b2_8_best, direct ONNX download.
  * Head pose &     — OpenVINO Open Model Zoo models with no canonical ONNX.
    Age/gender        Provide ONNX via env URLs, or convert their IR locally
                      (openvino → onnx) and drop the .onnx files in models/onnx/.

Env overrides (point at your own mirror / converted files):
  YUNET_ONNX_URL, EMOTION_ONNX_URL, HEAD_POSE_ONNX_URL, AGE_GENDER_ONNX_URL
"""
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from attention.config import load_dotenv, ONNX_MODELS_DIR
from server.models import (
    YUNET_FILE, HEAD_POSE_FILE, AGE_GENDER_FILE, EMOTION_FILE,
)

load_dotenv()

YUNET_URL = os.environ.get(
    "YUNET_ONNX_URL",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
EMOTION_URL = os.environ.get(
    "EMOTION_ONNX_URL",
    "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/"
    "affectnet_emotions/onnx/enet_b2_8_best.onnx",  # VERIFY path
)
# No canonical ONNX exists for these OpenVINO models — supply your own URL.
HEAD_POSE_URL  = os.environ.get("HEAD_POSE_ONNX_URL", "")
AGE_GENDER_URL = os.environ.get("AGE_GENDER_ONNX_URL", "")

DOWNLOADS = [
    (YUNET_URL,      YUNET_FILE),
    (EMOTION_URL,    EMOTION_FILE),
    (HEAD_POSE_URL,  HEAD_POSE_FILE),
    (AGE_GENDER_URL, AGE_GENDER_FILE),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip]  {dest.name}  ({dest.stat().st_size // 1024} KB)")
        return
    if not url:
        print(f"[MANUAL] {dest.name}: no URL — convert its OpenVINO IR to ONNX "
              f"and place it at {dest}, or set its *_ONNX_URL env var.")
        return
    print(f"[fetch] {dest.name}  ←  {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"        → {dest.name}  ({dest.stat().st_size // 1024} KB)")
    except Exception as exc:
        print(f"[ERROR] {dest.name}: {exc}")


if __name__ == "__main__":
    ONNX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for url, fname in DOWNLOADS:
        fetch(url, ONNX_MODELS_DIR / fname)
    print(f"\nModels dir: {ONNX_MODELS_DIR}")
    print("Self-test the models with:  python -m server.selftest")

"""Smoke-test the ONNX models on a synthetic frame.

    python -m server.selftest

Confirms each net loads and produces sane output shapes before wiring the full
pipeline (the top integration risk is ONNX node names / input formats).
"""
from __future__ import annotations

import numpy as np

from attention import config
from .models import VisionModels, crop_face


def main() -> None:
    print(f"[selftest] ONNX_MODELS_DIR = {config.ONNX_MODELS_DIR}")
    models = VisionModels()

    frame = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    dets = models.detector.detect(frame)
    print(f"[selftest] detector: {len(dets)} faces on noise (expect ~0)")

    # Exercise the per-face nets on a centered dummy crop.
    crop = crop_face(frame, (0.3, 0.3, 0.7, 0.7))
    print(f"[selftest] head_pose:  {models.head_pose(crop)}")
    if models.age_gender is not None:
        print(f"[selftest] age_gender: {models.age_gender(crop)}")
    if models.emotion is not None:
        print(f"[selftest] emotion:    {models.emotion(crop)}")
    print("[selftest] OK")


if __name__ == "__main__":
    main()

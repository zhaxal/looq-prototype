"""Host (GPU) inference for the four nets, via ONNX Runtime + OpenCV YuNet.

Pre/post mirror the on-device parsers in attention/processing.py
(extract_pose / extract_age_gender / extract_emotion) and the input sizes /
class labels in attention/config.py. Output tensor names vary across model
exports, so heads are resolved by name hint with positional fallback — flagged
with # VERIFY where it matters most.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from attention import config

# Filenames expected under config.ONNX_MODELS_DIR (see scripts/download_models_onnx.py).
YUNET_FILE      = "face_detection_yunet_2023mar.onnx"
HEAD_POSE_FILE  = "head-pose-estimation-adas-0001.onnx"
AGE_GENDER_FILE = "age-gender-recognition-retail-0013.onnx"
EMOTION_FILE    = "enet_b2_8_best.onnx"

# ImageNet stats for the HSEmotion (enet_b2) preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _providers() -> list[str]:
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("[models] WARNING: CUDAExecutionProvider not available — using CPU")
    return ["CPUExecutionProvider"]


def _session(path: Path) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(
            f"missing model {path} — run scripts/download_models_onnx.py"
        )
    return ort.InferenceSession(str(path), providers=_providers())


def _blob_bgr(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """OpenVINO-style blob: BGR, raw 0-255, NCHW float32."""
    img = cv2.resize(crop, size).astype(np.float32)
    return np.transpose(img, (2, 0, 1))[None, ...]


# --- Face detection (YuNet) --------------------------------------------------

class YuNetDetector:
    """Returns [(bbox_norm, score), ...] with bbox_norm = (xmin,ymin,xmax,ymax)."""

    def __init__(self, models_dir: Path) -> None:
        path = models_dir / YUNET_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"missing model {path} — run scripts/download_models_onnx.py"
            )
        # Input size is set per-frame in detect(); start with a placeholder.
        self._det = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320),
            score_threshold=config.FACE_CONFIDENCE,
        )
        self._wh: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[tuple[tuple, float]]:
        h, w = frame.shape[:2]
        if self._wh != (w, h):
            self._det.setInputSize((w, h))
            self._wh = (w, h)
        _, faces = self._det.detect(frame)
        out: list[tuple[tuple, float]] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh = f[0], f[1], f[2], f[3]
            score = float(f[14])
            bbox = (max(0.0, x / w), max(0.0, y / h),
                    min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
            if (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0:
                out.append((bbox, score))
        return out


# --- Head pose ---------------------------------------------------------------

class HeadPose:
    """(yaw, pitch, roll) in degrees from head-pose-estimation-adas-0001."""

    def __init__(self, models_dir: Path) -> None:
        self._sess = _session(models_dir / HEAD_POSE_FILE)
        self._in   = self._sess.get_inputs()[0].name
        names = [o.name for o in self._sess.get_outputs()]
        self._yaw   = self._pick(names, "y")    # VERIFY: angle_y_fc / fc_y
        self._pitch = self._pick(names, "p")    # VERIFY: angle_p_fc
        self._roll  = self._pick(names, "r")    # VERIFY: angle_r_fc

    @staticmethod
    def _pick(names: list[str], axis: str) -> str:
        for n in names:
            low = n.lower()
            if f"_{axis}_" in low or low.endswith(f"_{axis}") or f"angle_{axis}" in low:
                return n
        return names[{"y": 0, "p": 1, "r": 2}[axis]]   # positional fallback

    def __call__(self, crop: np.ndarray) -> tuple[float, float, float]:
        blob = _blob_bgr(crop, config.POSE_INPUT)
        out = {o.name: v for o, v in
               zip(self._sess.get_outputs(),
                   self._sess.run(None, {self._in: blob}))}
        return (float(np.ravel(out[self._yaw])[0]),
                float(np.ravel(out[self._pitch])[0]),
                float(np.ravel(out[self._roll])[0]))


# --- Age / gender ------------------------------------------------------------

class AgeGender:
    """(gender_str, age_int) from age-gender-recognition-retail-0013."""

    def __init__(self, models_dir: Path) -> None:
        self._sess = _session(models_dir / AGE_GENDER_FILE)
        self._in   = self._sess.get_inputs()[0].name
        names = [o.name for o in self._sess.get_outputs()]
        # age head is the 1-element regression; gender head is the 2-element softmax.
        self._age    = self._pick(names, "age")   # VERIFY: age_conv3 / fc_age
        self._gender = self._pick(names, "prob")  # VERIFY: prob / fc_gender

    @staticmethod
    def _pick(names: list[str], hint: str) -> str:
        for n in names:
            if hint in n.lower():
                return n
        return names[0] if hint == "age" else names[-1]

    def __call__(self, crop: np.ndarray) -> tuple[str, int]:
        blob = _blob_bgr(crop, config.AGE_GENDER_INPUT)
        out = {o.name: v for o, v in
               zip(self._sess.get_outputs(),
                   self._sess.run(None, {self._in: blob}))}
        age = round(float(np.ravel(out[self._age])[0]) * 100)
        gidx = int(np.argmax(np.ravel(out[self._gender])))
        gender = config.GENDER_CLASSES[gidx] if gidx < len(config.GENDER_CLASSES) else str(gidx)
        return gender, age


# --- Emotion (HSEmotion enet_b2_8) -------------------------------------------

class Emotion:
    """(label, confidence) from enet_b2_8_best (8 classes, RGB + ImageNet norm)."""

    def __init__(self, models_dir: Path) -> None:
        self._sess = _session(models_dir / EMOTION_FILE)
        self._in   = self._sess.get_inputs()[0].name
        self._out  = self._sess.get_outputs()[0].name

    def _blob(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, config.EMOTION_INPUT)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        logits = np.ravel(self._sess.run([self._out], {self._in: self._blob(crop)})[0])
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        label = (config.EMOTION_CLASSES[idx]
                 if idx < len(config.EMOTION_CLASSES) else str(idx))
        return label, float(probs[idx])


# --- Bundle ------------------------------------------------------------------

class VisionModels:
    """All nets loaded together, with optional age/gender + emotion."""

    def __init__(self, models_dir: Path | None = None,
                 age_gender: bool = True, emotion: bool = True) -> None:
        d = models_dir or config.ONNX_MODELS_DIR
        print(f"[models] loading from {d}  providers={_providers()}")
        self.detector   = YuNetDetector(d)
        self.head_pose  = HeadPose(d)
        self.age_gender = AgeGender(d) if age_gender else None
        self.emotion    = Emotion(d)   if emotion   else None


def crop_face(frame: np.ndarray, bbox_norm: tuple) -> np.ndarray | None:
    """Pixel crop from a normalized (xmin,ymin,xmax,ymax) bbox; None if empty."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox_norm[0] * w)); y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w)); y2 = min(h, int(bbox_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

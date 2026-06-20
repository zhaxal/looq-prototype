"""Runtime constants, model paths, and persisted user settings.

This build measures *ad attention*: how many people look at an advertisement and
for how long. The OAK-D-Lite sits **beside** the ad, so "looking at the ad" is a
head pose offset from the camera axis (see Settings.yaw_offset / Calibrate).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"


def load_dotenv() -> None:
    """Load DEPTHAI_HUB_API_KEY (and other vars) from .env into the environment."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# --- Model sources -----------------------------------------------------------

# Zoo slugs (downloaded by scripts/download_models.py, or restored from git).
FACE_MODEL_SLUG = "luxonis/yunet"                    # resolution appended at runtime
POSE_MODEL_SLUG = "luxonis/head-pose-estimation:60x60"

# --- NN input sizes ----------------------------------------------------------

POSE_INPUT = (60, 60)

# --- Pipeline constants ------------------------------------------------------

FACE_RESOLUTIONS = ("320x240", "640x360", "640x480", "960x720")

FACE_CONFIDENCE = 0.80
MIN_FACE_AREA   = 0.003   # normalized bbox area; filters distant blobs (~55x42 px @ 640x480)
DEBOUNCE_SECS   = 0.20    # commit looking/not-looking after this many seconds stable
IOU_THRESHOLD   = 0.2
POSE_UNSEEN     = (90.0, 90.0)   # sentinel used when no pose is cached yet


# --- Persisted user settings -------------------------------------------------

@dataclass
class Settings:
    """User-tunable settings, persisted to settings.json.

    The offsets aim the "looking" cone at the ad rather than the camera. They are
    signed degrees in the head-pose model's frame; the Calibrate step measures them
    empirically (and resolves the sign), so the operator never computes them by hand.
    """
    yaw_offset:   float = 0.0     # degrees; ad to one side of the camera flips the sign
    pitch_offset: float = 0.0     # degrees; usually ~0 unless device held above/below head
    yaw_tol:      float = 20.0    # half-width of the looking cone (yaw)
    pitch_tol:    float = 15.0    # half-width of the looking cone (pitch)
    face_res:     str   = "320x240"   # YuNet input; 320x240 is the Pi-friendly default
    fps:          float = 12.0
    log:          bool  = False   # write an attention_*.csv session log

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        """Load settings.json, falling back to defaults for any missing/invalid keys."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {k: data[k] for k in cls().__dict__ if k in data}
        return cls(**known)

    def save(self, path: Path = SETTINGS_FILE) -> None:
        """Persist settings to settings.json (pretty-printed for hand-editing)."""
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

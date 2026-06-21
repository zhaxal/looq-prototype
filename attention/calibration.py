"""Calibration profile: the head-pose direction that means "looking at the billboard".

The camera sits *beside* the billboard, so a person looking at the ad does NOT
face the camera — their head is turned by a fixed angle. Calibration measures
that angle empirically from one stable subject, so operators never hand-edit
yaw/pitch constants.

The profile is a small JSON file (see configs/example_calibration.json). It maps
directly onto the engine's existing Settings:

    yaw_mean_deg      -> Settings.yaw_offset     (centre of the looking cone, yaw)
    pitch_mean_deg    -> Settings.pitch_offset   (centre of the looking cone, pitch)
    yaw_tolerance_deg -> Settings.yaw_tol        (half-width of the cone, yaw)
    pitch_tolerance_deg -> Settings.pitch_tol    (half-width of the cone, pitch)
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config

# Reliability gates — calibration FAILS (no file written) unless all are met.
MIN_SAMPLES        = 15      # valid yaw/pitch samples from the single subject
MIN_VALID_SECONDS  = 2.0     # span of valid single-subject tracking
# Default cone half-widths if std-derived tolerances are not used.
DEFAULT_YAW_TOL    = 20.0
DEFAULT_PITCH_TOL  = 15.0

METHOD = "single_subject_5s_head_pose_billboard_direction"


class CalibrationError(Exception):
    """Raised when calibration input is invalid (zero/multi subject, too few samples)."""


@dataclass
class Calibration:
    """A persisted calibration profile. Field order matches the JSON schema."""
    calibration_id:    str
    created_at:        str
    camera_id:         str
    camera_height_m:   float
    billboard_id:      str
    billboard_width_m: float
    billboard_height_m: float
    method:            str
    target:            str
    yaw_mean_deg:      float
    pitch_mean_deg:    float
    yaw_std_deg:       float
    pitch_std_deg:     float
    sample_count:      int
    valid_duration_sec: float
    yaw_tolerance_deg:  float
    pitch_tolerance_deg: float
    notes: str = ("MVP calibration. Calibrated head-pose direction, "
                  "not verified geometric gaze.")

    # --- Persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        """Load and structurally validate a calibration JSON file."""
        p = Path(path)
        if not p.exists():
            raise CalibrationError(f"Calibration file not found: {p}")
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise CalibrationError(f"Calibration file is not valid JSON: {p} ({e})") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        required = ("yaw_mean_deg", "pitch_mean_deg",
                    "yaw_tolerance_deg", "pitch_tolerance_deg")
        missing = [k for k in required if k not in data]
        if missing:
            raise CalibrationError(
                f"Calibration file is missing required keys: {', '.join(missing)}")
        known = {f for f in cls.__dataclass_fields__}            # tolerate extra keys
        filtered = {k: v for k, v in data.items() if k in known}
        # Fill any non-essential missing fields with safe defaults.
        defaults = {
            "calibration_id": data.get("billboard_id", "unknown"),
            "created_at": "", "camera_id": "unknown",
            "camera_height_m": 0.0, "billboard_id": "unknown",
            "billboard_width_m": 0.0, "billboard_height_m": 0.0,
            "method": METHOD, "target": "center",
            "yaw_std_deg": 0.0, "pitch_std_deg": 0.0,
            "sample_count": 0, "valid_duration_sec": 0.0,
        }
        for k, v in defaults.items():
            filtered.setdefault(k, v)
        return cls(**filtered)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p

    # --- Apply to the engine -------------------------------------------------

    def apply_to_settings(self, settings: config.Settings) -> None:
        """Aim the looking cone at the calibrated billboard direction."""
        settings.yaw_offset   = round(self.yaw_mean_deg, 1)
        settings.pitch_offset = round(self.pitch_mean_deg, 1)
        settings.yaw_tol      = round(self.yaw_tolerance_deg, 1)
        settings.pitch_tol    = round(self.pitch_tolerance_deg, 1)


def _tolerances(yaw_std: float, pitch_std: float, robust: bool) -> tuple[float, float]:
    """Pick cone half-widths. Default = fixed robust values; optionally widen by std."""
    if not robust:
        return DEFAULT_YAW_TOL, DEFAULT_PITCH_TOL
    yaw_tol   = min(30.0, max(15.0, 2.5 * yaw_std))
    pitch_tol = min(25.0, max(12.0, 2.5 * pitch_std))
    return round(yaw_tol, 1), round(pitch_tol, 1)


def build_from_samples(
    samples: list[tuple[float, float]],
    valid_duration_sec: float,
    *,
    target: str = "center",
    camera_id: str = "unknown",
    camera_height_m: float = 0.0,
    billboard_id: str = "unknown",
    billboard_width_m: float = 0.0,
    billboard_height_m: float = 0.0,
    std_tolerances: bool = False,
) -> Calibration:
    """Build a Calibration from collected (yaw, pitch) samples, enforcing the
    reliability gates. Raises CalibrationError if the input is not trustworthy.
    """
    if len(samples) < MIN_SAMPLES:
        raise CalibrationError(
            f"Only {len(samples)} valid yaw/pitch samples (need >= {MIN_SAMPLES}). "
            "Move closer, improve lighting, and keep one face clearly visible.")
    if valid_duration_sec < MIN_VALID_SECONDS:
        raise CalibrationError(
            f"Only {valid_duration_sec:.1f}s of valid tracking "
            f"(need >= {MIN_VALID_SECONDS:.0f}s). Hold still and look at the billboard.")

    yaws   = [s[0] for s in samples]
    pitches = [s[1] for s in samples]
    yaw_mean   = round(statistics.median(yaws), 1)
    pitch_mean = round(statistics.median(pitches), 1)
    yaw_std    = round(statistics.pstdev(yaws), 1)   if len(yaws) > 1 else 0.0
    pitch_std  = round(statistics.pstdev(pitches), 1) if len(pitches) > 1 else 0.0
    yaw_tol, pitch_tol = _tolerances(yaw_std, pitch_std, std_tolerances)

    return Calibration(
        calibration_id=billboard_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        camera_id=camera_id,
        camera_height_m=camera_height_m,
        billboard_id=billboard_id,
        billboard_width_m=billboard_width_m,
        billboard_height_m=billboard_height_m,
        method=METHOD,
        target=target,
        yaw_mean_deg=yaw_mean,
        pitch_mean_deg=pitch_mean,
        yaw_std_deg=yaw_std,
        pitch_std_deg=pitch_std,
        sample_count=len(samples),
        valid_duration_sec=round(valid_duration_sec, 1),
        yaw_tolerance_deg=yaw_tol,
        pitch_tolerance_deg=pitch_tol,
    )

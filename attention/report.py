"""Builds the session summary.json — one schema, used by the live field run and
by the offline simulator so they can never drift apart.

The summary is deliberately honest about what the numbers mean (see `limitations`
and `attention_tier`). It contains ONLY anonymous counters and privacy guarantees;
no age/gender/emotion/identity field is ever written here.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics
from .calibration import Calibration

ATTENTION_TIER = "LIKELY_ATTENTION"
METRIC_DEFINITION = "unique_valid_face_head_tracks_with_calibrated_head_pose_direction"

LIMITATIONS = [
    "total_passed means unique valid face/head tracks, not final unique pedestrian reach",
    "LIKELY_ATTENTION means calibrated head-pose direction, not verified geometric gaze",
    "track loss may overcount the same person (no re-identification by design)",
    "calibration is placement-specific and should be repeated for each billboard/camera setup",
]

# Privacy guarantees enforced by code (there is no code path that does any of these).
PRIVACY_BLOCK = {
    "raw_video_upload":      False,
    "frame_upload":          False,
    "face_crop_storage":     False,
    "face_embedding_storage": False,
    "age_gender_enabled":    False,
    "emotion_enabled":       False,
    "demographics_enabled":  False,
    "cloud_inference":       False,
    "cloud_upload":          False,
}


# Field phases. 'live' = ad-hoc run with no pass/fail expectation.
FIELD_PHASES = ("controlled", "middle_traffic", "high_traffic",
                "calibration", "simulation", "live")


def _band(value, target, pass_tol, warn_tol) -> str:
    """pass if |value-target|<=pass_tol, warning if <=warn_tol, else fail."""
    d = abs(value - target)
    if d <= pass_tol:
        return "pass"
    if d <= warn_tol:
        return "warning"
    return "fail"


def _worst(*statuses) -> str:
    order = {"fail": 3, "warning": 2, "pass": 1, "not_evaluated": 0}
    return max(statuses, key=lambda s: order.get(s, 0))


def compute_field_decision(phase: str, counts: dict,
                           manual_total=None, manual_lookers=None) -> dict:
    """Field sanity check (NOT scientific). Returns {status, reason, next_step}.

    Controlled / middle: compare system counts to operator's manual ground truth.
    High: always a stress test (not_evaluated for accuracy).
    """
    total = counts.get("total_passed", 0)
    looked = counts.get("looked_0_5s", 0)   # 0.5s is the stronger looker signal

    if phase == "high_traffic":
        return {
            "status": "not_evaluated",
            "reason": "High traffic is stress validation only. Do not use as "
                      "accuracy claim without manual validation.",
            "next_step": "If it ran without crashing and FPS stayed reasonable, the "
                         "stress test passed. Collect a debug bundle if anything broke.",
        }

    if manual_total is None or manual_lookers is None:
        msg = ("middle traffic: watch for stable FPS, no crashes, reasonable counters"
               if phase == "middle_traffic" else
               "no manual counts provided — cannot judge accuracy")
        return {
            "status": "not_evaluated",
            "reason": msg,
            "next_step": "Provide --manual-total and --manual-lookers (or answer the "
                         "end-of-run prompts) to get a pass/warning/fail.",
        }

    # Lenient bands: absolute floor, widened by a fraction for larger targets.
    total_tol_pass = max(3, 0.3 * manual_total)
    total_tol_warn = max(5, 0.5 * manual_total)
    look_tol_pass  = max(2, 0.3 * manual_lookers)
    look_tol_warn  = max(3, 0.5 * manual_lookers)

    total_status = _band(total, manual_total, total_tol_pass, total_tol_warn)
    look_status  = _band(looked, manual_lookers, look_tol_pass, look_tol_warn)
    status = _worst(total_status, look_status)

    reason = (f"total_passed={total} vs manual {manual_total} → {total_status}; "
              f"looked_0_5s={looked} vs manual lookers {manual_lookers} → {look_status}")
    next_step = {
        "pass":    "Looks sane — proceed to the next phase.",
        "warning": "Borderline — review placement/ROI/calibration; may proceed with caution.",
        "fail":    "Recheck calibration, camera framing, and the counting ROI before "
                   "trusting numbers. Re-run the controlled test.",
    }[status]
    return {"status": status, "reason": reason, "next_step": next_step}


def blank_manual_validation(test_type: str = "controlled") -> dict:
    """The operator-fillable block. Left null so the field team can record ground truth."""
    return {
        "manual_total_visible_faces": None,
        "manual_likely_lookers":      None,
        "test_type":                  test_type,   # controlled | middle_traffic | high_traffic
        "operator_notes":             "",
    }


def build_summary(
    *,
    session_id: str,
    started_at: str,
    ended_at: str,
    duration_sec: float,
    counts: dict,
    yaw_tolerance_deg: float,
    pitch_tolerance_deg: float,
    calibration_status: str,
    calibration_file: str | None,
    mode: str = "privacy_safe",
    test_phase: str = "live",
    counting_roi: tuple | list | None = None,
    field_decision: dict | None = None,
    manual_validation: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the summary dict. `counts` comes from metrics.count_buckets()."""
    summary = {
        "session_id":   session_id,
        "started_at":   started_at,
        "ended_at":     ended_at,
        "duration_sec": round(duration_sec, 1),
        "hardware": {
            "camera": "OAK-D Lite",
            "host":   "Raspberry Pi 4 8GB",
        },
        "mode":              mode,
        "test_phase":        test_phase,
        "metric_definition": METRIC_DEFINITION,
        "attention_tier":    ATTENTION_TIER,
        "calibration_status": calibration_status,
        "calibration_file":  calibration_file,
        "counting_roi":      list(counting_roi) if counting_roi is not None else None,
        "field_decision":    field_decision or {
            "status": "not_evaluated", "reason": "no field decision computed",
            "next_step": "",
        },
        "total_passed": counts.get("total_passed", 0),
        "looked_total": counts.get("looked_total", 0),
        "looked_0_3s":  counts.get("looked_0_3s", 0),
        "looked_0_5s":  counts.get("looked_0_5s", 0),
        "looked_1_0s":  counts.get("looked_1_0s", 0),
        "thresholds": {
            "min_track_sec":       metrics.MIN_TRACK_SECS,
            "look_thresholds_sec": list(metrics.LOOK_THRESHOLDS_SECS),
            "yaw_tolerance_deg":   round(yaw_tolerance_deg, 1),
            "pitch_tolerance_deg": round(pitch_tolerance_deg, 1),
        },
        "privacy":    dict(PRIVACY_BLOCK),
        "limitations": list(LIMITATIONS),
        "manual_validation": manual_validation or blank_manual_validation(),
    }
    if extra:
        summary.update(extra)
    return summary


def write_summary(summary: dict, path: str | Path) -> Path:
    """Write summary.json, creating parent dirs. Returns the resolved path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2) + "\n")
    return p


def calibration_tolerances(calib: Calibration | None,
                           fallback_yaw: float, fallback_pitch: float) -> tuple[float, float]:
    """Tolerances to report: from calibration if present, else the live settings."""
    if calib is not None:
        return calib.yaw_tolerance_deg, calib.pitch_tolerance_deg
    return fallback_yaw, fallback_pitch

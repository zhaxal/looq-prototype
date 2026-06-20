"""Pi-only NN output parsers and tracklet (depthai-object) geometry helpers.

The depthai-free attention calculations (LookState, iou, is_looking, best_match,
total_dwell) live in calc.py and are re-exported here for backwards-compatible
imports.
"""
from __future__ import annotations

from . import config
from .calc import LookState, iou, is_looking, best_match, total_dwell  # noqa: F401

__all__ = [
    "LookState", "iou", "is_looking", "best_match", "total_dwell",
    "tracklet_bbox", "tracklet_too_small",
    "extract_pose", "extract_age_gender", "extract_emotion", "parse_gathered",
    "verify_enums", "probe_tracklet", "probe_gathered",
]


# --- Tracklet geometry (depthai objects) -------------------------------------

def tracklet_bbox(t) -> tuple:
    return (t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height)  # VERIFY .roi


def tracklet_too_small(t) -> bool:
    """True when the tracklet bbox area is below MIN_FACE_AREA (normalized)."""
    return t.roi.width * t.roi.height < config.MIN_FACE_AREA


# --- NN output parsers -------------------------------------------------------

def extract_pose(msg) -> tuple[float, float, float]:
    """(yaw, pitch, roll) from a 3-head MessageGroup.
    Keys "0","1","2" → yaw/pitch/roll. # VERIFY key ordering on first device run.
    """
    try:
        return (float(msg["0"].prediction),
                float(msg["1"].prediction),
                float(msg["2"].prediction))
    except (KeyError, AttributeError) as e:
        raise RuntimeError(
            f"Pose parse failed; keys="
            f"{list(msg.keys()) if hasattr(msg, 'keys') else dir(msg)}"
        ) from e


def extract_age_gender(msg) -> tuple[str, int]:
    """(gender_str, age_int) from age_gender-62x62.
    Head "0" → age regression [0, 1] × 100 = years; head "1" → gender classification.
    """
    try:
        age    = round(float(msg["0"].prediction) * 100)
        gender = str(msg["1"].classes[0])
        return gender, age
    except (KeyError, AttributeError, IndexError) as e:
        raise RuntimeError(
            f"Age/gender parse failed; keys="
            f"{list(msg.keys()) if hasattr(msg, 'keys') else dir(msg)}"
        ) from e


def extract_emotion(msg) -> tuple[str, float]:
    """(label, confidence) from enet_b2_8_best Classifications (single-head model)."""
    try:
        return str(msg.classes[0]), float(msg.scores[0])
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Emotion parse failed; attrs={dir(msg)}") from e


def parse_gathered(msg) -> list[tuple[tuple, object]]:
    """[(det_bbox, payload), ...] from a GatheredData message.

    Returns [] on empty frames — GatherData can emit a message with no items or
    a None reference_data when no faces are present, which would raise AttributeError
    if read blindly.
    """
    ref   = getattr(msg, "reference_data", None)
    dets  = getattr(ref, "detections", None) if ref is not None else None
    items = getattr(msg, "items", None)
    if not dets or not items:
        return []
    return [
        ((det.xmin, det.ymin, det.xmax, det.ymax), items[i])
        for i, det in enumerate(dets)
        if i < len(items)                            # VERIFY .items accessor
    ]


# --- Startup self-test -------------------------------------------------------

def verify_enums(dai) -> None:
    """Check that enum values used in the pipeline actually exist.
    Prints a [VERIFY] warning for each missing name; does not raise.
    """
    checks = [
        (dai.Tracklet.TrackingStatus, "LOST"),
        (dai.Tracklet.TrackingStatus, "REMOVED"),
        (dai.TrackerType,             "SHORT_TERM_IMAGELESS"),
        (dai.TrackerIdAssignmentPolicy, "UNIQUE_ID"),
    ]
    for obj, name in checks:
        if not hasattr(obj, name):
            available = [x for x in dir(obj) if not x.startswith("_")]
            print(f"[VERIFY] {type(obj).__name__}.{name} not found — "
                  f"available: {available}")


def probe_tracklet(t) -> None:
    """On the first live tracklet, verify the .roi accessor.
    Prints a [VERIFY] warning if the name has drifted; does not raise.
    """
    if not hasattr(t, "roi"):
        available = [a for a in dir(t) if not a.startswith("_")]
        print(f"[VERIFY] Tracklet has no .roi — available: {available}")
        return
    for attr in ("x", "y", "width", "height"):
        if not hasattr(t.roi, attr):
            available = [a for a in dir(t.roi) if not a.startswith("_")]
            print(f"[VERIFY] Tracklet.roi has no .{attr} — available: {available}")


def probe_gathered(msg) -> None:
    """On the first non-empty GatheredData message, verify reference_data and items.
    Prints a [VERIFY] warning if names have drifted; does not raise.
    """
    ref = getattr(msg, "reference_data", None)
    if ref is None:
        available = [a for a in dir(msg) if not a.startswith("_")]
        print(f"[VERIFY] GatheredData has no .reference_data — available: {available}")
    else:
        if not hasattr(ref, "detections"):
            available = [a for a in dir(ref) if not a.startswith("_")]
            print(f"[VERIFY] reference_data has no .detections — available: {available}")
    if not hasattr(msg, "items"):
        available = [a for a in dir(msg) if not a.startswith("_")]
        print(f"[VERIFY] GatheredData has no .items — available: {available}")

"""Per-track state, geometry helpers, and head-pose parsing."""
from __future__ import annotations

from . import config


# --- Per-track debounce ------------------------------------------------------

class LookState:
    """Commits looking/not-looking only after the reading is stable for DEBOUNCE_SECS."""

    __slots__ = ("committed", "tentative", "since")

    def __init__(self) -> None:
        self.committed = False
        self.tentative = False
        self.since     = 0.0

    def update(self, looking: bool, now: float) -> bool:
        if looking != self.tentative:
            self.tentative = looking
            self.since     = now
        elif now - self.since >= config.DEBOUNCE_SECS:
            self.committed = self.tentative
        return self.committed


# --- Geometry ----------------------------------------------------------------

def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (xmin, ymin, xmax, ymax) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def is_looking_at_ad(yaw: float, pitch: float, settings: config.Settings) -> bool:
    """True when the head pose points at the ad, not the camera.

    The camera sits beside the ad, so a viewer looking at the ad has their head
    turned by ~settings.yaw_offset. We test the pose against that offset instead of
    against zero (which would mean "looking straight at the camera/operator").
    """
    return (abs(yaw   - settings.yaw_offset)   < settings.yaw_tol and
            abs(pitch - settings.pitch_offset) < settings.pitch_tol)


def tracklet_bbox(t) -> tuple:
    return (t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height)  # VERIFY .roi


def tracklet_too_small(t) -> bool:
    """True when the tracklet bbox area is below MIN_FACE_AREA (normalized)."""
    return t.roi.width * t.roi.height < config.MIN_FACE_AREA


def best_match(query: tuple, candidates: list[tuple[int, tuple]]) -> int:
    """Return the index of the candidate with the highest IoU above IOU_THRESHOLD, or -1."""
    best_i, best_score = -1, config.IOU_THRESHOLD
    for i, bbox in candidates:
        score = iou(query, bbox)
        if score > best_score:
            best_i, best_score = i, score
    return best_i


# --- Dwell time --------------------------------------------------------------

def total_dwell(tid: int, now: float,
                accum: dict[int, float], since: dict[int, float]) -> float:
    """Cumulative looking seconds for tid, including any ongoing streak."""
    total = accum.get(tid, 0.0)
    if tid in since:
        total += now - since[tid]
    return total


# --- NN output parsers -------------------------------------------------------

def extract_pose(msg) -> tuple[float, float, float]:
    """(yaw, pitch, roll) from a 3-head MessageGroup.
    Keys "0","1","2" -> yaw/pitch/roll. # VERIFY key ordering on first device run.
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


def parse_gathered(msg) -> list[tuple[tuple, object]]:
    """[(det_bbox, payload), ...] from a GatheredData message.

    Returns [] on empty frames — GatherData can emit a message with no items or a
    None reference_data when no faces are present, which would raise AttributeError
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
        (dai.CameraImageOrientation,  "ROTATE_180_DEG"),
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

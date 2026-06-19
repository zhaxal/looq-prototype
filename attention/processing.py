"""Per-track state, geometry helpers, and NN output parsers."""
from __future__ import annotations

from . import config


# --- Per-track debounce ------------------------------------------------------

class LookState:
    """Requires DEBOUNCE_FRAMES consecutive identical readings before committing."""

    __slots__ = ("committed", "tentative", "count")

    def __init__(self) -> None:
        self.committed = False
        self.tentative = False
        self.count     = 0

    def update(self, looking: bool) -> bool:
        if looking == self.tentative:
            self.count += 1
        else:
            self.tentative = looking
            self.count     = 1
        if self.count >= config.DEBOUNCE_FRAMES:
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


def is_looking(yaw: float, pitch: float) -> bool:
    return abs(yaw) < config.YAW_LIMIT and abs(pitch) < config.PITCH_LIMIT


def tracklet_bbox(t) -> tuple:
    return (t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height)  # VERIFY .roi


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

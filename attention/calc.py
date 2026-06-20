"""Pure attention calculations — debounce, geometry, dwell.

depthai-free so it can run anywhere (e.g. the GPU server). The Pi-only NN
message parsers and tracklet/depthai-object helpers live in processing.py.
"""
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


def is_looking(yaw: float, pitch: float) -> bool:
    return abs(yaw) < config.YAW_LIMIT and abs(pitch) < config.PITCH_LIMIT


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

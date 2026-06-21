"""Pure, hardware-free session metrics — the single source of truth for the
field numbers we report tomorrow.

This module has NO DepthAI / camera / OpenCV dependency on purpose:

* the live engine feeds it per-track timing,
* the offline simulator (`--simulate-poses`) feeds it synthetic tracks,
* the doctor self-test feeds it canned tracks,

so the counting logic can be verified without OAK-D Lite attached.

Definitions for tomorrow (see CLAUDE.md / docs):
    total_passed  = unique valid face/head tracks in the session
    looked_total  = tracks whose calibrated looking dwell >= 0.3 s  (== looked_0_3s)
    looked_0_3s   = tracks with looking dwell >= 0.3 s
    looked_0_5s   = tracks with looking dwell >= 0.5 s
    looked_1_0s   = tracks with looking dwell >= 1.0 s

A track is "valid" only if it persisted at least MIN_TRACK_SECS — this drops
single-frame detection blips that would otherwise inflate total_passed.
"""
from __future__ import annotations

from dataclasses import dataclass

# Tracks shorter than this (wall-clock first->last seen) are ignored entirely.
MIN_TRACK_SECS = 0.30
# Dwell thresholds, ascending. Tuple order matters for the named buckets below.
LOOK_THRESHOLDS_SECS = (0.3, 0.5, 1.0)


@dataclass
class TrackRecord:
    """One local ObjectTracker ID over one session.

    looking_accum_sec is the *committed* (debounced) time the head pose stayed
    near the calibrated billboard direction — not raw frame counts.
    """
    track_id: int
    first_seen: float
    last_seen: float
    looking_accum_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


def count_buckets(
    records,
    min_track_secs: float = MIN_TRACK_SECS,
    thresholds: tuple = LOOK_THRESHOLDS_SECS,
) -> dict:
    """Compute the five field numbers from an iterable of TrackRecord.

    Returns a dict with keys: total_passed, looked_total, looked_0_3s,
    looked_0_5s, looked_1_0s. Robust to extra/missing thresholds, but the
    named keys assume the default (0.3, 0.5, 1.0) ordering.
    """
    valid = [r for r in records if r.duration_sec >= min_track_secs]
    t03, t05, t10 = thresholds
    looked_0_3s = sum(1 for r in valid if r.looking_accum_sec >= t03)
    looked_0_5s = sum(1 for r in valid if r.looking_accum_sec >= t05)
    looked_1_0s = sum(1 for r in valid if r.looking_accum_sec >= t10)
    return {
        "total_passed": len(valid),
        "looked_total": looked_0_3s,   # looked_total is defined as the >=0.3s bucket
        "looked_0_3s":  looked_0_3s,
        "looked_0_5s":  looked_0_5s,
        "looked_1_0s":  looked_1_0s,
    }


class SessionMetrics:
    """Accumulates per-track records and reports the buckets.

    Calling update() again for the same track_id overwrites that track's record
    (the engine passes cumulative first/last/looking values, so the latest call
    is always the most complete).
    """

    def __init__(
        self,
        min_track_secs: float = MIN_TRACK_SECS,
        thresholds: tuple = LOOK_THRESHOLDS_SECS,
    ) -> None:
        self.min_track_secs = min_track_secs
        self.thresholds = thresholds
        self._tracks: dict[int, TrackRecord] = {}

    def update(self, track_id: int, first_seen: float, last_seen: float,
               looking_accum_sec: float) -> None:
        self._tracks[track_id] = TrackRecord(
            track_id, first_seen, last_seen, looking_accum_sec)

    @property
    def records(self) -> list[TrackRecord]:
        return list(self._tracks.values())

    def counts(self) -> dict:
        return count_buckets(self._tracks.values(), self.min_track_secs, self.thresholds)

"""Host-side IoU tracker — replaces the OAK ObjectTracker.

Mimics SHORT_TERM_IMAGELESS + UNIQUE_ID: greedy IoU association of detections to
existing tracks, monotonically increasing unique IDs, and a max-age so a track
survives a few missed frames before being dropped. Reuses attention.calc.iou.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from attention import config
from attention.calc import iou


@dataclass
class Track:
    id: int
    bbox: tuple                       # (xmin,ymin,xmax,ymax) normalized
    misses: int = 0                   # consecutive frames unmatched
    alive: bool = True
    hits: int = field(default=1)


class IoUTracker:
    def __init__(self, iou_threshold: float | None = None, max_age: int = 8) -> None:
        self.iou_threshold = (iou_threshold if iou_threshold is not None
                              else config.IOU_THRESHOLD)
        self.max_age = max_age
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[tuple]) -> tuple[list[Track], list[int]]:
        """Associate `detections` (bbox tuples) to tracks.

        Returns (active_tracks, removed_ids). active_tracks are matched/new this
        frame; removed_ids are tracks that aged out (signal a purge downstream).
        """
        unmatched_dets = set(range(len(detections)))
        # Greedy by descending IoU over all (track, det) pairs.
        pairs = []
        for tid, tr in self._tracks.items():
            for di, det in enumerate(detections):
                score = iou(tr.bbox, det)
                if score > self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        for _score, tid, di in pairs:
            if tid in matched_tracks or di not in unmatched_dets:
                continue
            tr = self._tracks[tid]
            tr.bbox = detections[di]
            tr.misses = 0
            tr.hits += 1
            matched_tracks.add(tid)
            unmatched_dets.discard(di)

        # New tracks for leftover detections.
        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(id=tid, bbox=detections[di])
            matched_tracks.add(tid)

        # Age unmatched tracks; remove past max_age.
        removed_ids: list[int] = []
        for tid, tr in list(self._tracks.items()):
            if tid in matched_tracks:
                continue
            tr.misses += 1
            if tr.misses > self.max_age:
                removed_ids.append(tid)
                del self._tracks[tid]

        active = [self._tracks[tid] for tid in matched_tracks]
        return active, removed_ids

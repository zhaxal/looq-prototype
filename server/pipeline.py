"""Server vision pipeline: decode → detect → track → per-face nets → calc.

Replaces the on-device DepthAI graph + main.py host loop. One instance per
connected Pi (one camera).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from attention import config

from .decode import H264Decoder
from .models import VisionModels, crop_face
from .session import AttentionSession
from .tracker import IoUTracker


def _bbox_area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class VisionPipeline:
    def __init__(self, models: VisionModels, log_path: str | None = None) -> None:
        self.models   = models
        self.decoder  = H264Decoder()
        self.tracker  = IoUTracker()
        self.session  = AttentionSession(log_path)
        self._pending_ts: deque[float] = deque(maxlen=120)
        self._last_print = 0.0
        self._frame_count = 0
        self.latest_counts: dict = {}

    def ingest(self, capture_ts: float, chunk: bytes) -> None:
        self._pending_ts.append(capture_ts)
        for frame in self.decoder.decode(chunk):
            ts = self._pending_ts.popleft() if self._pending_ts else time.time()
            self._process_frame(frame, ts)

    def _process_frame(self, frame: np.ndarray, now: float) -> None:
        detections = [bbox for bbox, _score in self.models.detector.detect(frame)
                      if _bbox_area(bbox) >= config.MIN_FACE_AREA]
        active_tracks, removed_ids = self.tracker.update(detections)

        observations: list[dict] = []
        for tr in active_tracks:
            crop = crop_face(frame, tr.bbox)
            if crop is None:
                continue
            yaw, pitch, _roll = self.models.head_pose(crop)
            obs = {"id": tr.id, "bbox": tr.bbox, "yaw": yaw, "pitch": pitch}

            if (self.models.age_gender is not None
                    and self.session.needs_age_gender(tr.id, now)):
                obs["age_gender"] = self.models.age_gender(crop)
            if (self.models.emotion is not None
                    and self.session.needs_emotion(tr.id, now)):
                obs["emotion"] = self.models.emotion(crop)
            observations.append(obs)

        self.latest_counts = self.session.process_frame(now, observations, removed_ids)
        self._frame_count += 1
        self._maybe_print()

    def _maybe_print(self) -> None:
        wall = time.time()
        if wall - self._last_print < 0.5:
            return
        elapsed = (wall - self._last_print) if self._last_print else 0.5
        fps = self._frame_count / elapsed
        self._last_print = wall
        self._frame_count = 0
        c = self.latest_counts
        ag = self.session.age_gender_cache
        emo = self.session.emotion_cache
        extras = ""
        if ag:
            extras += "  ag=" + str({k: f"{v[0][0]}{v[1]}" for k, v in ag.items()})
        if emo:
            extras += "  emo=" + str({k: v[0] for k, v in emo.items()})
        print(f"Looking: {c.get('looking_total', 0):>2}  "
              f"tracked: {c.get('tracked_total', 0):>2}  "
              f"ids: {c.get('looking_ids', [])}  "
              f"{fps:.1f}fps{extras}")

    def finalize(self) -> None:
        self.session.finalize()
        self.decoder.close()

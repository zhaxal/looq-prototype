"""AttentionSession — per-track state + attention calculations (server side).

Ported from main.py's run() loop (lines 87-269, 336-366). Uses Pi-supplied
capture timestamps as `now` so debounce/dwell are immune to network jitter, and
reuses the pure functions in attention.calc.
"""
from __future__ import annotations

from datetime import datetime

from attention import config
from attention.calc import LookState, is_looking, total_dwell

from .csvlog import CsvLogger, print_summary


class AttentionSession:
    def __init__(self, log_path: str | None = None) -> None:
        self.track_states:         dict[int, LookState]           = {}
        self.pose_cache:           dict[int, tuple[float, float]] = {}
        self.age_gender_cache:     dict[int, tuple[str, int]]     = {}
        self.emotion_cache:        dict[int, tuple[str, float]]   = {}
        self.last_age_gender_time: dict[int, float]               = {}
        self.last_emotion_time:    dict[int, float]               = {}
        self.looking_ids:          set[int]                       = set()
        self.look_accum:           dict[int, float]               = {}
        self.look_since:           dict[int, float]               = {}

        self.session_ids:   set[int]         = set()
        self.session_dwell: dict[int, float] = {}
        self.peak_looking:  int              = 0
        self.session_start: float            = 0.0
        self._last_now:     float            = 0.0
        self.active_ids:    set[int]         = set()

        self._csv = CsvLogger(log_path)

    # --- throttle queries (pipeline asks before running heavy nets) ----------

    def needs_age_gender(self, tid: int, now: float) -> bool:
        return now - self.last_age_gender_time.get(tid, 0.0) >= config.AGE_GENDER_INTERVAL

    def needs_emotion(self, tid: int, now: float) -> bool:
        return now - self.last_emotion_time.get(tid, 0.0) >= config.EMOTION_INTERVAL

    # --- per-frame update ----------------------------------------------------

    def process_frame(self, now: float, observations: list[dict],
                      removed_ids: list[int]) -> dict:
        """observations: [{id, bbox, yaw, pitch, age_gender?, emotion?}, ...]

        yaw/pitch are None if no pose this frame (→ POSE_UNSEEN). age_gender /
        emotion are fresh values to cache when present (already throttled by the
        caller via needs_*). Returns the counts snapshot.
        """
        if self.session_start == 0.0:
            self.session_start = now
        self._last_now = now

        for tid in removed_ids:
            self._write_row(tid, "removed", now)
            self._purge(tid)

        active: set[int] = set()
        for obs in observations:
            tid = obs["id"]
            active.add(tid)
            self.track_states.setdefault(tid, LookState())

            yaw, pitch = obs.get("yaw"), obs.get("pitch")
            if yaw is not None and pitch is not None:
                self.pose_cache[tid] = (yaw, pitch)
            yaw, pitch = self.pose_cache.get(tid, config.POSE_UNSEEN)

            if self.track_states[tid].update(is_looking(yaw, pitch), now):
                self.looking_ids.add(tid)
                self.look_since.setdefault(tid, now)
            else:
                self.looking_ids.discard(tid)
                if tid in self.look_since:
                    self.look_accum[tid] = (self.look_accum.get(tid, 0.0)
                                            + now - self.look_since.pop(tid))

            ag = obs.get("age_gender")
            if ag is not None:
                self.age_gender_cache[tid]     = ag
                self.last_age_gender_time[tid] = now

            emo = obs.get("emotion")
            if emo is not None:
                self.emotion_cache[tid]     = emo
                self.last_emotion_time[tid] = now

        # Purge tracks not present this frame (mirrors main.py:265-267).
        for tid in list(self.track_states):
            if tid not in active:
                self._purge(tid)

        self.active_ids = active
        self.peak_looking = max(self.peak_looking, len(self.looking_ids))

        for tid in active:
            self._write_row(tid, "tick", now)

        return self.counts()

    # --- output --------------------------------------------------------------

    def counts(self) -> dict:
        now = self._last_now
        return {
            "looking_total": len(self.looking_ids),
            "tracked_total": len(self.active_ids),
            "looking_ids":   sorted(self.looking_ids),
            "peak_looking":  self.peak_looking,
            "dwell": {tid: round(total_dwell(tid, now, self.look_accum, self.look_since), 1)
                      for tid in self.active_ids},
        }

    def finalize(self) -> None:
        end = self._last_now or self.session_start
        for tid in list(self.track_states):
            self.session_ids.add(tid)
            self.session_dwell[tid] = (self.session_dwell.get(tid, 0.0)
                                       + total_dwell(tid, end, self.look_accum, self.look_since))
        self._csv.close()
        print_summary(self.session_ids, self.session_dwell, self.peak_looking,
                      end - self.session_start if self.session_start else 0.0)

    # --- internals -----------------------------------------------------------

    def _purge(self, tid: int) -> None:
        self.session_ids.add(tid)
        self.session_dwell[tid] = (self.session_dwell.get(tid, 0.0)
                                   + total_dwell(tid, self._last_now,
                                                 self.look_accum, self.look_since))
        self.track_states.pop(tid, None)
        self.pose_cache.pop(tid, None)
        self.age_gender_cache.pop(tid, None)
        self.emotion_cache.pop(tid, None)
        self.last_age_gender_time.pop(tid, None)
        self.last_emotion_time.pop(tid, None)
        self.looking_ids.discard(tid)
        self.look_accum.pop(tid, None)
        self.look_since.pop(tid, None)

    def _write_row(self, tid: int, event: str, now: float) -> None:
        _yaw, _pitch    = self.pose_cache.get(tid, (None, None))
        _gender, _age   = self.age_gender_cache.get(tid, (None, None))
        _emo, _emo_conf = self.emotion_cache.get(tid, (None, None))
        self._csv.write_row({
            "ts":            datetime.now().isoformat(timespec="milliseconds"),
            "track_id":      tid,
            "event":         event,
            "looking":       int(tid in self.looking_ids),
            "look_seconds":  f"{total_dwell(tid, now, self.look_accum, self.look_since):.1f}",
            "yaw":           f"{_yaw:.1f}"      if _yaw      is not None else "",
            "pitch":         f"{_pitch:.1f}"    if _pitch    is not None else "",
            "age":           _age               if _age      is not None else "",
            "gender":        _gender            if _gender   is not None else "",
            "emotion":       _emo               if _emo      is not None else "",
            "emotion_conf":  f"{_emo_conf:.2f}" if _emo_conf is not None else "",
            "looking_total": len(self.looking_ids),
            "tracked_total": len(self.active_ids),
        })

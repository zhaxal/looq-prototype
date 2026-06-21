"""Capture/inference engine.

Runs the DepthAI pipeline and the host-side matching loop in a background thread,
publishing an annotated frame and live stats to a thread-safe `SharedState` that the
GUI (or CLI) polls. All controls (start/stop/calibrate/offset/reset) are thread-safe.
"""
from __future__ import annotations

import csv
import statistics
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2

from . import config, metrics, pipeline as att_pipeline
from .processing import (
    LookState, is_looking_at_ad, is_track_inside_roi, extract_pose, parse_gathered,
    tracklet_bbox, tracklet_too_small, best_match, total_dwell,
    verify_enums, probe_tracklet, probe_gathered,
)

# depthai is imported by attention.pipeline; the entry point must call
# config.load_dotenv() before importing this module so the Hub key is set.
import depthai as dai


# --- Snapshot published to the UI --------------------------------------------

@dataclass
class TrackView:
    tid:     int
    looking: bool
    dwell:   float
    yaw:     float | None
    pitch:   float | None


@dataclass
class SharedState:
    running:      bool = False
    frame:        object = None          # BGR numpy array, or None
    fps:          float = 0.0
    looking_now:  int = 0
    tracked_now:  int = 0
    total_unique: int = 0
    looked_count: int = 0
    peak_looking: int = 0
    avg_dwell:    float = 0.0
    tracks:       list[TrackView] = field(default_factory=list)
    calibrating:  bool = False
    calib_remaining: float = 0.0
    message:      str = ""               # transient status / calibrate result / error
    error:        bool = False
    # --- Field metrics (dwell buckets) — the numbers reported tomorrow ---------
    total_passed: int = 0                # unique valid tracks (>= MIN_TRACK_SECS)
    looked_total: int = 0                # tracks with looking dwell >= 0.3s
    looked_0_3:   int = 0
    looked_0_5:   int = 0
    looked_1_0:   int = 0
    frame_seq:    int = 0                # increments on every publish (change detector)


class Engine:
    """Owns the worker thread and the shared state snapshot."""

    def __init__(self, settings: config.Settings) -> None:
        self.settings = settings
        self._lock    = threading.Lock()
        self._shared  = SharedState()
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()
        self._reset   = threading.Event()
        self._calib_secs: float | None = None
        self._pub_seq = 0
        # Optional privacy-safe events.csv (set by the field runner before start()).
        # Schema: timestamp,track_id,event,looking,dwell_sec,yaw_deg,pitch_deg,reason.
        # No age/gender/emotion is ever written.
        self.events_csv_path: str | None = None
        # Optional counting ROI (normalized x1,y1,x2,y2). Tracks whose center is
        # outside it are ignored entirely (not counted, no dwell). None = full frame.
        self.counting_roi: tuple | None = None

    # --- Public controls (called from the GUI/main thread) -------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Signal the worker to stop; returns immediately (non-blocking)."""
        self._stop.set()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish (blocking)."""
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=6.0)
        self._thread = None

    def reset_session(self) -> None:
        self._reset.set()

    def calibrate(self, secs: float = 5.0) -> None:
        self._calib_secs = secs

    def nudge_yaw_offset(self, delta: float) -> None:
        with self._lock:
            self.settings.yaw_offset = round(self.settings.yaw_offset + delta, 1)
            self.settings.save()

    def set_offsets(self, yaw: float, pitch: float) -> None:
        with self._lock:
            self.settings.yaw_offset   = round(yaw, 1)
            self.settings.pitch_offset = round(pitch, 1)
            self.settings.save()

    def snapshot(self) -> SharedState:
        """Return a stable copy of the shared state for the UI."""
        with self._lock:
            return replace(self._shared, tracks=list(self._shared.tracks))

    # --- Worker thread -------------------------------------------------------

    # If no tracklet arrives for this many seconds while the pipeline claims to
    # be running, treat it as a silent disconnect and force a reconnect.
    _STALE_TIMEOUT = 5.0
    # Reconnect backoff: starts at this value, doubles each failure, caps at max.
    _RECONNECT_MIN = 3.0
    _RECONNECT_MAX = 30.0

    def _publish(self, **changes) -> None:
        with self._lock:
            self._pub_seq += 1
            self._shared = replace(self._shared, frame_seq=self._pub_seq, **changes)

    def _run(self) -> None:
        verify_enums(dai)

        # Session-level state lives here so it survives camera reconnects.
        session_ids:   set[int]         = set()
        session_dwell: dict[int, float] = {}
        # track_id -> [first_seen, last_seen]; used for the MIN_TRACK_SECS filter.
        session_seen:  dict[int, list]  = {}
        peak_looking   = 0
        session_start  = time.time()

        csv_file = csv_writer = None
        if self.settings.log:
            log_path = f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_file = open(log_path, "w", newline="", buffering=1)
            csv_writer = csv.DictWriter(csv_file, fieldnames=[
                "ts", "track_id", "event", "looking", "look_seconds",
                "yaw", "pitch", "looking_total", "tracked_total",
            ])
            csv_writer.writeheader()
            print(f"[log] writing to {log_path}")

        # Optional privacy-safe events.csv (field mode). Anonymous columns only.
        events_file = events_writer = None
        if self.events_csv_path:
            ep = Path(self.events_csv_path)
            ep.parent.mkdir(parents=True, exist_ok=True)
            events_file = open(ep, "w", newline="", buffering=1)
            events_writer = csv.DictWriter(events_file, fieldnames=[
                "timestamp", "track_id", "event", "looking",
                "dwell_sec", "yaw_deg", "pitch_deg", "reason",
            ])
            events_writer.writeheader()
            print(f"[events] writing to {ep}")

        reconnect_delay = 0.0

        while not self._stop.is_set():
            # Handle reset request between reconnect attempts too.
            if self._reset.is_set():
                self._reset.clear()
                session_ids   = set()
                session_dwell = {}
                session_seen  = {}
                peak_looking  = 0
                session_start = time.time()

            if reconnect_delay > 0:
                # Wait in small slices so we can respond to stop/reset.
                deadline = time.time() + reconnect_delay
                while time.time() < deadline and not self._stop.is_set():
                    remaining = deadline - time.time()
                    self._publish(
                        running=False, error=True,
                        message=f"Camera disconnected — reconnecting in {remaining:.0f}s…",
                    )
                    time.sleep(0.5)
                if self._stop.is_set():
                    break

            self._publish(running=False, error=False, message="Connecting…")

            try:
                peak_looking = self._one_session(
                    session_ids, session_dwell, session_seen, peak_looking,
                    session_start, csv_writer, events_writer,
                )
                # Clean exit (stop requested): leave the retry loop.
                if self._stop.is_set():
                    break
                # Pipeline ended on its own without an exception — unexpected;
                # treat it like a disconnect and retry.
                reconnect_delay = self._RECONNECT_MIN

            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[engine] error: {exc}")
                reconnect_delay = min(
                    max(reconnect_delay * 2, self._RECONNECT_MIN),
                    self._RECONNECT_MAX,
                )

        # Flush any faces still active at the moment we stopped.
        end = time.time()
        summary = self._summary(session_ids, session_dwell, peak_looking,
                                 end - session_start)
        print("\n" + summary)
        counts = self._bucket_counts(session_ids, session_dwell, session_seen)
        self._publish(
            running=False, error=False, message=summary,
            total_passed=counts["total_passed"], looked_total=counts["looked_total"],
            looked_0_3=counts["looked_0_3s"], looked_0_5=counts["looked_0_5s"],
            looked_1_0=counts["looked_1_0s"],
        )
        if csv_file:
            csv_file.close()
        if events_file:
            events_file.close()

    # --- Field metrics helpers ----------------------------------------------

    @staticmethod
    def _bucket_counts(session_ids, session_dwell, session_seen,
                       active=None) -> dict:
        """Build TrackRecords from session (+ optional live active) data and count
        the dwell buckets. `active` is an iterable of
        (tid, first_seen, last_seen, looking_accum) for not-yet-purged tracks.
        """
        recs: dict[int, metrics.TrackRecord] = {}
        for tid in session_ids:
            first, last = session_seen.get(tid, (0.0, 0.0))
            recs[tid] = metrics.TrackRecord(tid, first, last,
                                            session_dwell.get(tid, 0.0))
        for tid, first, last, accum in (active or ()):
            recs[tid] = metrics.TrackRecord(tid, first, last, accum)
        return metrics.count_buckets(recs.values())

    @staticmethod
    def _emit_event(writer, ts, tid, event, looking, dwell, yaw, pitch, reason) -> None:
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "track_id":  tid,
            "event":     event,
            "looking":   int(looking),
            "dwell_sec": f"{dwell:.2f}",
            "yaw_deg":   f"{yaw:.1f}"   if yaw   is not None else "",
            "pitch_deg": f"{pitch:.1f}" if pitch is not None else "",
            "reason":    reason,
        })

    def _one_session(
        self,
        session_ids:   set,
        session_dwell: dict,
        session_seen:  dict,
        peak_looking:  int,
        session_start: float,
        csv_writer,
        events_writer=None,
    ) -> int:
        """Run the pipeline until stop is requested, a disconnect occurs, or a
        stale-data timeout fires.  Returns updated peak_looking.
        Raises on any DepthAI / pipeline error so the caller can reconnect.
        """
        # Per-pipeline-session state (resets on every reconnect).
        track_states: dict[int, LookState]           = {}
        pose_cache:   dict[int, tuple[float, float]] = {}
        looking_ids:  set[int]                       = set()
        look_accum:   dict[int, float]               = {}
        look_since:   dict[int, float]               = {}

        calib_until    = 0.0
        calib_samples: list[tuple[float, float]] = []

        def purge(tid: int, now: float, reason: str = "track_lost") -> None:
            session_ids.add(tid)
            session_dwell[tid] = (session_dwell.get(tid, 0.0)
                                  + total_dwell(tid, now, look_accum, look_since))
            if tid in session_seen:
                session_seen[tid][1] = now
            if events_writer is not None and tid in session_seen:
                yaw, pitch = pose_cache.get(tid, (None, None))
                self._emit_event(events_writer, now, tid, "exit",
                                 tid in looking_ids, session_dwell[tid],
                                 yaw, pitch, reason)
            track_states.pop(tid, None)
            pose_cache.pop(tid, None)
            looking_ids.discard(tid)
            look_accum.pop(tid, None)
            look_since.pop(tid, None)

        with dai.Pipeline() as pipeline:
            queues = att_pipeline.build(pipeline, self.settings)
            pipeline.start()
            print("[camera] pipeline started")
            self._publish(running=True, error=False, message="")

            last_tick        = 0.0
            last_data        = time.time()   # watchdog: time of last received tracklet
            frame_count      = 0
            fps_actual       = 0.0
            latest_frame     = None
            _tracklet_probed = False
            _pose_probed     = False

            while pipeline.isRunning() and not self._stop.is_set():
                if self._reset.is_set():
                    self._reset.clear()
                    session_ids.clear(); session_dwell.clear(); session_seen.clear()
                    peak_looking  = 0

                if self._calib_secs is not None:
                    calib_until   = time.time() + self._calib_secs
                    calib_samples = []
                    self._calib_secs = None

                frame_msg = queues["frame"].tryGet()
                if frame_msg is not None:
                    latest_frame = frame_msg.getCvFrame()

                track_msg = queues["tracklets"].tryGet()
                if track_msg is None:
                    # Watchdog: no data for too long → force a reconnect.
                    if time.time() - last_data > self._STALE_TIMEOUT:
                        print("[engine] stale — no data; forcing reconnect")
                        raise RuntimeError("Stale pipeline — no tracklet data")
                    time.sleep(0.005)
                    continue

                last_data = time.time()   # data arrived; reset watchdog

                raw_poses: list[tuple[tuple, float, float]] = []
                pose_msg = queues["poses"].tryGet()
                if pose_msg is not None:
                    if not _pose_probed:
                        probe_gathered(pose_msg)
                        _pose_probed = True
                    for bbox, item in parse_gathered(pose_msg):
                        yaw, pitch, _ = extract_pose(item)
                        raw_poses.append((bbox, yaw, pitch))

                now        = time.time()
                active_ids: set[int] = set()
                pose_idx   = [(i, p[0]) for i, p in enumerate(raw_poses)]

                for t in track_msg.tracklets:
                    if t.status in (dai.Tracklet.TrackingStatus.LOST,    # VERIFY enum
                                    dai.Tracklet.TrackingStatus.REMOVED):
                        if csv_writer:
                            self._write_row(csv_writer, t.id, t.status.name.lower(),
                                            now, pose_cache, looking_ids, look_accum,
                                            look_since, len(looking_ids), len(active_ids))
                        purge(t.id, now, reason=t.status.name.lower())
                        continue

                    if tracklet_too_small(t):
                        continue

                    tb = tracklet_bbox(t)
                    # Counting ROI: ignore background faces outside the zone. A
                    # track outside the ROI is treated as not present, so it is
                    # neither counted nor accumulates looking dwell while outside.
                    if not is_track_inside_roi(tb, self.counting_roi):
                        continue

                    if not _tracklet_probed:
                        probe_tracklet(t)
                        _tracklet_probed = True

                    active_ids.add(t.id)
                    track_states.setdefault(t.id, LookState())
                    # First sighting of this track id → record first_seen + "enter".
                    if t.id not in session_seen:
                        session_seen[t.id] = [now, now]
                        if events_writer is not None:
                            self._emit_event(events_writer, now, t.id, "enter",
                                             False, 0.0, None, None, "track_started")
                    else:
                        session_seen[t.id][1] = now

                    best_i = best_match(tb, pose_idx)
                    if best_i >= 0:
                        pose_cache[t.id] = (raw_poses[best_i][1], raw_poses[best_i][2])

                    yaw, pitch = pose_cache.get(t.id, config.POSE_UNSEEN)
                    looking = is_looking_at_ad(yaw, pitch, self.settings)
                    was_looking = t.id in looking_ids
                    if track_states[t.id].update(looking, now):
                        looking_ids.add(t.id)
                        look_since.setdefault(t.id, now)
                        if events_writer is not None and not was_looking:
                            self._emit_event(events_writer, now, t.id, "look_start",
                                             True, total_dwell(t.id, now, look_accum, look_since),
                                             pose_cache.get(t.id, (None, None))[0],
                                             pose_cache.get(t.id, (None, None))[1],
                                             "pose_in_billboard_cone")
                    else:
                        looking_ids.discard(t.id)
                        if t.id in look_since:
                            look_accum[t.id] = (look_accum.get(t.id, 0.0)
                                                + now - look_since.pop(t.id))
                            if events_writer is not None and was_looking:
                                self._emit_event(events_writer, now, t.id, "look_end",
                                                 False, look_accum.get(t.id, 0.0),
                                                 pose_cache.get(t.id, (None, None))[0],
                                                 pose_cache.get(t.id, (None, None))[1],
                                                 "pose_left_billboard_cone")

                for tid in list(track_states):
                    if tid not in active_ids:
                        purge(tid, now)

                peak_looking = max(peak_looking, len(looking_ids))
                frame_count += 1

                # Calibration sampling: use the largest active face.
                if now < calib_until:
                    big_tid, big_area = None, 0.0
                    for t in track_msg.tracklets:
                        if t.id in active_ids:
                            area = t.roi.width * t.roi.height
                            if area > big_area:
                                big_tid, big_area = t.id, area
                    if big_tid is not None and big_tid in pose_cache:
                        calib_samples.append(pose_cache[big_tid])
                elif calib_samples and calib_until:
                    yaws    = statistics.median(s[0] for s in calib_samples)
                    pitches = statistics.median(s[1] for s in calib_samples)
                    self.set_offsets(yaws, pitches)
                    msg = f"Calibrated: yaw {yaws:+.0f}°, pitch {pitches:+.0f}°"
                    print(f"[calibrate] {msg}  ({len(calib_samples)} samples)")
                    self._publish(message=msg)
                    calib_samples, calib_until = [], 0.0

                if latest_frame is not None and now - last_tick >= 0.1:
                    if last_tick:
                        fps_actual = frame_count / (now - last_tick)
                    frame_count = 0

                    dwell_map = {tid: total_dwell(tid, now, look_accum, look_since)
                                 for tid in active_ids}
                    annotated = self._annotate(latest_frame, track_msg.tracklets,
                                               active_ids, looking_ids, pose_cache, dwell_map)

                    looked = sum(1 for d in session_dwell.values() if d > 0)
                    looked += sum(1 for tid in active_ids
                                  if total_dwell(tid, now, look_accum, look_since) > 0)
                    n_total = len(session_ids | active_ids)
                    dwells  = list(session_dwell.values()) + list(dwell_map.values())
                    looked_dwells = [d for d in dwells if d > 0]
                    avg_dwell = (sum(looked_dwells) / len(looked_dwells)) if looked_dwells else 0.0

                    tracks = [TrackView(tid, tid in looking_ids,
                                        dwell_map.get(tid, 0.0),
                                        *(pose_cache.get(tid, (None, None))))
                              for tid in sorted(active_ids)]

                    # Live dwell-bucket counts: session totals + currently-active tracks.
                    active_recs = [
                        (tid, session_seen.get(tid, [now, now])[0],
                         session_seen.get(tid, [now, now])[1], dwell_map.get(tid, 0.0))
                        for tid in active_ids
                    ]
                    counts = self._bucket_counts(session_ids, session_dwell,
                                                 session_seen, active_recs)

                    self._publish(
                        running=True, frame=annotated, fps=fps_actual,
                        looking_now=len(looking_ids), tracked_now=len(active_ids),
                        total_unique=n_total, looked_count=looked,
                        peak_looking=peak_looking, avg_dwell=avg_dwell,
                        tracks=tracks,
                        calibrating=now < calib_until,
                        calib_remaining=max(0.0, calib_until - now),
                        total_passed=counts["total_passed"], looked_total=counts["looked_total"],
                        looked_0_3=counts["looked_0_3s"], looked_0_5=counts["looked_0_5s"],
                        looked_1_0=counts["looked_1_0s"],
                    )

                    if csv_writer:
                        for t in track_msg.tracklets:
                            if t.id in active_ids:
                                self._write_row(csv_writer, t.id, "tick", now, pose_cache,
                                                looking_ids, look_accum, look_since,
                                                len(looking_ids), len(active_ids))
                    last_tick = now

            # Tell the GUI immediately — pipeline.stop() can take a few seconds.
            self._publish(running=False, looking_now=0, tracked_now=0,
                          tracks=[], frame=None, message="Stopping…")
            pipeline.stop()

        # Flush per-session faces into the session accumulators.
        end = time.time()
        for tid in list(track_states):
            purge(tid, end)

        return peak_looking

    # --- Helpers -------------------------------------------------------------

    @staticmethod
    def _annotate(frame, tracklets, active_ids, looking_ids, pose_cache, dwell_map):
        f = frame.copy()
        h, w = f.shape[:2]
        for t in tracklets:
            if t.id not in active_ids:
                continue
            x1, y1 = int(t.roi.x * w), int(t.roi.y * h)              # VERIFY .roi
            x2 = int((t.roi.x + t.roi.width) * w)
            y2 = int((t.roi.y + t.roi.height) * h)
            looking = t.id in looking_ids
            color   = (0, 220, 0) if looking else (0, 0, 220)        # BGR
            cv2.rectangle(f, (x1, y1), (x2, y2), color, 2)
            parts = [f"id{t.id}"]
            if t.id in pose_cache:
                yaw, pitch = pose_cache[t.id]
                parts.append(f"y{yaw:+.0f} p{pitch:+.0f}")
            if looking:
                parts.append(f"LOOK {dwell_map.get(t.id, 0.0):.1f}s")
            cv2.putText(f, "  ".join(parts), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return f

    @staticmethod
    def _write_row(writer, tid, event, now, pose_cache, looking_ids,
                   look_accum, look_since, looking_total, tracked_total) -> None:
        yaw, pitch = pose_cache.get(tid, (None, None))
        writer.writerow({
            "ts":            datetime.now().isoformat(timespec="milliseconds"),
            "track_id":      tid,
            "event":         event,
            "looking":       int(tid in looking_ids),
            "look_seconds":  f"{total_dwell(tid, now, look_accum, look_since):.1f}",
            "yaw":           f"{yaw:.1f}"   if yaw   is not None else "",
            "pitch":         f"{pitch:.1f}" if pitch is not None else "",
            "looking_total": looking_total,
            "tracked_total": tracked_total,
        })

    @staticmethod
    def _summary(session_ids, session_dwell, peak_looking, duration) -> str:
        n_total  = len(session_ids)
        n_looked = sum(1 for d in session_dwell.values() if d > 0)
        avg = (sum(d for d in session_dwell.values() if d > 0) / n_looked) if n_looked else 0.0
        hrs, rem = divmod(int(duration), 3600)
        mins, sec = divmod(rem, 60)
        pct = f"{n_looked / n_total:.0%}" if n_total else "0%"
        return (
            "SESSION SUMMARY\n"
            f"  Duration     : {hrs:02d}:{mins:02d}:{sec:02d}\n"
            f"  Unique faces : {n_total}\n"
            f"  Looked at ad : {n_looked}  ({pct})\n"
            f"  Peak looking : {peak_looking}\n"
            f"  Avg dwell    : {avg:.1f}s"
        )

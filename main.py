#!/usr/bin/env python3
"""OAK-D-Lite attention counter — count people looking at the camera.

Pipeline (all NN inference on the OAK VPU; host only parses and matches):

    Camera (RGB)  [or --test-video]
      └─ ParsingNeuralNetwork[YuNet]  → face ImgDetections
           ├─ ObjectTracker           → tracklets (stable IDs)
           └─ passthrough ────────────┬─ FrameCropper[60×60] → head-pose NN
                                      │       └─ GatherData → poses (synced)
                                      ├─ FrameCropper[62×62] → age/gender NN  (--age-gender)
                                      │       └─ GatherData → age/gender results
                                      └─ FrameCropper[64×64] → emotion NN     (--emotion)
                                              └─ GatherData → emotion results

Phase 5 models require converted blobs in models/:
  models/age_gender/  — age_gender-62x62 superblob + config.json
  models/emotion/     — enet_b2_8_best superblob + config.json
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from attention.config import (
    load_dotenv,
    DEFAULT_FPS, FACE_RESOLUTIONS, AGE_GENDER_INTERVAL, EMOTION_INTERVAL,
    YAW_LIMIT, PITCH_LIMIT, DEBOUNCE_SECS, POSE_UNSEEN,
)
from attention import pipeline as att_pipeline
from attention.display import LiveDisplay, draw_preview
from attention.processing import (
    LookState, is_looking,
    extract_pose, extract_age_gender, extract_emotion,
    parse_gathered, tracklet_bbox, tracklet_too_small, best_match, total_dwell,
    verify_enums, probe_tracklet, probe_gathered,
)

load_dotenv()

import depthai as dai  # noqa: E402 — must come after load_dotenv sets hub API key

verify_enums(dai)


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAK-D-Lite attention counter")
    p.add_argument("--preview",      action="store_true",
                   help="annotated OpenCV window (laptop only)")
    p.add_argument("--fps",          type=float, default=DEFAULT_FPS)
    p.add_argument("--face-res",     default="960x720", choices=FACE_RESOLUTIONS,
                   help="YuNet input resolution; 320x240 recommended on Pi")
    p.add_argument("--age-gender",   action="store_true",
                   help="enable age/gender branch (models/age_gender/ required)")
    p.add_argument("--emotion",      action="store_true",
                   help="enable emotion branch (models/emotion/ required)")
    p.add_argument("--log",          nargs="?", const="", metavar="PATH",
                   help="write CSV session log; omit PATH for an auto-named file")
    p.add_argument("--tui",          action="store_true",
                   help="live in-place terminal dashboard (useful over SSH)")
    p.add_argument("--looking-gate", action="store_true",
                   help="run age/gender + emotion only on confirmed-looking faces "
                        "(efficient for crowded scenes; requires --age-gender / --emotion)")
    p.add_argument("--test-video",   type=Path, metavar="VIDEO",
                   help="replay a video file instead of the live camera "
                        "(demo / performance testing)")
    return p.parse_args()


# --- Session log -------------------------------------------------------------

_CSV_FIELDS = [
    "ts", "track_id", "event", "looking", "look_seconds",
    "yaw", "pitch",
    "age", "gender",
    "emotion", "emotion_conf",
    "looking_total", "tracked_total",
]


# --- Main loop ---------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    track_states:      dict[int, LookState]           = {}
    pose_cache:        dict[int, tuple[float, float]] = {}
    age_gender_cache:  dict[int, tuple[str, int]]     = {}
    emotion_cache:     dict[int, tuple[str, float]]   = {}
    last_age_gender_time: dict[int, float]             = {}
    last_emotion_time: dict[int, float]               = {}
    looking_ids:       set[int]                       = set()
    look_accum:        dict[int, float]               = {}
    look_since:        dict[int, float]               = {}

    session_ids:   set[int]         = set()
    session_dwell: dict[int, float] = {}
    peak_looking:  int              = 0
    session_start: float            = 0.0

    display = LiveDisplay(args.face_res, args.fps) if args.tui else None

    csv_file = csv_writer = None
    if args.log is not None:
        log_path   = args.log or f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file   = open(log_path, "w", newline="", buffering=1)
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        csv_writer.writeheader()
        print(f"[log] writing to {log_path}")

    def _purge(tid: int) -> None:
        session_ids.add(tid)
        session_dwell[tid] = (session_dwell.get(tid, 0.0)
                              + total_dwell(tid, time.time(), look_accum, look_since))
        track_states.pop(tid, None)
        pose_cache.pop(tid, None)
        age_gender_cache.pop(tid, None)
        emotion_cache.pop(tid, None)
        last_age_gender_time.pop(tid, None)
        last_emotion_time.pop(tid, None)
        looking_ids.discard(tid)
        look_accum.pop(tid, None)
        look_since.pop(tid, None)

    def _write_row(tid: int, event: str, now: float) -> None:
        if csv_writer is None:
            return
        _yaw, _pitch    = pose_cache.get(tid, (None, None))
        _gender, _age   = age_gender_cache.get(tid, (None, None))
        _emo, _emo_conf = emotion_cache.get(tid, (None, None))
        csv_writer.writerow({
            "ts":            datetime.now().isoformat(timespec="milliseconds"),
            "track_id":      tid,
            "event":         event,
            "looking":       int(tid in looking_ids),
            "look_seconds":  f"{total_dwell(tid, now, look_accum, look_since):.1f}",
            "yaw":           f"{_yaw:.1f}"      if _yaw      is not None else "",
            "pitch":         f"{_pitch:.1f}"    if _pitch    is not None else "",
            "age":           _age               if _age      is not None else "",
            "gender":        _gender            if _gender   is not None else "",
            "emotion":       _emo               if _emo      is not None else "",
            "emotion_conf":  f"{_emo_conf:.2f}" if _emo_conf is not None else "",
            "looking_total": len(looking_ids),
            "tracked_total": len(active_ids),
        })

    quit_app = False
    while not quit_app:
        started_ok = False
        try:
            with dai.Pipeline() as pipeline:
                queues     = att_pipeline.build(pipeline, args)
                pipeline.start()
                started_ok = True
                if session_start == 0.0:
                    session_start = time.time()
                print("[camera] pipeline started")

                gates            = queues.get("_gates", {})
                last_log         = 0.0
                frame_count      = 0
                fps_actual       = None
                _tracklet_probed = False
                _pose_probed     = False

                while pipeline.isRunning() and not quit_app:
                    # Block on tracklets — the per-frame driver.
                    # v3 has no pipeline.processTasks(); host nodes run in their
                    # own threads after pipeline.start().
                    track_msg = queues["tracklets"].get()
                    if track_msg is None:
                        continue

                    # --- Pose ---
                    raw_poses: list[tuple[tuple, float, float]] = []
                    pose_msg = queues["poses"].tryGet()
                    if pose_msg is not None:
                        if not _pose_probed:
                            probe_gathered(pose_msg)
                            _pose_probed = True
                        for bbox, item in parse_gathered(pose_msg):
                            yaw, pitch, _ = extract_pose(item)
                            raw_poses.append((bbox, yaw, pitch))

                    # --- Feed looking-gate for heavy branches ---
                    if gates and pose_msg is not None:
                        look_dets = None
                        for _name, in_q in gates.items():
                            if look_dets is None:
                                look_dets = att_pipeline.looking_detections(
                                    pose_msg, raw_poses
                                )
                            in_q.send(look_dets)

                    # --- Age / gender ---
                    raw_ag: list[tuple[tuple, object]] = []
                    if "age_gender" in queues:
                        ag_msg = queues["age_gender"].tryGet()
                        if ag_msg is not None:
                            raw_ag = parse_gathered(ag_msg)

                    # --- Emotion ---
                    raw_emo: list[tuple[tuple, object]] = []
                    if "emotion" in queues:
                        emo_msg = queues["emotion"].tryGet()
                        if emo_msg is not None:
                            raw_emo = parse_gathered(emo_msg)

                    # --- Per-track state update ---
                    now        = time.time()
                    active_ids: set[int] = set()

                    pose_idx = [(i, p[0]) for i, p in enumerate(raw_poses)]
                    ag_idx   = [(i, b) for i, (b, _) in enumerate(raw_ag)]
                    emo_idx  = [(i, b) for i, (b, _) in enumerate(raw_emo)]

                    for t in track_msg.tracklets:
                        if t.status in (dai.Tracklet.TrackingStatus.LOST,    # VERIFY enum
                                        dai.Tracklet.TrackingStatus.REMOVED):
                            _write_row(t.id, t.status.name.lower(), now)
                            _purge(t.id)
                            continue

                        if tracklet_too_small(t):
                            continue

                        if not _tracklet_probed:
                            probe_tracklet(t)
                            _tracklet_probed = True

                        active_ids.add(t.id)
                        if t.id not in track_states:
                            track_states[t.id] = LookState()

                        tb = tracklet_bbox(t)

                        best_i = best_match(tb, pose_idx)
                        if best_i >= 0:
                            pose_cache[t.id] = (raw_poses[best_i][1], raw_poses[best_i][2])

                        yaw, pitch = pose_cache.get(t.id, POSE_UNSEEN)
                        if track_states[t.id].update(is_looking(yaw, pitch), now):
                            looking_ids.add(t.id)
                            if t.id not in look_since:
                                look_since[t.id] = now
                        else:
                            looking_ids.discard(t.id)
                            if t.id in look_since:
                                look_accum[t.id] = (look_accum.get(t.id, 0.0)
                                                    + now - look_since.pop(t.id))

                        if raw_ag and now - last_age_gender_time.get(t.id, 0) >= AGE_GENDER_INTERVAL:
                            best_i = best_match(tb, ag_idx)
                            if best_i >= 0:
                                age_gender_cache[t.id]     = extract_age_gender(raw_ag[best_i][1])
                                last_age_gender_time[t.id] = now

                        if raw_emo and now - last_emotion_time.get(t.id, 0) >= EMOTION_INTERVAL:
                            best_i = best_match(tb, emo_idx)
                            if best_i >= 0:
                                emotion_cache[t.id]     = extract_emotion(raw_emo[best_i][1])
                                last_emotion_time[t.id] = now

                    for tid in list(track_states):
                        if tid not in active_ids:
                            _purge(tid)

                    peak_looking = max(peak_looking, len(looking_ids))
                    frame_count += 1

                    # --- Throttled output (0.5 s) ---
                    if now - last_log >= 0.5:
                        elapsed     = now - last_log if last_log else 0.5
                        fps_actual  = frame_count / elapsed
                        frame_count = 0
                        last_log    = now

                        if display is not None:
                            dwell_map = {tid: total_dwell(tid, now, look_accum, look_since)
                                         for tid in active_ids}
                            display.update(
                                looking_ids, active_ids, track_msg.tracklets,
                                pose_cache, age_gender_cache, emotion_cache,
                                fps_actual, dwell_map,
                            )
                        else:
                            extras = ""
                            if age_gender_cache:
                                extras += "  ag=" + str(
                                    {k: f"{v[0][0]}{v[1]}" for k, v in age_gender_cache.items()}
                                )
                            if emotion_cache:
                                extras += "  emo=" + str(
                                    {k: v[0] for k, v in emotion_cache.items()}
                                )
                            print(f"Looking: {len(looking_ids):>2}  "
                                  f"tracked: {len(active_ids):>2}  "
                                  f"ids: {sorted(looking_ids)}{extras}")

                        for t in track_msg.tracklets:
                            if t.status not in (dai.Tracklet.TrackingStatus.LOST,
                                                dai.Tracklet.TrackingStatus.REMOVED):
                                _write_row(t.id, "tick", now)

                    # --- Optional preview ---
                    if args.preview:
                        frame_msg = queues["frame"].tryGet()
                        if frame_msg is not None:
                            draw_preview(frame_msg, track_msg.tracklets, pose_cache,
                                         age_gender_cache, emotion_cache, looking_ids)

                pipeline.stop()

        except KeyboardInterrupt:
            quit_app = True
        except Exception as exc:
            import traceback
            print(f"[main] error: {exc}")
            traceback.print_exc()
            if not started_ok:
                # Build-time failure (wrong API name, bad model archive, drifted
                # VERIFY accessor) — retrying just hides the traceback.
                print("[main] error before pipeline start — exiting")
                break
            time.sleep(2.0)

    if display:
        display.close()
    if csv_file:
        csv_file.close()
    if args.preview:
        import cv2
        cv2.destroyAllWindows()

    # Flush any faces still active at exit (never received LOST/REMOVED).
    _end = time.time()
    for tid in list(track_states):
        session_ids.add(tid)
        session_dwell[tid] = (session_dwell.get(tid, 0.0)
                              + total_dwell(tid, _end, look_accum, look_since))

    _print_summary(session_ids, session_dwell, peak_looking,
                   _end - session_start if session_start else 0.0)


def _print_summary(session_ids: set, session_dwell: dict,
                   peak_looking: int, duration: float) -> None:
    n_total  = len(session_ids)
    n_looked = sum(1 for d in session_dwell.values() if d > 0)
    avg_dwell = (sum(session_dwell.values()) / n_looked) if n_looked else 0.0
    hrs, rem  = divmod(int(duration), 3600)
    mins, sec = divmod(rem, 60)
    W = 47
    print(f"\n{'═' * W}")
    print(f"  SESSION SUMMARY")
    print(f"{'═' * W}")
    print(f"  Duration      : {hrs:02d}:{mins:02d}:{sec:02d}")
    print(f"  Unique faces  : {n_total}")
    if n_total:
        print(f"  Looked at you : {n_looked}  ({n_looked / n_total:.0%})")
    else:
        print(f"  Looked at you : 0")
    print(f"  Peak looking  : {peak_looking}")
    print(f"  Avg dwell     : {avg_dwell:.1f}s  (among faces that looked)")
    print(f"{'═' * W}")


def main() -> None:
    args = parse_args()

    active = ["pose"]
    if args.age_gender: active.append("age/gender")
    if args.emotion:    active.append("emotion")
    source = f"video:{args.test_video}" if args.test_video else "camera"

    print(f"Branches: {', '.join(active)}  |  "
          f"face-res: {args.face_res}  fps: {args.fps}  source: {source}")
    print(f"Thresholds: |yaw|<{YAW_LIMIT}  |pitch|<{PITCH_LIMIT} deg  "
          f"debounce: {DEBOUNCE_SECS}s")
    print("Starting pipeline... (Ctrl-C to stop)\n")

    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attention counter for OAK-D-Lite — Phases 1-5.

Pipeline (all NN inference on the OAK VPU; host only parses + matches):

    Camera (RGB)
      └─ ParsingNeuralNetwork[YuNet]  → face ImgDetections
           ├─ ObjectTracker           → tracklets (stable IDs)
           └─ passthrough ────────────┬─ FrameCropper[60×60] → head-pose NN
                                      │       └─ GatherData → poses (synced)
                                      ├─ FrameCropper[62×62] → age/gender NN  (--age-gender)
                                      │       └─ GatherData → ag results
                                      └─ FrameCropper[64×64] → emotion NN     (--emotion)
                                              └─ GatherData → emo results

cam_out feeds only face_nn; everything else reuses face_nn.passthrough.

Phase 5 adds:
  - Age/gender cached per track ID (re-run only for new tracks, not every frame).
  - Emotion throttled to EMOTION_INTERVAL s per track (heavy net on Pi 4 budget).
  - Both features are opt-in (--age-gender / --emotion) until model blobs are
    confirmed — see MODEL CONVERSION note below.

Local model blobs (place next to this script):
  age_gender-62x62.rvc2.tar.xz  — Age-Gender_Recognition, 62×62, 2-head
  enet_b2_8_best.rvc2.tar.xz    — Emotion_Recognition, 260×260, 8-class

NOTE: Lines marked `# VERIFY` use API details that can drift between
depthai / depthai-nodes releases — run once on the OAK and adjust.
"""
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    FrameCropper,
    GatherData,
)

# --- Config -----------------------------------------------------------------
_HERE = Path(__file__).parent

POSE_MODEL        = "luxonis/head-pose-estimation:60x60"
POSE_INPUT        = (60, 60)

# Local model archives (downloaded from HubAI, placed next to this script).
AGE_GENDER_ARCHIVE = _HERE / "age_gender-62x62.rvc2.tar.xz"
EMOTION_ARCHIVE    = _HERE / "enet_b2_8_best.rvc2.tar.xz"
AGE_GENDER_INPUT  = (62, 62)
EMOTION_INPUT     = (260, 260)   # enet_b2_8_best expects 260×260

FACE_RESOLUTIONS  = ("320x240", "640x360", "640x480")

FPS               = 12
FACE_CONF         = 0.6
YAW_LIMIT         = 20.0
PITCH_LIMIT       = 15.0
DEBOUNCE_FRAMES   = 3
IOU_MATCH         = 0.2
POSE_UNSEEN       = (90.0, 90.0)

EMOTION_INTERVAL  = 2.0   # seconds between per-track emotion cache updates
# 8-class model from config.json
EMOTION_CLASSES   = ["Anger", "Contempt", "Disgust", "Fear",
                     "Happiness", "Neutral", "Sadness", "Surprise"]
GENDER_CLASSES    = ["Female", "Male"]


# --- Debounce ---------------------------------------------------------------

class LookState:
    """Per-track debounce: requires DEBOUNCE_FRAMES consecutive frames
    of the same raw value before committed state updates."""
    __slots__ = ("committed", "tentative", "count")

    def __init__(self):
        self.committed = False
        self.tentative = False
        self.count = 0

    def update(self, looking: bool) -> bool:
        if looking == self.tentative:
            self.count += 1
        else:
            self.tentative = looking
            self.count = 1
        if self.count >= DEBOUNCE_FRAMES:
            self.committed = self.tentative
        return self.committed


# --- Geometry / pose --------------------------------------------------------

def boxes_overlap(a, b):
    """IoU of two (xmin, ymin, xmax, ymax) boxes in normalised coords."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def is_looking(yaw, pitch):
    return abs(yaw) < YAW_LIMIT and abs(pitch) < PITCH_LIMIT


def extract_pose(msg):
    """(yaw, pitch, roll) from a 3-head MessageGroup.
    Keys "0","1","2" → yaw/pitch/roll, each with .prediction (float).
    # VERIFY key ordering on first device run.
    """
    try:
        return (float(msg["0"].prediction),
                float(msg["1"].prediction),
                float(msg["2"].prediction))
    except (KeyError, AttributeError) as e:
        raise RuntimeError(
            f"Pose parse failed; keys={list(msg.keys()) if hasattr(msg, 'keys') else dir(msg)}"
        ) from e


def extract_age_gender(msg):
    """(gender_str, age_int) from age_gender-62x62 MessageGroup.

    config.json heads:
      "0" → RegressionParser on age_conv3: .prediction in [0, 1] → × 100 = years
      "1" → ClassificationParser on prob: classes=["Female","Male"], softmax=True
    """
    try:
        age    = round(float(msg["0"].prediction) * 100)
        gender = str(msg["1"].classes[0])
        return gender, age
    except (KeyError, AttributeError, IndexError) as e:
        raise RuntimeError(
            f"Age/gender parse failed; keys={list(msg.keys()) if hasattr(msg, 'keys') else dir(msg)}"
        ) from e


def extract_emotion(msg):
    """(emotion_label, confidence) from enet_b2_8_best Classifications message.

    Single-head model → GatherData carries the Classifications message directly
    (not a MessageGroup), so read .classes / .scores off it without a "0" key.
    classes = [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
    """
    try:
        label = str(msg.classes[0])
        score = float(msg.scores[0])
        return label, score
    except (AttributeError, IndexError) as e:
        raise RuntimeError(
            f"Emotion parse failed; attrs={dir(msg)}"
        ) from e


# --- Pipeline ---------------------------------------------------------------

def _patch_frame_cropper():
    """OAK script engine doesn't accept keyword args for Size2f or
    addCropRotatedRect.  Patch the template to use positional args."""
    from string import Template
    tmpl = FrameCropper.IMG_DETECTIONS_SCRIPT_CONTENT.template
    tmpl = tmpl.replace(
        "Size2f(width=rot.size.width + 2*p, height=rot.size.height + 2*p, normalized=True)",
        "Size2f(rot.size.width + 2*p, rot.size.height + 2*p)",
    )
    tmpl = tmpl.replace(
        "cfg.addCropRotatedRect(rect=pad_rotated_rect(rot_rect, PADDING), normalizedCoords=True)",
        "cfg.addCropRotatedRect(pad_rotated_rect(rot_rect, PADDING), True)",
    )
    tmpl = "\n".join(
        ln for ln in tmpl.split("\n") if "setTimestampDevice" not in ln
    )
    FrameCropper.IMG_DETECTIONS_SCRIPT_CONTENT = Template(tmpl)


def _make_face_branch(pipeline, face_detections, face_image, input_size,
                      nn_source, fps, multi_head=True):
    """FrameCropper → ParsingNeuralNetwork → GatherData for a face-crop model.

    face_detections: face_nn.out  (ImgDetections — drives cropper + GatherData ref)
    face_image:      face_nn.passthrough  (ImgFrame — pixel data for cropper)
    nn_source:       zoo slug string or dai.NNArchive for local blobs
    multi_head:      True for models with ≥2 heads (use nn.outputs → MessageGroup);
                     False for single-head models (use nn.out → parser message).
    """
    cropper = pipeline.create(FrameCropper).fromImgDetections(
        inputImgDetections=face_detections,
        outputSize=input_size,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    ).build(inputImage=face_image)
    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cropper.out, nnSource=nn_source
    )
    # outputs (MessageGroup) needs ≥2 heads; out is the lone parser otherwise.
    nn_data = nn.outputs if multi_head else nn.out
    gathered = pipeline.create(GatherData).build(
        inputData=nn_data,
        inputReference=face_detections,
        cameraFps=fps,
    )
    return gathered.out.createOutputQueue(maxSize=2, blocking=False)  # VERIFY kwargs


def build_pipeline(pipeline, args):
    _patch_frame_cropper()

    cam_w, cam_h = (int(x) for x in args.face_res.split("x"))
    face_model   = f"luxonis/yunet:{args.face_res}"

    cam     = pipeline.create(dai.node.Camera).build()  # no socket — proven on RPi4
    cam_out = cam.requestOutput((cam_w, cam_h), dai.ImgFrame.Type.BGR888p, fps=args.fps)

    # Stage 1: face detection — cam_out has exactly one consumer.
    face_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cam_out, nnSource=face_model  # VERIFY arg names
    )
    try:
        face_nn.setConfidenceThreshold(FACE_CONF)  # VERIFY available on parser
    except AttributeError:
        pass

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)         # VERIFY enum
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    face_nn.out.link(tracker.inputDetections)
    face_nn.passthrough.link(tracker.inputDetectionFrame)
    face_nn.passthrough.link(tracker.inputTrackerFrame)

    # All downstream crop branches reuse face_nn.passthrough — no extra cam copies.
    queues = {
        "tracklets": tracker.out.createOutputQueue(maxSize=2, blocking=False),
        "poses": _make_face_branch(
            pipeline, face_nn.out, face_nn.passthrough, POSE_INPUT, POSE_MODEL, args.fps
        ),
    }

    if args.age_gender:
        queues["age_gender"] = _make_face_branch(
            pipeline, face_nn.out, face_nn.passthrough, AGE_GENDER_INPUT,
            dai.NNArchive(AGE_GENDER_ARCHIVE), args.fps
        )
    if args.emotion:
        queues["emotion"] = _make_face_branch(
            pipeline, face_nn.out, face_nn.passthrough, EMOTION_INPUT,
            dai.NNArchive(EMOTION_ARCHIVE), args.fps, multi_head=False
        )
    if args.preview:
        queues["frame"] = face_nn.passthrough.createOutputQueue(maxSize=2, blocking=False)

    return queues


# --- Preview ----------------------------------------------------------------

def draw_preview(frame_msg, tracklets, pose_cache, age_gender_cache,
                 emotion_cache, looking_ids):
    frame = frame_msg.getCvFrame()
    h, w  = frame.shape[:2]

    for t in tracklets:
        x1 = int(t.roi.x * w)                          # VERIFY roi accessor
        y1 = int(t.roi.y * h)
        x2 = int((t.roi.x + t.roi.width) * w)
        y2 = int((t.roi.y + t.roi.height) * h)

        looking = t.id in looking_ids
        color   = (0, 255, 0) if looking else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = [f"id={t.id}"]
        if t.id in pose_cache:
            yaw, pitch = pose_cache[t.id]
            parts.append(f"y={yaw:+.0f} p={pitch:+.0f}")
        if t.id in age_gender_cache:
            gender, age = age_gender_cache[t.id]
            parts.append(f"{gender[0].upper()}{age}")
        if t.id in emotion_cache:
            label, score = emotion_cache[t.id]
            parts.append(f"{label}({score:.0%})")

        cv2.putText(frame, "  ".join(parts), (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Looking: {len(looking_ids)}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow("attention", frame)
    cv2.waitKey(1)


# --- Helpers ----------------------------------------------------------------

def _bbox_for_tracklet(t):
    return (t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height)


def _best_match(tb, indexed_bboxes):
    """Return index of best IoU match, or -1 if none exceeds IOU_MATCH."""
    best_i, best_iou = -1, IOU_MATCH
    for i, bbox in indexed_bboxes:
        iou = boxes_overlap(tb, bbox)
        if iou > best_iou:
            best_i, best_iou = i, iou
    return best_i


def _parse_gathered(msg):
    """Return list of (det_bbox, MessageGroup_item) from a GatheredData message."""
    out = []
    for i, item in enumerate(msg.items):        # VERIFY .items accessor
        dets = msg.reference_data.detections
        if i >= len(dets):
            break
        det = dets[i]
        out.append(((det.xmin, det.ymin, det.xmax, det.ymax), item))
    return out


# --- Main -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OAK attention counter")
    p.add_argument("--preview",    action="store_true",
                   help="annotated OpenCV window (laptop only)")
    p.add_argument("--fps",        type=float, default=FPS)
    p.add_argument("--face-res",   default="640x480", choices=FACE_RESOLUTIONS,
                   help="YuNet input resolution; 320x240 recommended on Pi")
    p.add_argument("--age-gender", action="store_true",
                   help="enable age/gender branch (age_gender-62x62.rvc2.tar.xz)")
    p.add_argument("--emotion",    action="store_true",
                   help="enable emotion branch (enet_b2_8_best.rvc2.tar.xz)")
    return p.parse_args()


def main():
    args = parse_args()

    active_branches = ["pose"]
    if args.age_gender: active_branches.append("age/gender")
    if args.emotion:    active_branches.append("emotion")
    print(f"Branches: {', '.join(active_branches)}  |  face-res: {args.face_res}  fps: {args.fps}")
    print(f"Thresholds: |yaw|<{YAW_LIMIT}  |pitch|<{PITCH_LIMIT} deg  "
          f"debounce: {DEBOUNCE_FRAMES} frames")
    print("Starting pipeline... (Ctrl-C to stop)\n")

    track_states:      dict[int, LookState]            = {}
    pose_cache:        dict[int, tuple[float, float]]  = {}
    age_gender_cache:  dict[int, tuple[str, int]]      = {}
    emotion_cache:     dict[int, tuple[str, float]]    = {}
    last_emotion_time: dict[int, float]                = {}
    looking_ids:       set[int]                        = set()

    quit_app = False
    while not quit_app:
        try:
            with dai.Pipeline() as pipeline:
                queues = build_pipeline(pipeline, args)
                pipeline.start()
                print("[camera] pipeline started")
                last_log = 0.0

                while pipeline.isRunning() and not quit_app:
                    pipeline.processTasks()

                    # Tracklets drive every tick; all other queues are opportunistic.
                    track_msg = queues["tracklets"].tryGet()
                    if track_msg is None:
                        time.sleep(0.001)
                        continue

                    # --- Parse pose data ---
                    raw_poses: list[tuple[tuple, float, float]] = []
                    pose_msg = queues["poses"].tryGet()
                    if pose_msg is not None:
                        for bbox, item in _parse_gathered(pose_msg):
                            yaw, pitch, _ = extract_pose(item)
                            raw_poses.append((bbox, yaw, pitch))

                    # --- Parse age/gender data (new tracks only, applied after ID match) ---
                    raw_ag: list[tuple[tuple, object]] = []
                    if "age_gender" in queues:
                        ag_msg = queues["age_gender"].tryGet()
                        if ag_msg is not None:
                            raw_ag = _parse_gathered(ag_msg)

                    # --- Parse emotion data ---
                    raw_emo: list[tuple[tuple, object]] = []
                    if "emotion" in queues:
                        emo_msg = queues["emotion"].tryGet()
                        if emo_msg is not None:
                            raw_emo = _parse_gathered(emo_msg)

                    # --- Update per-track state ---
                    now = time.time()
                    active_ids: set[int] = set()

                    for t in track_msg.tracklets:
                        # VERIFY: dai.Tracklet.TrackingStatus enum values
                        if t.status in (dai.Tracklet.TrackingStatus.LOST,
                                        dai.Tracklet.TrackingStatus.REMOVED):
                            track_states.pop(t.id, None)
                            pose_cache.pop(t.id, None)
                            age_gender_cache.pop(t.id, None)
                            emotion_cache.pop(t.id, None)
                            last_emotion_time.pop(t.id, None)
                            looking_ids.discard(t.id)
                            continue

                        active_ids.add(t.id)
                        if t.id not in track_states:
                            track_states[t.id] = LookState()

                        tb = _bbox_for_tracklet(t)

                        # Refresh pose cache when a fresh match arrives.
                        best_i = _best_match(tb, [(i, p[0]) for i, p in enumerate(raw_poses)])
                        if best_i >= 0:
                            pose_cache[t.id] = (raw_poses[best_i][1], raw_poses[best_i][2])

                        # Debounce using cached pose.
                        yaw, pitch = pose_cache.get(t.id, POSE_UNSEEN)
                        if track_states[t.id].update(is_looking(yaw, pitch)):
                            looking_ids.add(t.id)
                        else:
                            looking_ids.discard(t.id)

                        # Age/gender: only cache once per track (no need to re-run).
                        if t.id not in age_gender_cache and raw_ag:
                            best_i = _best_match(tb, [(i, b) for i, (b, _) in enumerate(raw_ag)])
                            if best_i >= 0:
                                age_gender_cache[t.id] = extract_age_gender(raw_ag[best_i][1])

                        # Emotion: update at most every EMOTION_INTERVAL seconds per track.
                        if raw_emo and now - last_emotion_time.get(t.id, 0) >= EMOTION_INTERVAL:
                            best_i = _best_match(tb, [(i, b) for i, (b, _) in enumerate(raw_emo)])
                            if best_i >= 0:
                                emotion_cache[t.id]     = extract_emotion(raw_emo[best_i][1])
                                last_emotion_time[t.id] = now

                    # Prune state for tracks that silently disappeared.
                    for tid in list(track_states):
                        if tid not in active_ids:
                            del track_states[tid]
                            pose_cache.pop(tid, None)
                            age_gender_cache.pop(tid, None)
                            emotion_cache.pop(tid, None)
                            last_emotion_time.pop(tid, None)
                            looking_ids.discard(tid)

                    # Throttled console output.
                    if now - last_log >= 0.5:
                        last_log = now
                        extras = ""
                        if age_gender_cache:
                            extras += "  ag=" + str({k: f"{v[0][0]}{v[1]}"
                                                     for k, v in age_gender_cache.items()})
                        if emotion_cache:
                            extras += "  emo=" + str({k: v[0]
                                                      for k, v in emotion_cache.items()})
                        print(f"Looking: {len(looking_ids):>2}  "
                              f"tracked: {len(active_ids):>2}  "
                              f"ids: {sorted(looking_ids)}{extras}")

                    # Optional preview.
                    if args.preview:
                        frame_msg = queues["frame"].tryGet()
                        if frame_msg is not None:
                            draw_preview(frame_msg, track_msg.tracklets, pose_cache,
                                         age_gender_cache, emotion_cache, looking_ids)

                pipeline.stop()

        except KeyboardInterrupt:
            quit_app = True
        except Exception as exc:
            print(f"[main] camera error: {exc}")
            time.sleep(2.0)

    if args.preview:
        cv2.destroyAllWindows()
    print("\nPipeline stopped.")


if __name__ == "__main__":
    main()

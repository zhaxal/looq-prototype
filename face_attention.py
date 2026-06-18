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
import csv
import os
import time
from datetime import datetime
from pathlib import Path

# Load DEPTHAI_HUB_API_KEY from .env if present (needed for zoo model downloads).
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import cv2
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    FrameCropper,
    GatherData,
)

# --- Config -----------------------------------------------------------------
_HERE  = Path(__file__).parent
_MODELS = _HERE / "models"


def _model(slug: str, local_name: str):
    """Use local archive from models/ if present, else fall back to zoo slug."""
    p = _MODELS / local_name
    return dai.NNArchive(str(p)) if p.exists() else slug


POSE_MODEL        = "luxonis/head-pose-estimation:60x60"
POSE_INPUT        = (60, 60)

# Local model archives for age/gender and emotion (not in zoo; manual download).
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


def _decimate(pipeline, detections, frames, every_n):
    """On-device throttle: forward only every Nth (detections, frame) PAIR.

    Heavy stage-2 nets (age/gender, emotion) are cached/throttled on the host,
    but that doesn't stop the VPU from running them every frame. Gating what
    reaches their FrameCropper means the VPU only infers at the rate we actually
    consume — the real saving when all 4 nets share the chip.

    Critically, we gate the DETECTIONS and the IMAGE together in one Script so
    the cropper always receives a matched 1:1 pair — the same shape as the
    full-rate path that works. Gating detections alone (the old version) left
    the cropper pulling full-rate images against 1-in-N detections; its image
    queue filled and back-pressured the SHARED face_nn.passthrough, starving
    the tracker and freezing the host loop at 0 fps.

    Both Script inputs are NON-BLOCKING. Passthrough frames are available
    immediately, but detections lag by the NN's inference latency; with blocking
    inputs the Script (which reads dets first) stalls at startup while frames
    pile into its frame queue — once full, the SHARED face_nn.passthrough can no
    longer send, starving the tracker and producing zero faces. Non-blocking lets
    stale frames drop so passthrough (and the tracker) never block. every_n<=1 →
    no node, full rate. Returns (detections_out, frame_out); feed detections_out
    to BOTH the cropper and GatherData's reference so they stay aligned.
    """
    if every_n <= 1:
        return detections, frames
    script = pipeline.create(dai.node.Script)
    script.setScript(f"""
i = 0
while True:
    dets = node.inputs['dets'].get()
    frame = node.inputs['frame'].get()
    i = i + 1
    if i % {int(every_n)} == 0:
        node.io['dets_out'].send(dets)
        node.io['frame_out'].send(frame)
""")  # VERIFY node.inputs/node.io accessors on this depthai release
    detections.link(script.inputs["dets"])
    frames.link(script.inputs["frame"])
    # Non-blocking + small queue: drop, don't stall the shared producers.
    for _name in ("dets", "frame"):
        try:                               # VERIFY setBlocking/setMaxSize on Script inputs
            script.inputs[_name].setBlocking(False)
            script.inputs[_name].setMaxSize(2)
        except Exception:
            pass
    return script.outputs["dets_out"], script.outputs["frame_out"]


def _looking_gate(pipeline):
    """Host-fed detection gate for the busy-street case.

    The host already decides who is 'looking' (head-pose thresholds). Instead of
    cropping every face for the heavy nets, it sends ONLY the looking faces'
    detections into this Script, which forwards them to the heavy-net cropper.
    On a crowded street where most faces aren't looking, this is the big VPU
    saving: age/gender + emotion run on the few who looked, not the crowd.

    Uses the same host→Script input-queue pattern as the greenhouse `trigger`.
    Returns (output_for_cropper, host_input_queue).
    """
    script = pipeline.create(dai.node.Script)
    script.setScript("""
while True:
    d = node.inputs['dets'].get()
    node.io['out'].send(d)
""")  # VERIFY node.inputs/node.io accessors on this depthai release
    in_q = script.inputs["dets"].createInputQueue()   # VERIFY createInputQueue
    return script.outputs["out"], in_q


def _looking_detections(pose_msg, raw_poses):
    """Build an ImgDetections of only the looking faces, stamped with the
    source frame's timestamp so the device cropper matches it to the right
    buffered frame. raw_poses[i] aligns with reference detection i (same order
    out of GatherData)."""
    ref = pose_msg.reference_data                       # VERIFY accessor
    src = list(ref.detections)
    out = dai.ImgDetections()
    out.detections = [src[i] for i, (_, yaw, pitch) in enumerate(raw_poses)
                      if i < len(src) and is_looking(yaw, pitch)]
    try:                                                # VERIFY ts/seq round-trip
        out.setTimestamp(ref.getTimestamp())
        out.setSequenceNum(ref.getSequenceNum())
    except Exception:
        pass
    return out


def _unblock_inputs(node, max_size=4):
    """Make a node's inputs non-blocking so a decimated/slow branch can't apply
    backpressure up a SHARED producer (face_nn.passthrough) and stall the whole
    graph — the cause of the hang when --age-gender/--emotion are decimated.

    A blocking input queue, once full, stalls the producer; since passthrough
    fans out to the tracker AND every cropper, one full cropper-image queue
    freezes the tracker too. Non-blocking makes the slow branch drop frames
    instead. Accessors drift across depthai releases, so probe defensively.
    """
    inputs = []
    try:                                   # VERIFY getInputs() on this release
        inputs = list(node.getInputs())
    except Exception:
        pass
    if not inputs:                         # fallback: common named handles
        for name in ("inputImage", "inputFrame", "input", "inputImgDetections"):
            h = getattr(node, name, None)
            if h is not None:
                inputs.append(h)
    for inp in inputs:
        try:
            inp.setBlocking(False)
            inp.setMaxSize(max_size)
        except Exception:
            pass


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
    # Drop, don't stall: keeps a decimated branch from back-pressuring the
    # shared face_nn.passthrough and freezing the tracker. See _unblock_inputs.
    _unblock_inputs(cropper)
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
    face_model   = _model(f"luxonis/yunet:{args.face_res}",
                          f"yunet-{args.face_res}.rvc2.tar.xz")
    pose_model   = _model(POSE_MODEL, "head-pose-60x60.rvc2.tar.xz")

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
            pipeline, face_nn.out, face_nn.passthrough, POSE_INPUT, pose_model, args.fps
        ),
    }

    # When --looking-gate is set, heavy branches are fed by a host-controlled
    # gate (only looking faces); otherwise they fall back to on-device frame
    # decimation (the safe default for first device bring-up).
    gate_queues = {}   # name -> (host_input_queue, every_n)

    if args.age_gender:
        # Age/gender only needs one good read per new face → run ~once a second.
        # The host still caches the first result per track ID.
        ag_every = args.ag_every or max(1, round(args.fps))
        if args.looking_gate:
            ag_dets, ag_in = _looking_gate(pipeline)
            ag_img = face_nn.passthrough
            gate_queues["age_gender"] = (ag_in, ag_every)
            print(f"[opt] age/gender gated to LOOKING faces, every {ag_every} frame(s)")
        else:
            ag_dets, ag_img = _decimate(pipeline, face_nn.out, face_nn.passthrough, ag_every)
            print(f"[opt] age/gender NN gated to every {ag_every} frame(s) on device")
        queues["age_gender"] = _make_face_branch(
            pipeline, ag_dets, ag_img, AGE_GENDER_INPUT,
            dai.NNArchive(AGE_GENDER_ARCHIVE), args.fps
        )
    if args.emotion:
        # Emotion is the heaviest net (260x260) and host-throttled to
        # EMOTION_INTERVAL anyway → match that rate on the VPU.
        emo_every = args.emo_every or max(1, round(args.fps * EMOTION_INTERVAL))
        if args.looking_gate:
            emo_dets, emo_in = _looking_gate(pipeline)
            emo_img = face_nn.passthrough
            gate_queues["emotion"] = (emo_in, emo_every)
            print(f"[opt] emotion gated to LOOKING faces, every {emo_every} frame(s)")
        else:
            emo_dets, emo_img = _decimate(pipeline, face_nn.out, face_nn.passthrough, emo_every)
            print(f"[opt] emotion NN gated to every {emo_every} frame(s) on device")
        queues["emotion"] = _make_face_branch(
            pipeline, emo_dets, emo_img, EMOTION_INPUT,
            dai.NNArchive(EMOTION_ARCHIVE), args.fps, multi_head=False
        )

    queues["_gates"] = gate_queues
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


# --- Live terminal dashboard (--tui) ----------------------------------------

class LiveDisplay:
    """In-place ANSI dashboard for SSH testing. No extra dependencies."""

    _HIDE = "\033[?25l"   # hide cursor
    _SHOW = "\033[?25h"   # restore cursor
    _HOME = "\033[H"
    _CLEAR = "\033[2J"
    _BOLD = "\033[1m"
    _DIM  = "\033[2m"
    _RST  = "\033[0m"
    _GRN  = "\033[32m"
    _RED  = "\033[31m"
    _CYN  = "\033[36m"
    _YLW  = "\033[33m"
    _W    = 64

    def __init__(self, face_res, fps):
        self.face_res = face_res
        self.fps = fps
        print(self._HIDE, end="", flush=True)

    def close(self):
        print(self._SHOW, end="", flush=True)

    @staticmethod
    def _c(text, *codes):
        return "\033[" + ";".join(codes) + "m" + text + "\033[0m"

    def _rule(self, char="─"):
        return char * self._W

    def update(self, looking_ids, active_ids, tracklets,
               pose_cache, age_gender_cache, emotion_cache, fps_actual=None,
               dwell=None):
        fps_s = f"{fps_actual:.1f}fps" if fps_actual else f"{self.fps}fps"
        title = f" ATTENTION  {self.face_res} @ {fps_s}"
        lines = [
            self._CLEAR + self._HOME,
            self._c(title.ljust(self._W), "1", "7"),  # bold + reverse
            self._rule(),
        ]

        n_look  = len(looking_ids)
        n_total = len(active_ids)
        bar_on  = "█" * n_look
        bar_off = "░" * max(0, n_total - n_look)
        count_s = self._c(str(n_look), "1", "32")
        lines.append(f"  LOOKING  {count_s} / {n_total}   "
                     + self._c(bar_on, "32") + self._c(bar_off, "2"))
        lines.append(self._rule())

        # Header
        lines.append(self._c(
            f"  {'ID':>3}  {'LOOK':^4}  {'DWELL':>6}  {'YAW':>6}  {'PITCH':>6}  {'A/G':>6}  EMOTION",
            "2"))
        lines.append(self._rule("╌"))

        active_tracklets = [t for t in tracklets
                            if t.status not in (dai.Tracklet.TrackingStatus.LOST,
                                                dai.Tracklet.TrackingStatus.REMOVED)]
        if not active_tracklets:
            lines.append(self._c("  no faces detected", "2"))
        for t in active_tracklets:
            looking = t.id in looking_ids
            look_s  = self._c(" YES", "1", "32") if looking else self._c("  no", "31")

            secs     = (dwell or {}).get(t.id, 0.0)
            dwell_s  = f"{secs:5.1f}s"

            yp       = pose_cache.get(t.id)
            yaw_s    = f"{yp[0]:+5.0f}°" if yp else "     ?"
            pitch_s  = f"{yp[1]:+5.0f}°" if yp else "     ?"

            ag       = age_gender_cache.get(t.id)
            ag_s     = f"{ag[0][0]}{ag[1]:>3}" if ag else "     "

            em       = emotion_cache.get(t.id)
            emo_s    = f"{em[0]} {em[1]:.0%}" if em else ""

            row = f"  {t.id:>3}  {look_s}  {dwell_s}  {yaw_s}  {pitch_s}  {ag_s}  {emo_s}"
            lines.append(row)

        lines.append(self._rule())
        lines.append(self._c("  q / Ctrl-C to stop", "2"))
        print("\n".join(lines), end="", flush=True)


# --- Helpers ----------------------------------------------------------------

def _bbox_for_tracklet(t):
    return (t.roi.x, t.roi.y, t.roi.x + t.roi.width, t.roi.y + t.roi.height)


def _total_dwell(tid, now, look_accum, look_since):
    """Total seconds tid has been (committed) looking, including any live streak."""
    total = look_accum.get(tid, 0.0)
    if tid in look_since:
        total += now - look_since[tid]
    return total


def _best_match(tb, indexed_bboxes):
    """Return index of best IoU match, or -1 if none exceeds IOU_MATCH."""
    best_i, best_iou = -1, IOU_MATCH
    for i, bbox in indexed_bboxes:
        iou = boxes_overlap(tb, bbox)
        if iou > best_iou:
            best_i, best_iou = i, iou
    return best_i


def _parse_gathered(msg):
    """Return list of (det_bbox, MessageGroup_item) from a GatheredData message.

    Defensive against empty frames: when no faces are present GatherData can emit
    a message with no items or a missing/None reference_data. Reading
    `.reference_data.detections` blindly then raises AttributeError, which the
    main loop mistakes for a device disconnect and restarts every 2s — looking
    like "zero faces then crash". Guard all of it and just return [] instead.
    """
    ref = getattr(msg, "reference_data", None)
    dets = getattr(ref, "detections", None) if ref is not None else None
    items = getattr(msg, "items", None)
    if not dets or not items:
        return []
    out = []
    for i, item in enumerate(items):            # VERIFY .items accessor
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
    p.add_argument("--ag-every",   type=int, default=0,
                   help="run age/gender NN every Nth frame on device (0=auto ~1s; 1=full rate)")
    p.add_argument("--emo-every",  type=int, default=0,
                   help="run emotion NN every Nth frame on device (0=auto ~EMOTION_INTERVAL; 1=full rate)")
    p.add_argument("--log", nargs="?", const="", metavar="PATH",
                   help="write CSV session log; omit PATH for auto-named file")
    p.add_argument("--tui", action="store_true",
                   help="live in-place terminal dashboard (good for SSH testing)")
    p.add_argument("--looking-gate", action="store_true",
                   help="run age/gender + emotion only on faces confirmed looking "
                        "(scales to crowded scenes; needs --age-gender/--emotion)")
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
    look_accum:        dict[int, float]                = {}   # tid → cumulative looking seconds
    look_since:        dict[int, float]                = {}   # tid → start of current looking streak

    display = LiveDisplay(args.face_res, args.fps) if args.tui else None

    csv_file = csv_writer = None
    if args.log is not None:
        _CSV_FIELDS = ["ts", "track_id", "looking", "look_seconds",
                       "yaw", "pitch",
                       "age", "gender",
                       "emotion", "emotion_conf",
                       "looking_total", "tracked_total"]
        log_path = args.log or f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(log_path, "w", newline="", buffering=1)
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        csv_writer.writeheader()
        print(f"[log] writing to {log_path}")

    quit_app = False
    while not quit_app:
        started_ok = False
        try:
            with dai.Pipeline() as pipeline:
                queues = build_pipeline(pipeline, args)
                pipeline.start()
                started_ok = True
                print("[camera] pipeline started")
                gates = queues.get("_gates", {})
                last_log = 0.0
                frame_count = 0
                gate_tick   = 0
                fps_actual  = None

                while pipeline.isRunning() and not quit_app:
                    # Block on tracklets — the per-frame driver — exactly like the
                    # proven greenhouse app blocks on its preview queue. v3 has no
                    # pipeline.processTasks(); host nodes (parsers, GatherData) run
                    # in their own threads after pipeline.start(). All other queues
                    # are read opportunistically with tryGet() below.
                    track_msg = queues["tracklets"].get()
                    if track_msg is None:
                        continue

                    # --- Parse pose data ---
                    raw_poses: list[tuple[tuple, float, float]] = []
                    pose_msg = queues["poses"].tryGet()
                    if pose_msg is not None:
                        for bbox, item in _parse_gathered(pose_msg):
                            yaw, pitch, _ = extract_pose(item)
                            raw_poses.append((bbox, yaw, pitch))

                    # --- Feed the looking-gate (heavy nets only see looking faces) ---
                    if gates and pose_msg is not None:
                        look_dets = None
                        for _name, (in_q, every_n) in gates.items():
                            if gate_tick % every_n == 0:
                                if look_dets is None:
                                    look_dets = _looking_detections(pose_msg, raw_poses)
                                in_q.send(look_dets)
                        gate_tick += 1

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

                    # Build IoU-match indices once per frame, not per tracklet.
                    pose_idx = [(i, p[0]) for i, p in enumerate(raw_poses)]
                    ag_idx   = [(i, b) for i, (b, _) in enumerate(raw_ag)]
                    emo_idx  = [(i, b) for i, (b, _) in enumerate(raw_emo)]

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
                            look_accum.pop(t.id, None)
                            look_since.pop(t.id, None)
                            continue

                        active_ids.add(t.id)
                        if t.id not in track_states:
                            track_states[t.id] = LookState()

                        tb = _bbox_for_tracklet(t)

                        # Refresh pose cache when a fresh match arrives.
                        best_i = _best_match(tb, pose_idx)
                        if best_i >= 0:
                            pose_cache[t.id] = (raw_poses[best_i][1], raw_poses[best_i][2])

                        # Debounce using cached pose; accumulate dwell time on the
                        # committed state (a streak's seconds land in look_accum
                        # when it ends; the live streak is added on read).
                        yaw, pitch = pose_cache.get(t.id, POSE_UNSEEN)
                        if track_states[t.id].update(is_looking(yaw, pitch)):
                            looking_ids.add(t.id)
                            if t.id not in look_since:
                                look_since[t.id] = now
                        else:
                            looking_ids.discard(t.id)
                            if t.id in look_since:
                                look_accum[t.id] = (look_accum.get(t.id, 0.0)
                                                    + now - look_since.pop(t.id))

                        # Age/gender: only cache once per track (no need to re-run).
                        if t.id not in age_gender_cache and raw_ag:
                            best_i = _best_match(tb, ag_idx)
                            if best_i >= 0:
                                age_gender_cache[t.id] = extract_age_gender(raw_ag[best_i][1])

                        # Emotion: update at most every EMOTION_INTERVAL seconds per track.
                        if raw_emo and now - last_emotion_time.get(t.id, 0) >= EMOTION_INTERVAL:
                            best_i = _best_match(tb, emo_idx)
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
                            look_accum.pop(tid, None)
                            look_since.pop(tid, None)

                    frame_count += 1

                    # Throttled output + CSV logging.
                    if now - last_log >= 0.5:
                        elapsed     = now - last_log if last_log else 0.5
                        fps_actual  = frame_count / elapsed
                        frame_count = 0
                        last_log    = now

                        if display is not None:
                            dwell = {tid: _total_dwell(tid, now, look_accum, look_since)
                                     for tid in active_ids}
                            display.update(looking_ids, active_ids, track_msg.tracklets,
                                           pose_cache, age_gender_cache, emotion_cache,
                                           fps_actual, dwell)
                        else:
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

                        if csv_writer is not None:
                            ts = datetime.now().isoformat(timespec="milliseconds")
                            for t in track_msg.tracklets:
                                if t.status in (dai.Tracklet.TrackingStatus.LOST,
                                                dai.Tracklet.TrackingStatus.REMOVED):
                                    continue
                                _yaw, _pitch = pose_cache.get(t.id, (None, None))
                                _gender, _age = age_gender_cache.get(t.id, (None, None))
                                _emo, _emo_conf = emotion_cache.get(t.id, (None, None))
                                csv_writer.writerow({
                                    "ts":            ts,
                                    "track_id":      t.id,
                                    "looking":       int(t.id in looking_ids),
                                    "look_seconds":  f"{_total_dwell(t.id, now, look_accum, look_since):.1f}",
                                    "yaw":           f"{_yaw:.1f}"      if _yaw      is not None else "",
                                    "pitch":         f"{_pitch:.1f}"    if _pitch    is not None else "",
                                    "age":           _age               if _age      is not None else "",
                                    "gender":        _gender            if _gender   is not None else "",
                                    "emotion":       _emo               if _emo      is not None else "",
                                    "emotion_conf":  f"{_emo_conf:.2f}" if _emo_conf is not None else "",
                                    "looking_total": len(looking_ids),
                                    "tracked_total": len(active_ids),
                                })

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
            # Show the real error. A failure BEFORE pipeline.start() is a
            # build/bring-up bug (wrong API name, bad model archive, drifted
            # `# VERIFY` accessor) — retrying it just hides the traceback in a
            # 2s loop, so make it fatal. A failure AFTER start() is treated as a
            # device disconnect and retried, like the proven greenhouse app.
            import traceback
            print(f"[main] camera error: {exc}")
            traceback.print_exc()
            if not started_ok:
                print("[main] error occurred during pipeline build/start — "
                      "not a transient device issue; exiting so the traceback "
                      "above is visible.")
                break
            time.sleep(2.0)

    if display:
        display.close()
    if csv_file:
        csv_file.close()
    if args.preview:
        cv2.destroyAllWindows()
    print("\nPipeline stopped.")


if __name__ == "__main__":
    main()

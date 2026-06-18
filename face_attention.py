#!/usr/bin/env python3
"""Steps 1-3 of the attention-counter prototype.

Pipeline (all NN inference runs on the OAK VPU; host only parses + matches):

    Camera (RGB)
      |-- ParsingNeuralNetwork[YuNet]  -> face ImgDetections
      |        |-> ObjectTracker        -> tracklets (stable IDs)
      |        |-> FrameCropper         -> one 60x60 crop per face
      |                  |-> ParsingNeuralNetwork[head-pose] -> yaw/pitch/roll
      |                          |-> GatherData -> poses synced to face dets
      |
      `-- (full frame, optional, for --preview)

On the host we read two queues per frame -- tracklets (IDs) and gathered poses
(bbox + yaw/pitch/roll) -- and match them by bbox overlap so each stable track
ID gets a "looking?" flag. Counting/dedupe is step 4; age/gender/emotion step 5.

Dev on a laptop with `--preview`; deploy on the Pi headless (default).

NOTE: This is a first on-device draft. The lines marked `# VERIFY` use API
details that can drift between depthai / depthai-nodes releases -- run it once on
the OAK and adjust if a name/shape differs. None of it can be validated off-device.
"""
import argparse
import time

import depthai as dai

# Host-side helper + parser nodes from the depthai-nodes package.
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    FrameCropper,
    GatherData,
)

# --- Config -----------------------------------------------------------------
FACE_MODEL = "luxonis/yunet:640x480"          # stage 1: face detection
POSE_MODEL = "luxonis/head-pose-estimation:60x60"  # stage 2: head pose
POSE_INPUT = (60, 60)                          # crop size the pose model wants

FPS = 12                                       # keep modest for the Pi 4 host
FACE_CONF = 0.6                                # face detection confidence

# "Looking at camera" thresholds, in degrees. Tune these live in step 3.
YAW_LIMIT = 20.0
PITCH_LIMIT = 15.0


def parse_args():
    p = argparse.ArgumentParser(description="OAK attention counter (steps 1-3)")
    p.add_argument("--preview", action="store_true",
                   help="reserved for step 4 (annotated window); no-op for now")
    p.add_argument("--fps", type=float, default=FPS)
    return p.parse_args()


def boxes_overlap(a, b):
    """IoU of two (xmin, ymin, xmax, ymax) boxes in normalized coords."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def is_looking(yaw, pitch):
    return abs(yaw) < YAW_LIMIT and abs(pitch) < PITCH_LIMIT


def build_pipeline(args):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    # One output stream we reuse for detection input and (optionally) preview.
    cam_out = cam.requestOutput((1280, 720), dai.ImgFrame.Type.NV12, fps=args.fps)

    # Stage 1: face detection. ParsingNeuralNetwork auto-attaches YuNetParser.
    face_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cam_out, nnSource=FACE_MODEL  # VERIFY arg names (input / nnSource)
    )
    # Filter weak detections if the parser exposes it; otherwise filter on host.
    try:
        face_nn.setConfidenceThreshold(FACE_CONF)  # VERIFY available on parser
    except AttributeError:
        pass

    # On-device tracker -> stable IDs (this is what makes counting possible).
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)      # VERIFY enum path
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    face_nn.out.link(tracker.inputDetections)
    face_nn.passthrough.link(tracker.inputDetectionFrame)
    face_nn.passthrough.link(tracker.inputTrackerFrame)

    # Stage 2: crop each face, run head pose on the crop.
    cropper = pipeline.create(FrameCropper).fromImgDetections(
        inputImgDetections=face_nn.out,
        outputSize=POSE_INPUT,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    ).build(inputImage=cam_out)

    pose_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cropper.out, nnSource=POSE_MODEL
    )

    # Sync each pose result back to the face detection that produced its crop.
    gathered = pipeline.create(GatherData).build(
        inputData=pose_nn.out,
        inputReference=face_nn.out,
        cameraFps=args.fps,
    )

    queues = {
        "tracklets": tracker.out.createOutputQueue(),
        "poses": gathered.out.createOutputQueue(),
    }
    return pipeline, queues


def extract_pose(pose_msg):
    """Pull (yaw, pitch, roll) out of the head-pose parser message.

    The Luxonis head-pose model emits three regression values. The exact
    attribute name from the parser needs a one-time check on-device -- print
    the message and adjust here.  # VERIFY ordering + accessor
    """
    # Common shapes seen across depthai-nodes versions; try them in order.
    for attr in ("angles", "prediction", "predictions"):
        vals = getattr(pose_msg, attr, None)
        if vals is not None:
            v = list(vals)
            if len(v) >= 3:
                return float(v[0]), float(v[1]), float(v[2])
    raise RuntimeError(f"Could not read angles from {type(pose_msg)}; "
                       f"dir={[a for a in dir(pose_msg) if not a.startswith('_')]}")


def main():
    args = parse_args()
    pipeline, queues = build_pipeline(args)

    print(f"Models: {FACE_MODEL} + {POSE_MODEL}")
    print(f"Looking thresholds: |yaw|<{YAW_LIMIT} |pitch|<{PITCH_LIMIT} deg")
    print("Starting pipeline... (Ctrl-C to stop)\n")

    pipeline.start()
    last_log = 0.0
    while pipeline.isRunning():
        pipeline.processTasks()

        track_msg = queues["tracklets"].tryGet()
        pose_msg = queues["poses"].tryGet()
        if track_msg is None or pose_msg is None:
            time.sleep(0.001)
            continue

        # Each gathered item carries a face detection bbox + its pose result.
        poses = []
        for item in pose_msg.gathered:  # VERIFY: name of the gathered list
            det = item.reference         # the face ImgDetection
            bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
            yaw, pitch, roll = extract_pose(item.data)
            poses.append((bbox, yaw, pitch))

        # Match each stable track to the closest pose by bbox overlap.
        now = time.time()
        if now - last_log >= 0.5:  # throttle console output
            last_log = now
            for t in track_msg.tracklets:
                tb = (t.roi.x, t.roi.y, t.roi.x + t.roi.width,
                      t.roi.y + t.roi.height)  # VERIFY roi accessor
                best = max(poses, key=lambda p: boxes_overlap(tb, p[0]),
                           default=None)
                if best and boxes_overlap(tb, best[0]) > 0.2:
                    _, yaw, pitch = best
                    flag = "LOOKING" if is_looking(yaw, pitch) else "       "
                    print(f"id={t.id:>3}  yaw={yaw:+6.1f}  pitch={pitch:+6.1f}  {flag}")

    print("Pipeline stopped.")


if __name__ == "__main__":
    main()

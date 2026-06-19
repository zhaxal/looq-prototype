"""DepthAI pipeline construction."""
from __future__ import annotations

from string import Template

import depthai as dai
from depthai_nodes.node import FrameCropper, GatherData, ParsingNeuralNetwork

from . import config
from .processing import is_looking


# --- Model resolution --------------------------------------------------------

def resolve_model(slug: str, local_name: str) -> str | dai.NNArchive:
    """Use a local archive from models/ when present; fall back to the zoo slug."""
    p = config.MODELS_DIR / local_name
    return dai.NNArchive(str(p)) if p.exists() else slug


# --- Pipeline helpers --------------------------------------------------------

def patch_frame_cropper() -> None:
    """OAK script engine doesn't accept keyword args for Size2f or
    addCropRotatedRect — patch the template to use positional args."""
    tmpl = FrameCropper.IMG_DETECTIONS_SCRIPT_CONTENT.template
    tmpl = tmpl.replace(
        "Size2f(width=rot.size.width + 2*p, height=rot.size.height + 2*p, normalized=True)",
        "Size2f(rot.size.width + 2*p, rot.size.height + 2*p)",
    )
    tmpl = tmpl.replace(
        "cfg.addCropRotatedRect(rect=pad_rotated_rect(rot_rect, PADDING), normalizedCoords=True)",
        "cfg.addCropRotatedRect(pad_rotated_rect(rot_rect, PADDING), True)",
    )
    tmpl = "\n".join(ln for ln in tmpl.split("\n") if "setTimestampDevice" not in ln)
    FrameCropper.IMG_DETECTIONS_SCRIPT_CONTENT = Template(tmpl)


def unblock_inputs(node, max_size: int = 4) -> None:
    """Set all inputs on node to non-blocking.

    Prevents a slow/decimated branch from back-pressuring the shared
    face_nn.passthrough and stalling the tracker.
    """
    inputs = []
    try:
        inputs = list(node.getInputs())              # VERIFY getInputs() on this release
    except Exception:
        pass
    if not inputs:
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


def decimate(pipeline, detections, frames, every_n: int):
    """Forward only every Nth (detections, frame) pair to throttle heavy branches.

    Gates both streams together in one Script so the downstream cropper always
    receives matched pairs. Inputs are non-blocking to prevent back-pressure on
    the shared face_nn.passthrough. every_n <= 1 returns the originals unchanged.
    Returns (detections_out, frame_out).
    """
    if every_n <= 1:
        return detections, frames
    script = pipeline.create(dai.node.Script)
    script.setScript(f"""
i = 0
while True:
    dets  = node.inputs['dets'].get()
    frame = node.inputs['frame'].get()
    i = i + 1
    if i % {int(every_n)} == 0:
        node.io['dets_out'].send(dets)
        node.io['frame_out'].send(frame)
""")  # VERIFY node.inputs / node.io accessors on this depthai release
    detections.link(script.inputs["dets"])
    frames.link(script.inputs["frame"])
    for name in ("dets", "frame"):
        try:                                         # VERIFY setBlocking / setMaxSize on Script inputs
            script.inputs[name].setBlocking(False)
            script.inputs[name].setMaxSize(2)
        except Exception:
            pass
    return script.outputs["dets_out"], script.outputs["frame_out"]


def looking_gate(pipeline):
    """Host-controlled detection gate.

    The host sends only the ImgDetections for confirmed-looking faces, so the
    heavy nets run on those faces only — the main saving for crowded scenes.
    Returns (output_for_cropper, host_input_queue).
    """
    script = pipeline.create(dai.node.Script)
    script.setScript("""
while True:
    d = node.inputs['dets'].get()
    node.io['out'].send(d)
""")  # VERIFY node.inputs / node.io accessors on this depthai release
    in_q = script.inputs["dets"].createInputQueue()  # VERIFY createInputQueue
    return script.outputs["out"], in_q


def looking_detections(pose_msg, raw_poses: list) -> dai.ImgDetections:
    """Build an ImgDetections containing only the faces whose pose passes the looking test.

    Stamped with the source frame's timestamp so the device cropper can match it
    to the right buffered frame. raw_poses[i] aligns with detection i (GatherData order).
    """
    ref = pose_msg.reference_data                    # VERIFY accessor
    src = list(ref.detections)
    out = dai.ImgDetections()
    out.detections = [
        src[i] for i, (_, yaw, pitch) in enumerate(raw_poses)
        if i < len(src) and is_looking(yaw, pitch)
    ]
    try:                                             # VERIFY timestamp / sequence round-trip
        out.setTimestamp(ref.getTimestamp())
        out.setSequenceNum(ref.getSequenceNum())
    except Exception:
        pass
    return out


def make_face_branch(pipeline, face_detections, face_image,
                     input_size: tuple, nn_source, fps: float,
                     multi_head: bool = True):
    """FrameCropper → ParsingNeuralNetwork → GatherData for a face-crop model.

    face_detections: ImgDetections — drives the cropper and GatherData reference.
    face_image:      ImgFrame     — pixel source for the cropper.
    multi_head:      True for models with ≥2 heads (nn.outputs → MessageGroup);
                     False for single-head models (nn.out → parser message).
    Returns a non-blocking output queue for the gathered results.
    """
    cropper = pipeline.create(FrameCropper).fromImgDetections(
        inputImgDetections=face_detections,
        outputSize=input_size,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    ).build(inputImage=face_image)
    unblock_inputs(cropper)

    nn      = pipeline.create(ParsingNeuralNetwork).build(
        input=cropper.out, nnSource=nn_source        # VERIFY arg names
    )
    nn_data = nn.outputs if multi_head else nn.out

    gathered = pipeline.create(GatherData).build(
        inputData=nn_data,
        inputReference=face_detections,
        cameraFps=fps,
    )
    return gathered.out.createOutputQueue(maxSize=2, blocking=False)  # VERIFY kwargs


def make_input_source(pipeline, args, cam_w: int, cam_h: int):
    """Return a BGR888p ImgFrame output at (cam_w, cam_h): live camera or video replay."""
    if not getattr(args, "test_video", None):
        cam = pipeline.create(dai.node.Camera).build()
        return cam.requestOutput((cam_w, cam_h), dai.ImgFrame.Type.BGR888p, fps=args.fps)

    # Video replay: swap the camera for a file source; all VPU inference is unchanged.
    replay = pipeline.create(dai.node.ReplayVideo)    # VERIFY node name in v3
    replay.setReplayVideoFile(str(args.test_video))    # VERIFY method name
    replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)  # VERIFY method name
    replay.setLoop(True)                               # VERIFY loop support

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(cam_w, cam_h)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(cam_w * cam_h * 3)
    replay.out.link(manip.inputImage)                  # VERIFY output name
    return manip.out


# --- Top-level builder -------------------------------------------------------

def build(pipeline, args) -> dict:
    """Assemble the full pipeline and return a dict of output queues.

    Keys: "tracklets", "poses", optionally "age_gender", "emotion", "frame".
    "_gates" holds host-side gate input queues when --looking-gate is active.
    """
    patch_frame_cropper()

    cam_w, cam_h = (int(x) for x in args.face_res.split("x"))
    face_model   = resolve_model(
        f"{config.FACE_MODEL_SLUG}:{args.face_res}",
        f"yunet-{args.face_res}.rvc2.tar.xz",
    )
    pose_model   = resolve_model(config.POSE_MODEL_SLUG, "head-pose-60x60.rvc2.tar.xz")

    cam_out = make_input_source(pipeline, args, cam_w, cam_h)

    face_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cam_out, nnSource=face_model            # VERIFY arg names
    )
    try:
        face_nn.setConfidenceThreshold(config.FACE_CONFIDENCE)  # VERIFY available on parser
    except AttributeError:
        pass

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)          # VERIFY enum
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    face_nn.out.link(tracker.inputDetections)
    face_nn.passthrough.link(tracker.inputDetectionFrame)
    face_nn.passthrough.link(tracker.inputTrackerFrame)

    queues: dict = {
        "tracklets": tracker.out.createOutputQueue(maxSize=2, blocking=False),
        "poses": make_face_branch(
            pipeline, face_nn.out, face_nn.passthrough,
            config.POSE_INPUT, pose_model, args.fps,
        ),
    }

    gate_queues: dict = {}

    if args.age_gender:
        ag_every = args.ag_every or max(1, round(args.fps))
        if args.looking_gate:
            ag_dets, ag_in = looking_gate(pipeline)
            gate_queues["age_gender"] = (ag_in, ag_every)
            ag_img = face_nn.passthrough
            print(f"[opt] age/gender gated to LOOKING faces, every {ag_every} frame(s)")
        else:
            ag_dets, ag_img = decimate(pipeline, face_nn.out, face_nn.passthrough, ag_every)
            print(f"[opt] age/gender NN gated to every {ag_every} frame(s) on device")
        queues["age_gender"] = make_face_branch(
            pipeline, ag_dets, ag_img, config.AGE_GENDER_INPUT,
            dai.NNArchive(str(config.AGE_GENDER_ARCHIVE)), args.fps,
        )

    if args.emotion:
        emo_every = args.emo_every or max(1, round(args.fps * config.EMOTION_INTERVAL))
        if args.looking_gate:
            emo_dets, emo_in = looking_gate(pipeline)
            gate_queues["emotion"] = (emo_in, emo_every)
            emo_img = face_nn.passthrough
            print(f"[opt] emotion gated to LOOKING faces, every {emo_every} frame(s)")
        else:
            emo_dets, emo_img = decimate(pipeline, face_nn.out, face_nn.passthrough, emo_every)
            print(f"[opt] emotion NN gated to every {emo_every} frame(s) on device")
        queues["emotion"] = make_face_branch(
            pipeline, emo_dets, emo_img, config.EMOTION_INPUT,
            dai.NNArchive(str(config.EMOTION_ARCHIVE)), args.fps,
            multi_head=False,
        )

    queues["_gates"] = gate_queues
    if args.preview:
        queues["frame"] = face_nn.passthrough.createOutputQueue(maxSize=2, blocking=False)

    return queues

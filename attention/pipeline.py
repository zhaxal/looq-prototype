"""DepthAI v3 pipeline construction — single head-pose branch.

    Camera (BGR888p)
      └─ ParsingNeuralNetwork[YuNet]  → face ImgDetections
           ├─ ObjectTracker            → tracklets (stable IDs)
           └─ FrameCropper[60x60] → ParsingNeuralNetwork[head-pose] → GatherData → poses
"""
from __future__ import annotations

from string import Template

import depthai as dai
from depthai_nodes.node import FrameCropper, GatherData, ParsingNeuralNetwork

from . import config


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

    Prevents a slow branch from back-pressuring the shared face_nn.passthrough
    and stalling the tracker.
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


def make_face_branch(pipeline, face_detections, face_image,
                     input_size: tuple, nn_source, fps: float):
    """FrameCropper → ParsingNeuralNetwork → GatherData for a face-crop model.

    face_detections: ImgDetections — drives the cropper and GatherData reference.
    face_image:      ImgFrame      — pixel source for the cropper.
    Returns a non-blocking output queue for the gathered results.
    """
    cropper = pipeline.create(FrameCropper).fromImgDetections(
        inputImgDetections=face_detections,
        outputSize=input_size,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    ).build(inputImage=face_image)
    unblock_inputs(cropper)

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cropper.out, nnSource=nn_source        # VERIFY arg names
    )

    gathered = pipeline.create(GatherData).build(
        inputData=nn.outputs,                        # head-pose is multi-head -> MessageGroup
        inputReference=face_detections,
        cameraFps=fps,
    )
    return gathered.out.createOutputQueue(maxSize=2, blocking=False)  # VERIFY kwargs


# --- Top-level builder -------------------------------------------------------

def build(pipeline, settings: config.Settings) -> dict:
    """Assemble the pipeline and return a dict of output queues.

    Keys: "tracklets", "poses", "frame".
    """
    patch_frame_cropper()

    cam_w, cam_h = (int(x) for x in settings.face_res.split("x"))
    face_model   = resolve_model(
        f"{config.FACE_MODEL_SLUG}:{settings.face_res}",
        f"yunet-{settings.face_res}.rvc2.tar.xz",
    )
    pose_model   = resolve_model(config.POSE_MODEL_SLUG, "head-pose-60x60.rvc2.tar.xz")

    cam     = pipeline.create(dai.node.Camera).build()
    if settings.flip_180:
        # Camera is mounted upside down; rotate on the sensor so every downstream
        # node (face NN, tracker, head-pose, preview) gets an upright image.
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)  # VERIFY enum
    cam_out = cam.requestOutput((cam_w, cam_h), dai.ImgFrame.Type.BGR888p, fps=settings.fps)

    face_nn = pipeline.create(ParsingNeuralNetwork).build(
        input=cam_out, nnSource=face_model            # VERIFY arg names
    )
    face_nn.getParser(0).setConfidenceThreshold(config.FACE_CONFIDENCE)

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)          # VERIFY enum
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    face_nn.out.link(tracker.inputDetections)
    face_nn.passthrough.link(tracker.inputDetectionFrame)
    face_nn.passthrough.link(tracker.inputTrackerFrame)

    return {
        "tracklets": tracker.out.createOutputQueue(maxSize=2, blocking=False),
        "poses": make_face_branch(
            pipeline, face_nn.out, face_nn.passthrough,
            config.POSE_INPUT, pose_model, settings.fps,
        ),
        "frame": face_nn.passthrough.createOutputQueue(maxSize=2, blocking=False),
    }

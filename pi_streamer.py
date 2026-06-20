#!/usr/bin/env python3
"""OAK-D-Lite H.264 streamer — Pi side of the server-vision split.

The Pi runs NO neural nets. The OAK hardware-encodes the RGB camera to H.264 and
this script ships the encoded frames to the GPU server, which runs the entire
vision pipeline (detect → track → pose/age/gender/emotion) and the attention
calculations.

    python pi_streamer.py [--server ws://host:8000/ingest] [--fps 12]
                          [--res 640x480] [--test-video VIDEO]

Server endpoint defaults to LOOQ_SERVER_URL (see .env / attention/config.py).
"""
import argparse
import time
from pathlib import Path

from attention.config import (
    load_dotenv,
    DEFAULT_FPS, FACE_RESOLUTIONS, LOOQ_SERVER_URL,
    ENCODER_BITRATE_KBPS, ENCODER_KEYFRAME_FREQ,
)
from attention.netclient import FrameSender

load_dotenv()

import depthai as dai  # noqa: E402 — after load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAK-D-Lite H.264 streamer")
    p.add_argument("--server",     default=LOOQ_SERVER_URL,
                   help="server WebSocket ingest URL")
    p.add_argument("--fps",        type=float, default=DEFAULT_FPS)
    p.add_argument("--res",        default="640x480", choices=FACE_RESOLUTIONS,
                   help="capture/encode resolution")
    p.add_argument("--test-video", type=Path, metavar="VIDEO",
                   help="replay a video file instead of the live camera")
    return p.parse_args()


def _source(pipeline, args, w: int, h: int):
    """NV12 ImgFrame output (encoder-ready): live camera or video replay."""
    if not args.test_video:
        cam = pipeline.create(dai.node.Camera).build()
        return cam.requestOutput((w, h), dai.ImgFrame.Type.NV12, fps=args.fps)

    replay = pipeline.create(dai.node.ReplayVideo)        # VERIFY v3 node name
    replay.setReplayVideoFile(str(args.test_video))       # VERIFY method
    replay.setOutFrameType(dai.ImgFrame.Type.NV12)        # VERIFY method
    replay.setLoop(True)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(w, h)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    manip.setMaxOutputFrameSize(w * h * 3)
    replay.out.link(manip.inputImage)
    return manip.out


def _capture_ts(pkt) -> float:
    """Monotonic per-frame timestamp (seconds) for the server's debounce/dwell."""
    try:
        return pkt.getTimestampDevice().total_seconds()   # VERIFY accessor
    except Exception:
        try:
            return pkt.getTimestamp().total_seconds()
        except Exception:
            return time.time()


def run(args: argparse.Namespace) -> None:
    w, h = (int(x) for x in args.res.split("x"))
    sender = FrameSender(args.server).start()

    print(f"Streaming {args.res} @ {args.fps}fps  →  {args.server}")
    print(f"Source: {'video:' + str(args.test_video) if args.test_video else 'camera'}"
          f"  (Ctrl-C to stop)\n")

    quit_app = False
    while not quit_app:
        started_ok = False
        try:
            with dai.Pipeline() as pipeline:
                src = _source(pipeline, args, w, h)

                enc = pipeline.create(dai.node.VideoEncoder)
                enc.setDefaultProfilePreset(                 # VERIFY v3 API
                    args.fps, dai.VideoEncoderProperties.Profile.H264_MAIN
                )
                enc.setBitrateKbps(ENCODER_BITRATE_KBPS)
                enc.setKeyframeFrequency(ENCODER_KEYFRAME_FREQ)
                src.link(enc.input)
                bitstream = enc.bitstream.createOutputQueue(maxSize=4, blocking=False)

                pipeline.start()
                started_ok = True
                print("[camera] pipeline started")

                seq        = 0
                last_stats = time.time()
                while pipeline.isRunning() and not quit_app:
                    pkt = bitstream.get()
                    if pkt is None:
                        continue
                    sender.send(_capture_ts(pkt), seq, bytes(pkt.getData()))
                    seq += 1

                    now = time.time()
                    if now - last_stats >= 2.0:
                        sent, dropped, queued = sender.stats()
                        link = "up" if sender.connected else "DOWN"
                        print(f"[stream] {link}  sent={sent} dropped={dropped} "
                              f"queued={queued} seq={seq}")
                        last_stats = now

                pipeline.stop()
        except KeyboardInterrupt:
            quit_app = True
        except Exception as exc:
            import traceback
            print(f"[stream] error: {exc}")
            traceback.print_exc()
            if not started_ok:
                print("[stream] error before pipeline start — exiting")
                break
            time.sleep(2.0)

    sender.close()
    print("[stream] stopped")


if __name__ == "__main__":
    run(parse_args())

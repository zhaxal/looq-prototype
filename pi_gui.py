#!/usr/bin/env python3
"""Fullscreen Pi GUI for the H.264 streamer.

    python pi_gui.py [--server ws://host:8000/ingest] [--fps 12] [--res 640x480]
                     [--test-video VIDEO]

LOOQ_SERVER_URL (and other env vars) can be set in .env — loaded automatically.
"""
import argparse
from pathlib import Path

from attention.config import load_dotenv, LOOQ_SERVER_URL, DEFAULT_FPS, FACE_RESOLUTIONS

load_dotenv()   # must run before depthai is imported inside StreamEngine

from attention.streamer_gui import launch   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Looq Pi streamer GUI")
    p.add_argument("--server",     default=LOOQ_SERVER_URL,
                   help="server WebSocket ingest URL")
    p.add_argument("--fps",        type=float, default=DEFAULT_FPS)
    p.add_argument("--res",        default="640x480", choices=FACE_RESOLUTIONS)
    p.add_argument("--test-video", type=Path, metavar="VIDEO",
                   help="replay a video file instead of the live camera")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch(server_url=args.server, fps=args.fps, res=args.res,
           test_video=args.test_video)

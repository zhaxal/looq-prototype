#!/usr/bin/env python3
"""Stream a local video to the server as H.264 — test the server without a Pi.

    python scripts/replay_to_server.py path/to/clip.mp4 [--server ws://host:8000/ingest]

Demuxes/transcodes the file to H.264 with PyAV and sends each encoded frame using
the same wire format as the Pi streamer (struct "<dI" header + bytes), at roughly
the source frame rate.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import av  # PyAV
import websocket  # websocket-client

sys.path.insert(0, str(Path(__file__).parent.parent))
from attention.config import load_dotenv, LOOQ_SERVER_URL

load_dotenv()

_HEADER = struct.Struct("<dI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--server", default=LOOQ_SERVER_URL)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    ws = websocket.create_connection(args.server, timeout=5.0)
    print(f"connected to {args.server}; streaming {args.video}")

    inp = av.open(str(args.video))
    out = av.open("/dev/null", mode="w", format="h264")
    enc = out.add_stream("libx264", rate=int(args.fps))
    enc.pix_fmt = "yuv420p"

    seq = 0
    period = 1.0 / args.fps
    for frame in inp.decode(video=0):
        for pkt in enc.encode(frame):
            ws.send_binary(_HEADER.pack(time.time(), seq) + bytes(pkt))
            seq += 1
        time.sleep(period)
    for pkt in enc.encode(None):  # flush
        ws.send_binary(_HEADER.pack(time.time(), seq) + bytes(pkt))
        seq += 1

    ws.close()
    print(f"done; sent {seq} packets")


if __name__ == "__main__":
    main()

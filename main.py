#!/usr/bin/env python3
"""Headless / SSH entry point — same Engine as the GUI, no display required.

Useful for dev over SSH and for calibrating without the touchscreen:

    python main.py                       # live one-line stats, Ctrl-C to stop
    python main.py --calibrate 5         # measure & save the ad yaw/pitch offset
    python main.py --log --face-res 320x240
"""
import argparse
import time

from attention import config

config.load_dotenv()           # must run before depthai is imported

from attention.engine import Engine   # noqa: E402 — import after load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ad-attention counter (headless)")
    p.add_argument("--face-res",     choices=config.FACE_RESOLUTIONS)
    p.add_argument("--fps",          type=float)
    p.add_argument("--yaw-offset",   type=float)
    p.add_argument("--pitch-offset", type=float)
    p.add_argument("--yaw-tol",      type=float)
    p.add_argument("--pitch-tol",    type=float)
    p.add_argument("--log",          action="store_true",
                   help="write an attention_*.csv session log")
    p.add_argument("--calibrate",    nargs="?", type=float, const=5.0, metavar="SECS",
                   help="measure & save the ad offset over SECS seconds, then exit")
    return p.parse_args()


def build_settings(args: argparse.Namespace) -> config.Settings:
    s = config.Settings.load()
    for field in ("face_res", "fps", "yaw_offset", "pitch_offset", "yaw_tol", "pitch_tol"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(s, field, val)
    if args.log:
        s.log = True
    return s


def run_calibrate(engine: Engine, secs: float) -> None:
    print(f"Calibrating for {secs:.0f}s — stand where viewers stand and look at the ad.")
    engine.start()
    time.sleep(1.0)                     # let the pipeline warm up
    engine.calibrate(secs)
    deadline = time.time() + secs + 8.0
    while time.time() < deadline:
        snap = engine.snapshot()
        if snap.error:
            print(snap.message)
            break
        if snap.message.startswith("Calibrated"):
            print(snap.message)
            break
        time.sleep(0.2)
    else:
        print("Calibration timed out — no stable face detected.")
    engine.stop()


def run_live(engine: Engine) -> None:
    engine.start()
    print("Running — Ctrl-C to stop.")
    try:
        while True:
            s = engine.snapshot()
            if s.error:
                print(s.message)
                break
            print(f"\rLooking: {s.looking_now:>2}  tracked: {s.tracked_now:>2}  "
                  f"total: {s.total_unique:>3}  avg-dwell: {s.avg_dwell:4.1f}s  "
                  f"{s.fps:4.1f}fps   ", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
    finally:
        engine.stop()


def main() -> None:
    args     = parse_args()
    settings = build_settings(args)
    engine   = Engine(settings)
    print(f"face-res: {settings.face_res}  fps: {settings.fps}  "
          f"yaw-offset: {settings.yaw_offset:+.0f}  "
          f"cone: |yaw|<{settings.yaw_tol} |pitch|<{settings.pitch_tol}")
    if args.calibrate is not None:
        run_calibrate(engine, args.calibrate)
    else:
        run_live(engine)


if __name__ == "__main__":
    main()

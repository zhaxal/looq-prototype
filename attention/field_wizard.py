"""Operator 'magic wand' — short field commands that run the right long command.

    python main.py field doctor
    python main.py field preview
    python main.py field calibrate
    python main.py field controlled
    python main.py field middle
    python main.py field high
    python main.py field bundle

Each command prints the EXACT underlying command (so operators learn it / can
copy-paste), writes results into a timestamped runs/ folder, and then executes it.
Extra flags after the subcommand are forwarded to main.py, e.g.:

    python main.py field controlled --counting-roi 0.15,0.20,0.85,0.95 \
        --manual-total 10 --manual-lookers 5

Add `--print` to print the command without running it.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import config

ROOT = config.PROJECT_ROOT

# --- Default field constants (the metro billboard) ---------------------------
CAL_FILE      = "configs/metro_billboard_calibration.json"
CAMERA_ID     = "oak_d_lite_01"
CAMERA_HEIGHT = "1.6"
BILLBOARD_ID  = "metro_billboard_001"
BILLBOARD_W   = "1.0"
BILLBOARD_H   = "2.0"     # billboard is 1.0m x 2.0m (NOT 3.0 — that's the top edge height)

PHASES = ("doctor", "preview", "calibrate", "controlled", "middle", "high", "bundle")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], print_only: bool, capture_to: Path | None = None) -> int:
    """Print the command, then execute it (unless print_only)."""
    printable = "python " + " ".join(cmd[1:])     # cmd[0] is the python executable
    print("\n>>> " + printable + "\n")
    if print_only:
        print("(--print: not executed)")
        return 0
    if capture_to is not None:
        # Doctor: capture plain output to a file AND show it.
        proc = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8",
                              errors="replace", capture_output=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        sys.stdout.write(out)
        capture_to.parent.mkdir(parents=True, exist_ok=True)
        capture_to.write_text(out, encoding="utf-8")
        (capture_to.parent / "command.txt").write_text(printable + "\n", encoding="utf-8")
        return proc.returncode
    # Interactive phases (TUI / preview / calibrate): inherit the terminal.
    return subprocess.run(cmd, cwd=ROOT).returncode


def _py(*main_args: str) -> list[str]:
    return [sys.executable, "main.py", *main_args]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    phase, extra = argv[0], list(argv[1:])
    print_only = "--print" in extra
    extra = [a for a in extra if a != "--print"]

    if phase not in PHASES:
        print(f"Unknown field command: {phase}")
        _print_help()
        return 2

    ts = _ts()
    print(f"=== LOOQ FIELD WIZARD — phase: {phase} ===")

    if phase == "doctor":
        out = ROOT / "runs" / f"{ts}_doctor" / "stdout.txt"
        print("Checking the rig (software / OAK-D Lite / privacy / output)…")
        rc = _run(_py("--doctor", "--calibration-file", CAL_FILE) + extra,
                  print_only, capture_to=out)
        if not print_only:
            print(f"\nDoctor output saved → {out}")
        return rc

    if phase == "preview":
        print("Framing check: faces visible, people 2–5 m away, camera beside the "
              "billboard aimed at faces. Frames are LOCAL only (never stored).")
        return _run(_py("--privacy-safe", "--allow-uncalibrated-camera-facing",
                        "--debug-local-preview", "--preview",
                        "--calibration-file", CAL_FILE, "--test-phase", "live") + extra,
                    print_only)

    if phase == "calibrate":
        print("Calibration: exactly ONE person looks at the billboard centre for 5s.")
        return _run(_py("--privacy-safe", "--calibrate", "center", "--seconds", "5",
                        "--camera-id", CAMERA_ID, "--camera-height-m", CAMERA_HEIGHT,
                        "--billboard-id", BILLBOARD_ID,
                        "--billboard-width-m", BILLBOARD_W,
                        "--billboard-height-m", BILLBOARD_H,
                        "--calibration-file", CAL_FILE, "--tui") + extra,
                    print_only)

    if phase in ("controlled", "middle", "high"):
        test_phase = {"controlled": "controlled",
                      "middle": "middle_traffic",
                      "high": "high_traffic"}[phase]
        out = f"runs/{ts}_{phase}/summary.json"
        base = _py("--privacy-safe", "--calibration-file", CAL_FILE, "--tui",
                   "--test-phase", test_phase)
        if not any(a == "--summary-out" for a in extra):
            base += ["--summary-out", out]
        if phase == "controlled":
            print("Controlled test: 10 passes — 5 look, 5 don't. Each person fully "
                  "leaves the frame before the next. Enter manual_total / manual_lookers.")
        elif phase == "middle":
            print("Middle traffic: 3–5 min moderate flow. Watch FPS, no crashes.")
        else:
            print("High traffic: 3–5 min dense flow. STRESS TEST ONLY — not accuracy.")
        return _run(base + extra, print_only)

    if phase == "bundle":
        print("Collecting a privacy-safe debug bundle (text/JSON only, no images)…")
        return _run([sys.executable, "scripts/collect_debug_bundle.py",
                     "--calibration-file", CAL_FILE] + extra, print_only)

    return 2


def _print_help() -> None:
    print(__doc__)
    print("Phases:", ", ".join(PHASES))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

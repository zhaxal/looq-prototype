#!/usr/bin/env python3
"""Collect a PRIVACY-SAFE debug bundle the field team can send back.

Creates runs/debug_bundle_<timestamp>.zip containing ONLY anonymous text/JSON/CSV:
doctor output, software versions, git commit, summaries, events, calibration files,
and any error logs. It NEVER includes images, video, screenshots, or face crops —
only files with a small whitelist of text suffixes are ever added.

Usage:
    python scripts/collect_debug_bundle.py
    python scripts/collect_debug_bundle.py --calibration-file configs/metro_billboard_calibration.json
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 so emoji/glyphs in captured output don't crash a Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
except Exception:
    pass

# Only these suffixes are ever bundled. This is the privacy guarantee: no image
# or video suffix is on the list, so frames/crops can never be included.
SAFE_SUFFIXES = {".json", ".csv", ".txt", ".log", ".md"}
# Belt-and-suspenders: explicitly refuse anything that looks visual.
BLOCKED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff",
                    ".mp4", ".avi", ".mov", ".mkv", ".h264", ".h265", ".yuv",
                    ".npy", ".npz", ".webp", ".heic"}


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:
        return f"<failed to run {' '.join(cmd)}: {e}>\n"


def _version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"<not importable: {e}>"


def collect() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    bundle = runs / f"debug_bundle_{ts}.zip"

    # --- Generated diagnostics (in-memory, added as text entries) ------------
    doctor_txt = _run([sys.executable, "main.py", "--doctor"])

    versions = [
        f"timestamp_local : {datetime.now().isoformat(timespec='seconds')}",
        f"python          : {platform.python_version()}",
        f"platform        : {platform.platform()}",
        f"machine         : {platform.machine()}",
        f"depthai         : {_version('depthai')}",
        f"depthai_nodes   : {_version('depthai_nodes')}",
        f"numpy           : {_version('numpy')}",
        f"opencv (cv2)    : {_version('cv2')}",
    ]
    git_commit = _run(["git", "rev-parse", "HEAD"]).strip() or "<no git>"
    git_status = _run(["git", "status", "--short", "--branch"])
    command_used = "python " + " ".join(sys.argv)

    generated = {
        "doctor.txt":       doctor_txt,
        "versions.txt":     "\n".join(versions) + "\n",
        "git_commit.txt":   git_commit + "\n",
        "git_status.txt":   git_status,
        "command_used.txt": command_used + "\n",
        "README.txt": (
            "LOOQ debug bundle (privacy-safe).\n"
            "Contains only anonymous text/JSON/CSV: doctor output, versions, git\n"
            "commit, session summaries, events, and calibration files.\n"
            "NO images, video, screenshots, or face crops are included.\n"),
    }

    # --- Files on disk (filtered) --------------------------------------------
    candidates: list[Path] = []
    candidates += sorted((ROOT / "runs").rglob("summary.json"))
    candidates += sorted((ROOT / "runs").rglob("events.csv"))
    candidates += sorted((ROOT / "configs").glob("*.json")) if (ROOT / "configs").exists() else []
    candidates += sorted(ROOT.glob("attention_*.csv"))      # legacy logs, if any
    if args_calib := _arg_calibration():
        p = Path(args_calib)
        if p.exists():
            candidates.append(p)

    added: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in generated.items():
            zf.writestr(f"debug_bundle_{ts}/{name}", text)
            added.append(name)
        seen: set[Path] = set()
        for f in candidates:
            f = f.resolve()
            if f in seen or not f.exists() or f == bundle.resolve():
                continue
            seen.add(f)
            if f.suffix.lower() in BLOCKED_SUFFIXES or f.suffix.lower() not in SAFE_SUFFIXES:
                skipped.append(str(f))
                continue
            arc = f.relative_to(ROOT) if ROOT in f.parents else Path(f.name)
            zf.writestr(f"debug_bundle_{ts}/{arc.as_posix()}", f.read_text(errors="replace"))
            added.append(arc.as_posix())

    print(f"✅ wrote {bundle}")
    print(f"   included {len(added)} files:")
    for a in added:
        print(f"     - {a}")
    if skipped:
        print(f"   skipped (non-text / not privacy-safe): {len(skipped)}")
    print("\nSend this single .zip back to the CV team.")
    return bundle


def _arg_calibration() -> str | None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--calibration-file", dest="calibration_file", default=None)
    known, _ = ap.parse_known_args()
    return known.calibration_file


if __name__ == "__main__":
    collect()

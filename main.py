#!/usr/bin/env python3
"""LOOQ — privacy-safe billboard attention counter (field / headless CLI).

This is the entry point the field team uses on the Raspberry Pi tomorrow. It runs
the SAME OAK-D Lite + DepthAI v3 pipeline as the touch GUI (app.py), but with a
text UI, calibration workflow, and an anonymous summary.json — no display, no
stored frames, no uploads.

Quick reference
---------------
    python main.py --doctor                      # check the rig (safe w/o camera)
    python main.py --simulate-poses ...          # verify counters w/o hardware
    python main.py --privacy-safe --calibrate center --seconds 5 \
        --calibration-file configs/metro_billboard_calibration.json --tui
    python main.py --privacy-safe \
        --calibration-file configs/metro_billboard_calibration.json \
        --tui --summary-out runs/metro_test/summary.json

The numbers to report: total_passed, looked_total, looked_0_3s, looked_0_5s, looked_1_0s.
Do NOT report age/gender/emotion/identity — the system does not produce them.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 so ✅/❌ render on a Windows dev box too (the Pi is already UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
except Exception:
    pass

from attention import config

config.load_dotenv()                              # must run before depthai is imported

# Pure modules — safe to import without depthai installed (doctor/simulate need this).
from attention import privacy, report             # noqa: E402
from attention import processing as att_processing  # noqa: E402  (no depthai dep)
from attention.calibration import (                # noqa: E402
    Calibration, CalibrationError, build_from_samples,
)


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOOQ privacy-safe billboard attention counter",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Commands (mutually exclusive in spirit; checked in main()).
    p.add_argument("--doctor", action="store_true",
                   help="run hardware/software/privacy checks and exit (safe without camera)")
    p.add_argument("--simulate-poses", dest="simulate_poses", action="store_true",
                   help="run the offline counter simulation (no hardware) and exit")
    p.add_argument("--calibrate", nargs="?", const="center", default=None, metavar="TARGET",
                   help="measure the billboard-looking head pose, write a calibration "
                        "file, then exit (TARGET defaults to 'center')")

    # Mode / privacy.
    p.add_argument("--privacy-safe", dest="privacy_safe", action="store_true",
                   help="strict mode: reject unsafe flags (recommended for the field)")
    p.add_argument("--tui", action="store_true", help="full-screen text UI for operators")

    # Calibration / session metadata.
    p.add_argument("--seconds", type=float, default=5.0,
                   help="calibration capture duration (default 5)")
    p.add_argument("--calibration-file", dest="calibration_file", metavar="PATH",
                   help="calibration profile to write (calibrate) or load (live run)")
    p.add_argument("--summary-out", dest="summary_out", metavar="PATH",
                   help="where to write summary.json (default runs/<session>/summary.json)")
    p.add_argument("--camera-id", dest="camera_id", default="oak_d_lite_01")
    p.add_argument("--camera-height-m", dest="camera_height_m", type=float, default=0.0)
    p.add_argument("--billboard-id", dest="billboard_id", default="billboard_001")
    p.add_argument("--billboard-width-m", dest="billboard_width_m", type=float, default=0.0)
    p.add_argument("--billboard-height-m", dest="billboard_height_m", type=float, default=0.0)
    p.add_argument("--allow-uncalibrated-camera-facing",
                   dest="allow_uncalibrated", action="store_true",
                   help="run without calibration using a CAMERA-FACING signal (not "
                        "billboard attention) — clearly flagged in the summary")
    p.add_argument("--std-tolerances", dest="std_tolerances", action="store_true",
                   help="derive cone tolerances from sample std instead of fixed defaults")
    p.add_argument("--no-manual-prompt", dest="no_manual_prompt", action="store_true",
                   help="skip the end-of-session manual validation questions")

    # Field phase + counting ROI + manual ground truth (V1.2).
    p.add_argument("--test-phase", dest="test_phase", default="live",
                   choices=list(report.FIELD_PHASES),
                   help="which field phase this run is (drives the pass/fail decision)")
    p.add_argument("--counting-roi", dest="counting_roi", metavar="x1,y1,x2,y2",
                   help="only count tracks whose center is inside this normalized "
                        "(0..1) box, e.g. 0.15,0.20,0.85,0.95 (background filter)")
    p.add_argument("--manual-total", dest="manual_total", type=int,
                   help="operator's manual count of visible face/head passes")
    p.add_argument("--manual-lookers", dest="manual_lookers", type=int,
                   help="operator's manual count of people who looked at the billboard")
    p.add_argument("--operator-notes", dest="operator_notes", default=None,
                   help="free-text operator notes saved into the summary")

    # Local debug preview (never stored, never uploaded).
    p.add_argument("--debug-local-preview", dest="debug_local_preview", action="store_true",
                   help="acknowledge LOCAL-ONLY preview; required to enable --preview")
    p.add_argument("--preview", action="store_true",
                   help="show a local preview window (needs --debug-local-preview)")

    # Advanced pipeline overrides (optional).
    p.add_argument("--face-res", choices=config.FACE_RESOLUTIONS)
    p.add_argument("--fps", type=float)
    p.add_argument("--yaw-tol", dest="yaw_tol", type=float)
    p.add_argument("--pitch-tol", dest="pitch_tol", type=float)
    p.add_argument("--flip-180", dest="flip_180", action=argparse.BooleanOptionalAction,
                   default=None, help="rotate 180° for an upside-down mount")
    p.add_argument("--log", action="store_true", help="also write the legacy attention_*.csv")

    # Forbidden-in-privacy-safe flags — recognised only so we can REJECT them loudly.
    for flag in ("--age-gender", "--emotion", "--demographics", "--save-crops",
                 "--save-frames", "--record", "--cloud-upload", "--upload"):
        p.add_argument(flag, dest=flag.lstrip("-").replace("-", "_"),
                       action="store_true", help=argparse.SUPPRESS)

    return p.parse_args()


def _import_engine():
    """Import the DepthAI-backed Engine, with an operator-friendly message on failure."""
    try:
        from attention.engine import Engine
        return Engine
    except ImportError as e:
        print(f"ERROR: cannot load the camera engine — {e}")
        print("  This needs depthai on the Raspberry Pi. Fix:")
        print("    python -m pip install -r requirements.txt")
        print("  Then check the rig with:  python main.py --doctor")
        raise SystemExit(1)


def build_settings(args: argparse.Namespace) -> config.Settings:
    """Load settings.json and apply optional pipeline overrides (not persisted)."""
    s = config.Settings.load()
    for fld in ("face_res", "fps", "yaw_tol", "pitch_tol", "flip_180"):
        val = getattr(args, fld, None)
        if val is not None:
            setattr(s, fld, val)
    if getattr(args, "log", False):
        s.log = True
    return s


# --- Calibration command -----------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> int:
    if not args.calibration_file:
        print("ERROR: --calibration-file is required for calibration (where to save it).")
        return 2

    Engine = _import_engine()               # deferred import (needs depthai)

    settings = build_settings(args)
    engine = Engine(settings)

    print(f"Calibration target='{args.calibrate}'  duration={args.seconds:.0f}s")
    print("Make sure ONLY ONE person is visible and looking at the billboard.\n")
    engine.start()

    if not _wait_running(engine, timeout=15.0):
        print("ERROR: camera did not start — run `python main.py --doctor` to diagnose.")
        engine.stop()
        return 1

    # Collect (yaw, pitch) from a single subject, one sample per processed frame.
    samples: list[tuple[float, float]] = []
    sample_times: list[float] = []
    frames_total = frames_zero = frames_multi = 0
    last_seq = -1
    deadline = time.time() + args.seconds
    print(f"Capturing… look at the billboard for {args.seconds:.0f}s")
    try:
        while time.time() < deadline:
            s = engine.snapshot()
            if s.frame_seq != last_seq and s.running:
                last_seq = s.frame_seq
                frames_total += 1
                valid = [t for t in s.tracks if t.yaw is not None]
                if len(s.tracks) == 0:
                    frames_zero += 1
                elif len(s.tracks) > 1:
                    frames_multi += 1
                elif valid:
                    samples.append((valid[0].yaw, valid[0].pitch))
                    sample_times.append(time.time())
            remaining = max(0.0, deadline - time.time())
            print(f"\r  samples={len(samples):3d}  "
                  f"faces_now={len(s.tracks)}  {remaining:4.1f}s left   ",
                  end="", flush=True)
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        engine.stop()

    # --- Reliability gates ---------------------------------------------------
    if frames_total == 0:
        print("ERROR: no frames received from the camera. Check the USB connection.")
        return 1
    if not samples and frames_zero:
        print("ERROR: zero faces visible during calibration.")
        print("  Fix: move closer, improve lighting, ensure the face is clearly visible.")
        return 1
    if frames_multi >= max(3, int(0.25 * frames_total)):
        print(f"ERROR: more than one face visible ({frames_multi}/{frames_total} frames).")
        print("  Fix: clear the scene — only the ONE calibration subject should be visible.")
        return 1

    valid_duration = (sample_times[-1] - sample_times[0]) if len(sample_times) > 1 else 0.0
    try:
        calib = build_from_samples(
            samples, valid_duration,
            target=args.calibrate or "center",
            camera_id=args.camera_id, camera_height_m=args.camera_height_m,
            billboard_id=args.billboard_id,
            billboard_width_m=args.billboard_width_m,
            billboard_height_m=args.billboard_height_m,
            std_tolerances=args.std_tolerances,
        )
    except CalibrationError as e:
        print(f"ERROR: calibration failed — {e}")
        return 1

    path = calib.save(args.calibration_file)

    # Also drop a timestamped copy in a run folder so calibration can't be lost.
    run_dir = config.PROJECT_ROOT / "runs" / f"{_timestamp()}_calibration"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_copy = calib.save(run_dir / "calibration.json")
    _write_run_meta(run_dir, "calibration",
                    f"Calibration capture ({calib.sample_count} samples, "
                    f"{calib.valid_duration_sec:.1f}s). Main copy: {path}")

    print(f"\n✅ Calibration saved:")
    print(f"   - {path}")
    print(f"   - {run_copy}")
    print(f"   yaw  {calib.yaw_mean_deg:+.1f}° ± {calib.yaw_tolerance_deg:.0f}°")
    print(f"   pitch {calib.pitch_mean_deg:+.1f}° ± {calib.pitch_tolerance_deg:.0f}°")
    print(f"   {calib.sample_count} samples over {calib.valid_duration_sec:.1f}s")
    print("\nNext: run the field session with the same --calibration-file.")
    return 0


# --- Run-folder persistence helpers ------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_run_meta(run_dir: Path, phase: str, description: str) -> None:
    """Write command.txt + README.txt into a run folder (never overwrites results)."""
    (run_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")
    (run_dir / "README.txt").write_text(
        f"LOOQ run folder\n"
        f"phase       : {phase}\n"
        f"created     : {datetime.now().isoformat(timespec='seconds')}\n"
        f"description : {description}\n\n"
        f"Files here are privacy-safe (anonymous counters/JSON/CSV only).\n"
        f"If you need the terminal log, copy-paste it into stdout.txt in this folder.\n")


# --- Live field command ------------------------------------------------------

def cmd_live(args: argparse.Namespace) -> int:
    # Calibration is required by default (Phase 5).
    calib: Calibration | None = None
    calibration_status = "loaded"
    if args.calibration_file and Path(args.calibration_file).exists():
        try:
            calib = Calibration.load(args.calibration_file)
        except CalibrationError as e:
            print(f"ERROR: {e}")
            return 2
    elif args.allow_uncalibrated:
        calibration_status = "missing_camera_relative_fallback"
        print("WARNING: using camera-facing signal, not billboard attention.")
    else:
        print("ERROR: No calibration file provided. Run calibration first.")
        print("  python main.py --privacy-safe --calibrate center --seconds 5 \\")
        print("    --calibration-file configs/metro_billboard_calibration.json --tui")
        return 2

    # Counting ROI (background filter). Optional but strongly recommended in metro.
    try:
        roi = att_processing.parse_roi(args.counting_roi)
    except ValueError as e:
        print(f"ERROR: bad --counting-roi: {e}")
        return 2
    if roi is None:
        print("WARNING: no counting ROI provided. Background faces may be counted.")

    settings = build_settings(args)
    if calib is not None:
        calib.apply_to_settings(settings)

    # Resolve output paths.
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.summary_out:
        summary_path = Path(args.summary_out)
        session_dir = summary_path.parent
    else:
        session_dir = config.PROJECT_ROOT / "runs" / session_id
        summary_path = session_dir / "summary.json"
    events_path = session_dir / "events.csv"

    Engine = _import_engine()               # deferred import (needs depthai)
    engine = Engine(settings)
    engine.events_csv_path = str(events_path)
    engine.counting_roi = roi

    # Preview banner (Phase 3).
    preview_on = bool(args.preview and args.debug_local_preview)
    if args.debug_local_preview:
        print("LOCAL DEBUG ONLY: preview frames are displayed locally and are not "
              "stored or uploaded.")

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_mono = time.time()
    engine.start()

    yaw_tol, pitch_tol = report.calibration_tolerances(
        calib, settings.yaw_tol, settings.pitch_tol)

    print(f"Phase: {args.test_phase}  ROI: {_roi_label(roi)}")
    print("Running — press Ctrl+C to finish and write the summary.\n")
    try:
        _run_tui_loop(engine, settings, calib, calibration_status,
                      tui=args.tui, preview=preview_on,
                      phase=args.test_phase, roi=roi)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFinishing…")
        engine.stop()
        if preview_on:
            _close_preview()

    ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    duration = time.time() - started_mono
    snap = engine.snapshot()
    counts = {
        "total_passed": snap.total_passed, "looked_total": snap.looked_total,
        "looked_0_3s": snap.looked_0_3, "looked_0_5s": snap.looked_0_5,
        "looked_1_0s": snap.looked_1_0,
    }

    manual = _ask_manual_validation(args)
    decision = report.compute_field_decision(
        args.test_phase, counts,
        manual.get("manual_total_visible_faces"),
        manual.get("manual_likely_lookers"))

    summary = report.build_summary(
        session_id=session_id, started_at=started_at, ended_at=ended_at,
        duration_sec=duration, counts=counts,
        yaw_tolerance_deg=yaw_tol, pitch_tolerance_deg=pitch_tol,
        calibration_status=calibration_status,
        calibration_file=args.calibration_file, mode="privacy_safe",
        test_phase=args.test_phase, counting_roi=roi, field_decision=decision,
        manual_validation=manual,
    )
    path = report.write_summary(summary, summary_path)

    # Persist run-folder artifacts so results can't be lost (Task 3).
    _write_run_meta(session_dir, args.test_phase,
                    f"{args.test_phase} session, {counts['total_passed']} passed, "
                    f"{counts['looked_0_5s']} looked>=0.5s")
    if args.calibration_file and Path(args.calibration_file).exists():
        try:
            (session_dir / "calibration.json").write_text(
                Path(args.calibration_file).read_text())
        except OSError:
            pass

    print("\n================ SESSION SUMMARY ================")
    print(f"  phase        : {args.test_phase}")
    print(f"  total_passed : {counts['total_passed']}")
    print(f"  looked_total : {counts['looked_total']}")
    print(f"  looked_0_3s  : {counts['looked_0_3s']}")
    print(f"  looked_0_5s  : {counts['looked_0_5s']}")
    print(f"  looked_1_0s  : {counts['looked_1_0s']}")
    print(f"  decision     : {decision['status'].upper()} — {decision['reason']}")
    print(f"  next step    : {decision['next_step']}")
    print("=================================================")
    print(f"  summary → {path}")
    if events_path.exists():
        print(f"  events  → {events_path}")
    print(f"  folder  → {session_dir}")
    return 0


# --- TUI ---------------------------------------------------------------------

def _wait_running(engine, timeout: float) -> bool:
    deadline = time.time() + timeout
    last_msg = ""
    while time.time() < deadline:
        s = engine.snapshot()
        if s.running:
            return True
        if s.message and s.message != last_msg:
            last_msg = s.message
            print(f"  {s.message}")
        time.sleep(0.2)
    return False


def _roi_label(roi) -> str:
    if roi is None:
        return "full frame (no ROI — background faces may count)"
    return ",".join(f"{v:.2f}" for v in roi)


def _render_tui(snap, settings, calib, calibration_status, phase="live", roi=None) -> str:
    calib_line = "loaded" if calibration_status == "loaded" else (
        "MISSING (camera-facing fallback)" if "fallback" in calibration_status else "missing")
    head = f"{settings.yaw_offset:+.0f} / {settings.pitch_offset:+.0f}"
    status = ("calibrated billboard direction" if calibration_status == "loaded"
              else "CAMERA-FACING fallback (not billboard attention)")
    warn = ""
    if calibration_status != "loaded":
        warn = ("\n  ##############################################\n"
                "  #  WARNING: NO CALIBRATION — camera-facing!  #\n"
                "  ##############################################")
    return (
        "LOOQ Privacy-Safe Billboard Attention Counter\n"
        "\n"
        f"  Phase: {phase}\n"
        "  Mode: privacy_safe\n"
        "  Camera: OAK-D Lite        Host: Raspberry Pi 4\n"
        f"  Calibration: {calib_line}\n"
        f"  ROI: {_roi_label(roi)}\n"
        "  Raw upload: OFF   Frames stored: OFF   Face crops stored: OFF\n"
        "  Age/gender: OFF   Emotion: OFF   Demographics: OFF\n"
        f"{warn}\n"
        "\n"
        f"  Active tracks : {snap.tracked_now}\n"
        f"  Total passed  : {snap.total_passed}\n"
        f"  Looked total  : {snap.looked_total}\n"
        f"  Looked >=0.3s : {snap.looked_0_3}\n"
        f"  Looked >=0.5s : {snap.looked_0_5}\n"
        f"  Looked >=1.0s : {snap.looked_1_0}\n"
        f"  FPS           : {snap.fps:.1f}\n"
        "\n"
        f"  Yaw/Pitch center    : {head}\n"
        f"  Yaw/Pitch tolerance : {settings.yaw_tol:.0f} / {settings.pitch_tol:.0f}\n"
        f"  Status: {status}\n"
        f"  {('● ' + snap.message) if snap.message else ''}\n"
        "  Ctrl+C: finish and write summary"
    )


def _run_tui_loop(engine, settings, calib, calibration_status, tui: bool, preview: bool,
                  phase="live", roi=None):
    is_tty = sys.stdout.isatty()
    while True:
        snap = engine.snapshot()
        if tui:
            block = _render_tui(snap, settings, calib, calibration_status, phase, roi)
            if is_tty:
                sys.stdout.write("\033[2J\033[H" + block + "\n")
            else:
                sys.stdout.write(block + "\n\n")
            sys.stdout.flush()
        else:
            print(f"\r[{phase}] passed:{snap.total_passed:3d} look>=0.3:{snap.looked_0_3:3d} "
                  f">=0.5:{snap.looked_0_5:3d} >=1.0:{snap.looked_1_0:3d} "
                  f"active:{snap.tracked_now:2d} {snap.fps:4.1f}fps  ", end="", flush=True)
        if preview and snap.frame is not None:
            preview = _show_preview(snap.frame, roi)   # disables itself on failure
        time.sleep(0.5)


# Local-only preview helpers (never write frames to disk).
_preview_failed = False


def _show_preview(frame, roi=None) -> bool:
    global _preview_failed
    if _preview_failed:
        return False
    try:
        import cv2
        shown = frame
        if roi is not None:
            shown = frame.copy()                    # don't mutate the shared frame
            h, w = shown.shape[:2]
            x1, y1, x2, y2 = roi
            cv2.rectangle(shown, (int(x1 * w), int(y1 * h)),
                          (int(x2 * w), int(y2 * h)), (0, 200, 255), 2)
            cv2.putText(shown, "COUNTING ROI", (int(x1 * w) + 4, int(y1 * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow("LOOQ LOCAL DEBUG (not stored)", shown)
        cv2.waitKey(1)
        return True
    except Exception as e:
        _preview_failed = True
        print(f"\n[preview] disabled — local preview needs full opencv (have headless?): {e}")
        return False


def _close_preview() -> None:
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass


def _ask_manual_validation(args) -> dict:
    manual = report.blank_manual_validation(getattr(args, "test_phase", "controlled"))
    # Values passed on the command line win and are never re-prompted.
    if getattr(args, "manual_total", None) is not None:
        manual["manual_total_visible_faces"] = args.manual_total
    if getattr(args, "manual_lookers", None) is not None:
        manual["manual_likely_lookers"] = args.manual_lookers
    if getattr(args, "operator_notes", None):
        manual["operator_notes"] = args.operator_notes

    need = (manual["manual_total_visible_faces"] is None
            or manual["manual_likely_lookers"] is None
            or not manual["operator_notes"])
    if args.no_manual_prompt or not need or not sys.stdin.isatty():
        return manual

    print("\nManual validation (press Enter to skip each):")
    try:
        if manual["manual_total_visible_faces"] is None:
            v = input("  Manual visible face/head passes? ").strip()
            if v.isdigit():
                manual["manual_total_visible_faces"] = int(v)
        if manual["manual_likely_lookers"] is None:
            v = input("  Manual likely lookers? ").strip()
            if v.isdigit():
                manual["manual_likely_lookers"] = int(v)
        if not manual["operator_notes"]:
            v = input("  Operator notes? ").strip()
            if v:
                manual["operator_notes"] = v
    except (EOFError, KeyboardInterrupt):
        print()
    return manual


# --- Dispatch ----------------------------------------------------------------

def main() -> None:
    # Magic-wand field wizard: `python main.py field <doctor|preview|calibrate|...>`.
    # Handled before argparse so the subcommand grammar doesn't clash with flags.
    if len(sys.argv) >= 2 and sys.argv[1] == "field":
        from attention import field_wizard
        sys.exit(field_wizard.main(sys.argv[2:]))

    args = parse_args()

    # Diagnostics first — these run without a camera (and without depthai).
    if args.doctor:
        from attention import doctor
        sys.exit(doctor.run(args))

    if args.simulate_poses:
        if args.privacy_safe:
            privacy.assert_privacy_safe(args)
        from attention import simulate
        sys.exit(simulate.run(args))

    # Live / calibration paths.
    if args.privacy_safe:
        privacy.assert_privacy_safe(args)

    if args.calibrate is not None:
        sys.exit(cmd_calibrate(args))

    sys.exit(cmd_live(args))


if __name__ == "__main__":
    main()

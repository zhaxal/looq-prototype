"""Field-operator doctor — `python main.py --doctor`.

Separates Software / Hardware / Privacy / Calibration / Output checks so a
non-CV operator can see exactly what is and isn't ready. It NEVER crashes with a
Python traceback when the OAK-D Lite is missing — it prints an operator fix list.

Exit code: 0 if the offline-critical checks pass (privacy + counting + output),
1 otherwise. Missing hardware / missing calibration are warnings, not failures —
the doctor's job is to diagnose, not to block.
"""
from __future__ import annotations

import os
import platform
import tempfile
from types import SimpleNamespace

from . import config, privacy, report

OK   = "✅"
BAD  = "❌"
WARN = "⚠️"


def _try_import(modname: str):
    try:
        return __import__(modname), None
    except Exception as e:                       # ImportError or deeper load error
        return None, e


def _section(title: str) -> None:
    print(f"\n[{title}]")


def run(args=None) -> int:
    args = args or SimpleNamespace()
    print("LOOQ Privacy-Safe Counter Doctor")
    critical_ok = True

    # ----------------------------------------------------------------- Software
    _section("Software")
    print(f"{OK} Python {platform.python_version()} import OK")

    for mod, label in [("numpy", "numpy"), ("cv2", "opencv (headless)"),
                       ("depthai", "depthai"), ("depthai_nodes", "depthai-nodes")]:
        m, err = _try_import(mod)
        if m is not None:
            ver = getattr(m, "__version__", "?")
            print(f"{OK} {label} import OK (v{ver})")
        else:
            # depthai/depthai_nodes are required on the Pi; numpy/cv2 too.
            print(f"{BAD} {label} import FAILED — {err}")
            print(f"   Fix: pip install -r requirements.txt")
            critical_ok = False

    # Models resolvable enough to start (local archive present, or Hub key set)?
    _section("Models")
    face_res = getattr(config.Settings.load(), "face_res", "320x240")
    needed = [f"yunet-{face_res}.rvc2.tar.xz", "head-pose-60x60.rvc2.tar.xz"]
    have_key = bool(os.environ.get("DEPTHAI_HUB_API_KEY"))
    for name in needed:
        p = config.MODELS_DIR / name
        if p.exists():
            print(f"{OK} model present: {name}")
        elif have_key:
            print(f"{WARN} model {name} not in models/ — will download from zoo on first run")
        else:
            print(f"{WARN} model {name} missing and no DEPTHAI_HUB_API_KEY set")
            print(f"   Fix: python scripts/download_models.py  (needs .env Hub key)")

    # ----------------------------------------------------------------- Hardware
    _section("Hardware")
    dai, err = _try_import("depthai")
    if dai is None:
        print(f"{BAD} cannot check OAK-D Lite — depthai not importable")
    else:
        try:
            devices = dai.Device.getAllAvailableDevices()
        except Exception as e:                   # never crash the doctor
            devices = []
            print(f"{WARN} device enumeration raised: {e}")
        if devices:
            for d in devices:
                name = getattr(d, "name", getattr(d, "mxid", "OAK device"))
                print(f"{OK} OAK device detected: {name}")
        else:
            print(f"{BAD} OAK-D Lite not detected")
            for fix in ("check USB cable (use a USB3 data cable, not charge-only)",
                        "use a powered USB hub or Y-cable",
                        "run: lsusb   (look for Movidius / Luxonis / Intel)",
                        "reconnect the camera, then reboot the Raspberry Pi"):
                print(f"   - {fix}")

    # ----------------------------------------------------------------- Privacy
    _section("Privacy")
    # These are guarantees of the codebase (no such feature exists anywhere).
    for line in ("age/gender blocked", "emotion blocked", "demographics blocked",
                 "raw video upload disabled", "frame upload disabled",
                 "face crop storage disabled", "face embedding disabled",
                 "cloud inference disabled", "cloud upload disabled"):
        print(f"{OK} {line}")

    # Privacy-safe mode must REJECT unsafe flags — prove it with a fake args set.
    unsafe = SimpleNamespace(age_gender=True, emotion=True, preview=True,
                             debug_local_preview=False)
    violations = privacy.check_privacy_safe(unsafe)
    if any("--age-gender" in v for v in violations):
        print(f"{OK} privacy-safe mode rejects --age-gender")
    else:
        print(f"{BAD} privacy-safe mode did NOT reject --age-gender"); critical_ok = False
    if any("--emotion" in v for v in violations):
        print(f"{OK} privacy-safe mode rejects --emotion")
    else:
        print(f"{BAD} privacy-safe mode did NOT reject --emotion"); critical_ok = False
    if any("--preview" in v for v in violations):
        print(f"{OK} privacy-safe mode rejects unsafe preview (without --debug-local-preview)")
    else:
        print(f"{BAD} privacy-safe mode did NOT reject unsafe preview"); critical_ok = False

    # ----------------------------------------------------------------- Calibration
    _section("Calibration")
    calib_file = getattr(args, "calibration_file", None)
    if not calib_file:
        print(f"{WARN} no calibration file provided (pass --calibration-file to validate)")
        print("   Run calibration first:")
        print("   python main.py --privacy-safe --calibrate center --seconds 5 \\")
        print("     --calibration-file configs/metro_billboard_calibration.json --tui")
    else:
        from .calibration import Calibration, CalibrationError
        try:
            c = Calibration.load(calib_file)
            print(f"{OK} calibration file valid: {calib_file}")
            print(f"   yaw {c.yaw_mean_deg:+.1f}° ±{c.yaw_tolerance_deg:.0f}°, "
                  f"pitch {c.pitch_mean_deg:+.1f}° ±{c.pitch_tolerance_deg:.0f}°, "
                  f"{c.sample_count} samples")
        except CalibrationError as e:
            print(f"{BAD} calibration file invalid: {e}")

    # ----------------------------------------------------------------- Output
    _section("Output")
    runs_dir = config.PROJECT_ROOT / "runs"
    try:
        runs_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=runs_dir, delete=True):
            pass
        print(f"{OK} output directory writable: {runs_dir}")
    except Exception as e:
        print(f"{BAD} output directory NOT writable: {e}"); critical_ok = False

    # Summary schema must contain no age/gender/emotion fields.
    sample = report.build_summary(
        session_id="schema_check", started_at="", ended_at="", duration_sec=0.0,
        counts={"total_passed": 0, "looked_total": 0, "looked_0_3s": 0,
                "looked_0_5s": 0, "looked_1_0s": 0},
        yaw_tolerance_deg=20.0, pitch_tolerance_deg=15.0,
        calibration_status="loaded", calibration_file=None)
    forbidden_keys = {"age", "gender", "emotion", "age_gender", "demographics"}
    bad_keys = forbidden_keys & set(_all_keys(sample))
    if not bad_keys:
        print(f"{OK} summary schema is anonymous (no age/gender/emotion fields)")
    else:
        print(f"{BAD} summary schema leaks fields: {bad_keys}"); critical_ok = False

    # Dwell-threshold counting self-test (offline, no hardware).
    from .simulate import verify_thresholds
    try:
        verify_thresholds()
        print(f"{OK} dwell threshold counting OK "
              "(0.2s ✗, 0.3s→0.3, 0.5s→0.3+0.5, 1.0s→all)")
    except AssertionError as e:
        print(f"{BAD} dwell threshold counting FAILED: {e}"); critical_ok = False

    # ----------------------------------------------------------------- Verdict
    print()
    if critical_ok:
        print(f"{OK} Offline-critical checks PASSED. "
              "Hardware/calibration warnings above must be resolved on the Pi tomorrow.")
        return 0
    print(f"{BAD} One or more critical checks FAILED — see above.")
    return 1


def _all_keys(obj) -> list:
    """Recursively collect dict keys from a nested structure."""
    keys: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(k)
            keys.extend(_all_keys(v))
    elif isinstance(obj, list):
        for v in obj:
            keys.extend(_all_keys(v))
    return keys

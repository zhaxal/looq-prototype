"""Offline simulation — verify the metric counters WITHOUT OAK-D Lite hardware.

This feeds synthetic local tracks (yaw/pitch + dwell patterns) into the SAME
metrics engine (`attention.metrics.count_buckets`) that the live field run uses,
then asserts the dwell-threshold logic and writes a real summary.json.

Run:
    python main.py --simulate-poses --privacy-safe \
        --calibration-file configs/example_calibration.json \
        --summary-out runs/sim_test/summary.json
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import metrics, report
from .calibration import Calibration, CalibrationError


def synthetic_tracks() -> list[metrics.TrackRecord]:
    """10 tracks; each present 2.0s (well above MIN_TRACK_SECS) so total_passed=10.

    Looking dwell pattern (per the field spec):
        2 tracks look 0.2s  -> count in nothing
        2 tracks look 0.3s  -> count in looked_0_3s
        2 tracks look 0.5s  -> count in looked_0_3s + looked_0_5s
        1 track  looks 1.0s -> count in all buckets
        3 tracks never look (0.0s)
    """
    dwell_pattern = [0.0, 0.0, 0.0, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 1.0]
    return [metrics.TrackRecord(track_id=i, first_seen=0.0, last_seen=2.0,
                                looking_accum_sec=d)
            for i, d in enumerate(dwell_pattern)]


def verify_thresholds() -> list[str]:
    """Per-track threshold assertions. Returns a list of human-readable check lines;
    raises AssertionError on any failure (so the doctor/CLI can report a hard fail).
    """
    lines: list[str] = []

    def bucket_of(accum: float) -> dict:
        return metrics.count_buckets([metrics.TrackRecord(1, 0.0, 2.0, accum)])

    # 0.2s -> nothing
    c = bucket_of(0.2)
    assert c["looked_0_3s"] == 0 and c["looked_0_5s"] == 0 and c["looked_1_0s"] == 0, c
    lines.append("0.2s dwell counts in: none ✓")

    # 0.3s -> looked_0_3s only
    c = bucket_of(0.3)
    assert c["looked_0_3s"] == 1 and c["looked_0_5s"] == 0 and c["looked_1_0s"] == 0, c
    lines.append("0.3s dwell counts in: looked_0_3s ✓")

    # 0.5s -> looked_0_3s + looked_0_5s
    c = bucket_of(0.5)
    assert c["looked_0_3s"] == 1 and c["looked_0_5s"] == 1 and c["looked_1_0s"] == 0, c
    lines.append("0.5s dwell counts in: looked_0_3s + looked_0_5s ✓")

    # 1.0s -> all buckets
    c = bucket_of(1.0)
    assert c["looked_0_3s"] == 1 and c["looked_0_5s"] == 1 and c["looked_1_0s"] == 1, c
    lines.append("1.0s dwell counts in: all buckets ✓")

    # Track shorter than MIN_TRACK_SECS is dropped from total_passed
    short = metrics.count_buckets(
        [metrics.TrackRecord(1, 0.0, metrics.MIN_TRACK_SECS / 2, 5.0)])
    assert short["total_passed"] == 0, short
    lines.append(f"track < {metrics.MIN_TRACK_SECS}s excluded from total_passed ✓")

    return lines


def run(args) -> int:
    """Entry point for `python main.py --simulate-poses`. Returns an exit code."""
    print("LOOQ offline simulation (no hardware required)\n")

    # 1) Per-track threshold checks.
    try:
        for line in verify_thresholds():
            print(f"  ✅ {line}")
    except AssertionError as e:
        print(f"  ❌ threshold check FAILED: {e}")
        return 1

    # 2) Aggregate scenario.
    tracks = synthetic_tracks()
    counts = metrics.count_buckets(tracks)
    expected = {"total_passed": 10, "looked_total": 5,
                "looked_0_3s": 5, "looked_0_5s": 3, "looked_1_0s": 1}
    print("\n  Scenario: 10 tracks (3 idle, 2×0.2s, 2×0.3s, 2×0.5s, 1×1.0s)")
    for k in expected:
        ok = counts[k] == expected[k]
        print(f"    {'✅' if ok else '❌'} {k:13s} = {counts[k]}  (expected {expected[k]})")
    if counts != {**counts, **expected}:
        print("\n  ❌ aggregate counts did not match expected values")
        return 1

    # 3) Calibration (optional) — populate summary fields honestly.
    calib = None
    calib_status = "simulated_no_calibration"
    calib_file = getattr(args, "calibration_file", None)
    if calib_file:
        try:
            calib = Calibration.load(calib_file)
            calib_status = "loaded"
        except CalibrationError as e:
            print(f"\n  ⚠️  calibration file not usable: {e}")
            calib_status = "invalid"

    yaw_tol, pitch_tol = report.calibration_tolerances(
        calib, fallback_yaw=20.0, fallback_pitch=15.0)

    # 4) Write a real summary.json if requested.
    out = getattr(args, "summary_out", None)
    if out:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        summary = report.build_summary(
            session_id=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=now, ended_at=now, duration_sec=0.0,
            counts=counts, yaw_tolerance_deg=yaw_tol, pitch_tolerance_deg=pitch_tol,
            calibration_status=calib_status, calibration_file=calib_file,
            mode="simulation", test_phase="simulation",
            field_decision={"status": "not_evaluated",
                            "reason": "offline simulation — synthetic data, not a field run",
                            "next_step": "run on hardware: python main.py field doctor"},
            manual_validation=report.blank_manual_validation("simulation"),
            extra={"simulation": True,
                   "note": "Synthetic data — no camera, no people, offline test only."},
        )
        path = report.write_summary(summary, out)
        print(f"\n  📄 wrote simulated summary → {path}")

    print("\n✅ Simulation passed — metric counters behave as specified.")
    return 0

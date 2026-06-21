"""Privacy-safe mode enforcement.

There is no code path in this project that does age/gender/emotion, saves face
crops/frames, or uploads anything — privacy is by construction. This module makes
that guarantee *loud*: in --privacy-safe mode, passing a forbidden flag fails hard
with a clear message instead of being silently ignored.
"""
from __future__ import annotations

# (argparse dest, flag spelling). Each, if truthy, is forbidden in privacy-safe mode.
FORBIDDEN_FLAGS = [
    ("age_gender",   "--age-gender"),
    ("emotion",      "--emotion"),
    ("demographics", "--demographics"),
    ("save_crops",   "--save-crops"),
    ("save_frames",  "--save-frames"),
    ("record",       "--record"),
    ("cloud_upload", "--cloud-upload"),
    ("upload",       "--upload"),
]


def check_privacy_safe(args) -> list[str]:
    """Return a list of human-readable violation messages ([] means OK)."""
    violations: list[str] = []
    for attr, flag in FORBIDDEN_FLAGS:
        if getattr(args, attr, False):
            violations.append(f"{flag} is forbidden in --privacy-safe mode.")
    # Preview is allowed only when the operator explicitly opts into LOCAL debug.
    if getattr(args, "preview", False) and not getattr(args, "debug_local_preview", False):
        violations.append(
            "--preview requires --debug-local-preview in --privacy-safe mode "
            "(preview is shown locally only and is never stored or uploaded).")
    return violations


def assert_privacy_safe(args) -> None:
    """Print every violation as `ERROR: ...` and raise SystemExit(2) if any."""
    violations = check_privacy_safe(args)
    if violations:
        for v in violations:
            print(f"ERROR: {v}")
        raise SystemExit(2)

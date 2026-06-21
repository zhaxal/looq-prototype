#!/usr/bin/env python3
"""Operator field wizard — same as `python main.py field <cmd>`.

    python scripts/field_wizard.py doctor
    python scripts/field_wizard.py preview
    python scripts/field_wizard.py calibrate
    python scripts/field_wizard.py controlled
    python scripts/field_wizard.py middle
    python scripts/field_wizard.py high
    python scripts/field_wizard.py bundle
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")     # type: ignore[attr-defined]
except Exception:
    pass

from attention import config        # noqa: E402

config.load_dotenv()

from attention import field_wizard   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(field_wizard.main(sys.argv[1:]))

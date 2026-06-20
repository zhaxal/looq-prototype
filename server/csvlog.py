"""CSV session log + summary — moved from main.py (server side now)."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

_CSV_FIELDS = [
    "ts", "track_id", "event", "looking", "look_seconds",
    "yaw", "pitch",
    "age", "gender",
    "emotion", "emotion_conf",
    "looking_total", "tracked_total",
]


class CsvLogger:
    def __init__(self, path: str | None) -> None:
        self._file = self._writer = None
        if path is None:
            return
        log_path = path or f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._file   = open(log_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()
        print(f"[log] writing to {log_path}")

    def write_row(self, row: dict) -> None:
        if self._writer is not None:
            self._writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def print_summary(session_ids: set, session_dwell: dict,
                  peak_looking: int, duration: float) -> None:
    n_total  = len(session_ids)
    n_looked = sum(1 for d in session_dwell.values() if d > 0)
    avg_dwell = (sum(session_dwell.values()) / n_looked) if n_looked else 0.0
    hrs, rem  = divmod(int(duration), 3600)
    mins, sec = divmod(rem, 60)
    W = 47
    print(f"\n{'═' * W}")
    print(f"  SESSION SUMMARY")
    print(f"{'═' * W}")
    print(f"  Duration      : {hrs:02d}:{mins:02d}:{sec:02d}")
    print(f"  Unique faces  : {n_total}")
    if n_total:
        print(f"  Looked at you : {n_looked}  ({n_looked / n_total:.0%})")
    else:
        print(f"  Looked at you : 0")
    print(f"  Peak looking  : {peak_looking}")
    print(f"  Avg dwell     : {avg_dwell:.1f}s  (among faces that looked)")
    print(f"{'═' * W}")

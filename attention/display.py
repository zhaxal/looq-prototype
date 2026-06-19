"""OpenCV preview overlay and ANSI terminal dashboard."""
from __future__ import annotations

import cv2
import depthai as dai

from . import config


def draw_preview(frame_msg, tracklets, pose_cache: dict,
                 age_gender_cache: dict, emotion_cache: dict,
                 looking_ids: set) -> None:
    frame = frame_msg.getCvFrame()
    h, w  = frame.shape[:2]

    for t in tracklets:
        x1 = int(t.roi.x * w)                       # VERIFY roi accessor
        y1 = int(t.roi.y * h)
        x2 = int((t.roi.x + t.roi.width) * w)
        y2 = int((t.roi.y + t.roi.height) * h)

        looking = t.id in looking_ids
        color   = (0, 255, 0) if looking else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = [f"id={t.id}"]
        if t.id in pose_cache:
            yaw, pitch = pose_cache[t.id]
            parts.append(f"y={yaw:+.0f} p={pitch:+.0f}")
        if t.id in age_gender_cache:
            gender, age = age_gender_cache[t.id]
            parts.append(f"{gender[0].upper()}{age}")
        if t.id in emotion_cache:
            label, score = emotion_cache[t.id]
            parts.append(f"{label}({score:.0%})")

        cv2.putText(frame, "  ".join(parts), (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Looking: {len(looking_ids)}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow("attention", frame)
    cv2.waitKey(1)


class LiveDisplay:
    """In-place ANSI dashboard for headless / SSH testing."""

    _HIDE  = "\033[?25l"
    _SHOW  = "\033[?25h"
    _HOME  = "\033[H"
    _CLEAR = "\033[2J"
    _W     = 64

    def __init__(self, face_res: str, fps: float) -> None:
        self.face_res = face_res
        self.fps      = fps
        print(self._HIDE, end="", flush=True)

    def close(self) -> None:
        print(self._SHOW, end="", flush=True)

    @staticmethod
    def _c(text: str, *codes: str) -> str:
        return "\033[" + ";".join(codes) + "m" + text + "\033[0m"

    def _rule(self, char: str = "─") -> str:
        return char * self._W

    def update(self, looking_ids: set, active_ids: set, tracklets,
               pose_cache: dict, age_gender_cache: dict, emotion_cache: dict,
               fps_actual: float | None = None, dwell: dict | None = None) -> None:
        fps_s = f"{fps_actual:.1f}fps" if fps_actual else f"{self.fps}fps"
        title = f" ATTENTION  {self.face_res} @ {fps_s}"
        lines = [
            self._CLEAR + self._HOME,
            self._c(title.ljust(self._W), "1", "7"),
            self._rule(),
        ]

        n_look  = len(looking_ids)
        n_total = len(active_ids)
        count_s = self._c(str(n_look), "1", "32")
        lines.append(
            f"  LOOKING  {count_s} / {n_total}   "
            + self._c("█" * n_look, "32")
            + self._c("░" * max(0, n_total - n_look), "2")
        )
        lines.append(self._rule())
        lines.append(self._c(
            f"  {'ID':>3}  {'LOOK':^4}  {'DWELL':>6}  {'YAW':>6}  {'PITCH':>6}  {'A/G':>6}  EMOTION",
            "2"))
        lines.append(self._rule("╌"))

        active_tracklets = [
            t for t in tracklets
            if t.status not in (dai.Tracklet.TrackingStatus.LOST,
                                dai.Tracklet.TrackingStatus.REMOVED)
        ]
        if not active_tracklets:
            lines.append(self._c("  no faces detected", "2"))
        for t in active_tracklets:
            looking = t.id in looking_ids
            look_s  = self._c(" YES", "1", "32") if looking else self._c("  no", "31")
            secs    = (dwell or {}).get(t.id, 0.0)
            yp      = pose_cache.get(t.id)
            ag      = age_gender_cache.get(t.id)
            em      = emotion_cache.get(t.id)
            yaw_s   = f"{yp[0]:+5.0f}°" if yp else "     ?"
            pitch_s = f"{yp[1]:+5.0f}°" if yp else "     ?"
            ag_s    = f"{ag[0][0]}{ag[1]:>3}" if ag else "      "
            emo_s   = f"{em[0]} {em[1]:.0%}" if em else ""
            lines.append(
                f"  {t.id:>3}  {look_s}  {secs:5.1f}s  {yaw_s}  {pitch_s}  {ag_s}  {emo_s}"
            )

        lines.append(self._rule())
        lines.append(self._c("  q / Ctrl-C to stop", "2"))
        print("\n".join(lines), end="", flush=True)

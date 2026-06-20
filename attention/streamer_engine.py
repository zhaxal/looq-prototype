"""StreamEngine — Pi streamer worker thread + server-status poller.

Mirrors the Engine pattern from the gaze-detection branch: runs the DepthAI
H.264 pipeline (same logic as pi_streamer.py) in a background thread and exposes
a thread-safe snapshot() for the GUI to poll at ~30Hz.

A second background thread polls GET /status on the server every 0.5 s so the
GUI can display live attention counts without any depthai dependency.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .netclient import FrameSender


def _ws_to_http(ws_url: str) -> str:
    """Convert ws://host/ingest -> http://host/status (or wss -> https)."""
    url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    # Strip the path and replace with /status.
    from urllib.parse import urlparse, urlunparse
    p = urlparse(url)
    return urlunparse(p._replace(path="/status", query="", fragment=""))


@dataclass
class StreamSnapshot:
    running: bool          = False
    connected: bool        = False   # WebSocket to server is live
    server_reachable: bool = False   # last GET /status succeeded
    fps: float             = 0.0
    sent: int              = 0
    dropped: int           = 0
    queued: int            = 0
    looking_total: int     = 0
    tracked_total: int     = 0
    looking_ids: list      = field(default_factory=list)
    peak_looking: int      = 0
    message: str           = ""
    error: bool            = False


class _SharedState:
    """Thread-safe container updated by worker threads; snapshot() copies it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._s    = StreamSnapshot()

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._s, k, v)

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            s = self._s
            return StreamSnapshot(
                running=s.running, connected=s.connected,
                server_reachable=s.server_reachable,
                fps=s.fps, sent=s.sent, dropped=s.dropped, queued=s.queued,
                looking_total=s.looking_total, tracked_total=s.tracked_total,
                looking_ids=list(s.looking_ids), peak_looking=s.peak_looking,
                message=s.message, error=s.error,
            )


class StreamEngine:
    """Manages the Pi's DepthAI H.264 pipeline + server-status polling."""

    def __init__(self, server_url: Optional[str] = None,
                 fps: float = config.DEFAULT_FPS,
                 res: str = "640x480",
                 test_video: Optional[Path] = None) -> None:
        self.server_url = server_url or config.LOOQ_SERVER_URL
        self.fps        = fps
        self.res        = res
        self.test_video = test_video

        self._state   = _SharedState()
        self._stop_ev = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- Public API ----------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return   # already running
        self._stop_ev.clear()
        self._threads = [
            threading.Thread(target=self._pipeline_thread, name="pipeline",
                             daemon=True),
            threading.Thread(target=self._poll_thread, name="status-poll",
                             daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._state.update(running=True, message="Starting…")

    def request_stop(self) -> None:
        self._stop_ev.set()

    def stop(self) -> None:
        self._stop_ev.set()
        for t in self._threads:
            t.join(timeout=6.0)
        self._threads = []
        self._state.update(running=False, connected=False, message="Stopped")

    def snapshot(self) -> StreamSnapshot:
        return self._state.snapshot()

    # --- Pipeline thread -----------------------------------------------------

    def _pipeline_thread(self) -> None:
        import depthai as dai  # local import — only available on Pi

        sender = FrameSender(self.server_url).start()
        w, h   = (int(x) for x in self.res.split("x"))
        last_stats = time.time()
        frame_count = 0
        last_fps_time = time.time()

        def _capture_ts(pkt) -> float:
            try:
                return pkt.getTimestampDevice().total_seconds()
            except Exception:
                try:
                    return pkt.getTimestamp().total_seconds()
                except Exception:
                    return time.time()

        while not self._stop_ev.is_set():
            started_ok = False
            try:
                with dai.Pipeline() as pipeline:
                    # Camera or replay source (NV12 for encoder).
                    if not self.test_video:
                        cam = pipeline.create(dai.node.Camera).build()
                        src = cam.requestOutput(
                            (w, h), dai.ImgFrame.Type.NV12, fps=self.fps)
                    else:
                        replay = pipeline.create(dai.node.ReplayVideo)
                        replay.setReplayVideoFile(str(self.test_video))
                        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
                        replay.setLoop(True)
                        manip = pipeline.create(dai.node.ImageManip)
                        manip.initialConfig.setResize(w, h)
                        manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
                        manip.setMaxOutputFrameSize(w * h * 3)
                        replay.out.link(manip.inputImage)
                        src = manip.out

                    enc = pipeline.create(dai.node.VideoEncoder)
                    enc.setDefaultProfilePreset(
                        self.fps,
                        dai.VideoEncoderProperties.Profile.H264_MAIN)
                    enc.setBitrateKbps(config.ENCODER_BITRATE_KBPS)
                    enc.setKeyframeFrequency(config.ENCODER_KEYFRAME_FREQ)
                    src.link(enc.input)
                    bitstream = enc.bitstream.createOutputQueue(
                        maxSize=4, blocking=False)

                    pipeline.start()
                    started_ok = True
                    seq = 0
                    self._state.update(message="Streaming…")

                    while pipeline.isRunning() and not self._stop_ev.is_set():
                        pkt = bitstream.get()
                        if pkt is None:
                            continue
                        sender.send(_capture_ts(pkt), seq, bytes(pkt.getData()))
                        seq += 1
                        frame_count += 1

                        now = time.time()
                        if now - last_fps_time >= 1.0:
                            fps_actual = frame_count / (now - last_fps_time)
                            frame_count = 0
                            last_fps_time = now
                            s, d, q = sender.stats()
                            self._state.update(
                                connected=sender.connected,
                                fps=fps_actual,
                                sent=s, dropped=d, queued=q,
                                error=False,
                            )

                    pipeline.stop()

            except Exception as exc:
                self._state.update(message=f"Pipeline error: {exc}",
                                   error=True, connected=False)
                if not started_ok:
                    break
                if not self._stop_ev.wait(timeout=2.0):
                    continue

        sender.close()
        self._state.update(running=False, connected=False)

    # --- Status poll thread --------------------------------------------------

    def _poll_thread(self) -> None:
        status_url = _ws_to_http(self.server_url)
        while not self._stop_ev.is_set():
            try:
                with urllib.request.urlopen(status_url, timeout=1.0) as resp:
                    data = json.loads(resp.read())
                latest = data.get("latest") or {}
                self._state.update(
                    server_reachable=True,
                    looking_total=latest.get("looking_total", 0),
                    tracked_total=latest.get("tracked_total", 0),
                    looking_ids=latest.get("looking_ids", []),
                    peak_looking=latest.get("peak_looking", 0),
                )
            except Exception:
                self._state.update(server_reachable=False)
            self._stop_ev.wait(timeout=0.5)

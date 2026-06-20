"""Resilient WebSocket sender for the Pi streamer.

Ships H.264 frames to the GPU server without ever blocking or crashing the
camera loop: a background thread owns the socket, send() enqueues onto a bounded
deque (drop-oldest when full / disconnected), and the thread reconnects with
backoff after any failure.

Wire format (binary WebSocket message, little-endian):
    struct "<dI"  ->  capture_ts (float64 seconds), seq (uint32)
    followed by the raw H.264 bitstream bytes for that frame.
"""
from __future__ import annotations

import struct
import threading
import time
from collections import deque

import websocket  # websocket-client

from . import config

_HEADER = struct.Struct("<dI")


class FrameSender:
    """Background WebSocket sender with a bounded drop-oldest queue."""

    def __init__(self, url: str | None = None, queue_max: int | None = None) -> None:
        self.url       = url or config.LOOQ_SERVER_URL
        self._queue: deque[bytes] = deque(
            maxlen=queue_max or config.LOOQ_SEND_QUEUE_MAX
        )
        self._lock     = threading.Lock()
        self._wake     = threading.Event()
        self._stop     = threading.Event()
        self._connected = False
        self._dropped   = 0
        self._sent      = 0
        self._thread    = threading.Thread(target=self._run, name="frame-sender",
                                           daemon=True)

    def start(self) -> "FrameSender":
        self._thread.start()
        return self

    @property
    def connected(self) -> bool:
        return self._connected

    def send(self, capture_ts: float, seq: int, payload: bytes) -> None:
        """Enqueue one encoded frame. Never blocks; drops oldest when full."""
        msg = _HEADER.pack(capture_ts, seq & 0xFFFFFFFF) + payload
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            self._queue.append(msg)
        self._wake.set()

    def stats(self) -> tuple[int, int, int]:
        """(sent, dropped, queued)."""
        with self._lock:
            return self._sent, self._dropped, len(self._queue)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    # --- internals -----------------------------------------------------------

    def _drain_one(self) -> bytes | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            ws = None
            try:
                ws = websocket.create_connection(self.url, timeout=5.0)
                self._connected = True
                backoff = 0.5
                print(f"[net] connected to {self.url}")
                while not self._stop.is_set():
                    msg = self._drain_one()
                    if msg is None:
                        self._wake.wait(timeout=0.5)
                        self._wake.clear()
                        continue
                    ws.send_binary(msg)
                    self._sent += 1
            except Exception as exc:  # connection refused, reset, timeout, …
                self._connected = False
                if not self._stop.is_set():
                    print(f"[net] disconnected ({exc}); retry in {backoff:.1f}s")
                    self._stop.wait(timeout=backoff)
                    backoff = min(backoff * 2, 5.0)
            finally:
                self._connected = False
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

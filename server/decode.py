"""H.264 → BGR frame decoding via PyAV."""
from __future__ import annotations

import av
import numpy as np


class H264Decoder:
    """Stateful decoder fed one encoded chunk at a time.

    The OAK sends one encoded frame per WebSocket message, but H.264 decoding can
    emit zero, one, or several frames per chunk (startup before the first
    keyframe, B-frame reordering). decode() returns a list of BGR ndarrays so the
    caller can pair them with the pending capture timestamps in order.
    """

    def __init__(self) -> None:
        self._codec = av.CodecContext.create("h264", "r")

    def decode(self, chunk: bytes) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        try:
            for packet in self._codec.parse(chunk):
                for frame in self._codec.decode(packet):
                    frames.append(frame.to_ndarray(format="bgr24"))
        except av.AVError:
            # Corrupt/partial data before the first keyframe — skip until sync.
            return frames
        return frames

    def close(self) -> None:
        try:
            self._codec.close()
        except Exception:
            pass

"""Fullscreen Tkinter GUI for the Pi H.264 streamer.

Polls StreamEngine.snapshot() at ~30Hz and displays:
  - Stream health: connection indicator, FPS, sent/dropped counts
  - Attention counts from the server: LOOKING NOW, TRACKED, PEAK

Adapted from attention/gui.py on the gaze-detection branch.
No camera preview — the Pi never receives decoded frames from the server.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from typing import Optional

from . import config
from .streamer_engine import StreamEngine

# Palette — same as gaze-detection branch
BG    = "#101418"
PANEL = "#1b2127"
TEXT  = "#e8eef2"
MUTED = "#8a97a3"
GREEN = "#28c76f"
RED   = "#ea5455"
BLUE  = "#3a7afe"
AMBER = "#ff9f43"
GREY  = "#5a6573"


class StreamerGUI:
    def __init__(self, engine: StreamEngine) -> None:
        self.engine = engine

        self.root = tk.Tk()
        self.root.title("Looq — Pi Streamer")
        self.root.configure(bg=BG)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self._tick_id: Optional[str] = None
        self._quitting = False

        self._build_widgets()
        self._tick_id = self.root.after(50, self._tick)

    # --- Layout --------------------------------------------------------------

    def _build_widgets(self) -> None:
        big   = tkfont.Font(family="DejaVu Sans", size=13, weight="bold")
        huge  = tkfont.Font(family="DejaVu Sans", size=28, weight="bold")
        label = tkfont.Font(family="DejaVu Sans", size=10)
        mono  = tkfont.Font(family="DejaVu Sans Mono", size=10)
        btn   = tkfont.Font(family="DejaVu Sans", size=12, weight="bold")

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(side="top", fill="x", padx=8, pady=(4, 2))
        tk.Label(header, text="LOOQ STREAMER", font=big, fg=TEXT, bg=BG).pack(side="left")
        self.fps_lbl = tk.Label(header, text="", font=label, fg=MUTED, bg=BG)
        self.fps_lbl.pack(side="right")

        # Controls (bottom — packed before body so they're always visible)
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(side="bottom", fill="x", padx=8, pady=(2, 6))

        def make_btn(parent, text, color, cmd, width=6):
            b = tk.Button(parent, text=text, font=btn, fg="#0b0f12", bg=color,
                          activebackground=color, relief="flat", width=width,
                          height=1, command=cmd, takefocus=0)
            b.pack(side="left", padx=3)
            return b

        self.start_btn = make_btn(controls, "START", GREEN, self._start)
        self.stop_btn  = make_btn(controls, "STOP",  RED,   self._stop)
        make_btn(controls, "QUIT", GREY, self._quit, width=5)

        # Server URL label (right side of controls)
        url_text = self.engine.server_url.replace("ws://", "").replace("wss://", "")
        tk.Label(controls, text=f"server: {url_text}", font=label,
                 fg=MUTED, bg=BG).pack(side="right", padx=8)

        # Status bar (second-to-bottom)
        self.status = tk.Label(self.root, text="Tap START to begin", font=label,
                               fg=MUTED, bg=BG, anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 2))

        # Body: health panel (left) + stats panel (right)
        body = tk.Frame(self.root, bg=BG)
        body.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        # --- Left: stream health ---
        health = tk.Frame(body, bg=PANEL)
        health.pack(side="left", fill="both", expand=True, padx=(0, 4))

        tk.Label(health, text="STREAM", font=big, fg=MUTED, bg=PANEL).pack(
            anchor="w", padx=16, pady=(12, 4))

        conn_row = tk.Frame(health, bg=PANEL)
        conn_row.pack(anchor="w", padx=16, pady=4)
        self.conn_dot = tk.Label(conn_row, text="●", font=big, fg=RED, bg=PANEL)
        self.conn_dot.pack(side="left")
        self.conn_lbl = tk.Label(conn_row, text="NOT STARTED", font=btn,
                                 fg=RED, bg=PANEL)
        self.conn_lbl.pack(side="left", padx=6)

        self.metrics_lbl = tk.Label(health, text="", font=mono, fg=MUTED, bg=PANEL)
        self.metrics_lbl.pack(anchor="w", padx=16, pady=(2, 4))

        # Queue/buffer indicator
        self.queue_lbl = tk.Label(health, text="", font=label, fg=MUTED, bg=PANEL)
        self.queue_lbl.pack(anchor="w", padx=16)

        # Looking IDs list (scrolling label — just a text label for simplicity)
        tk.Label(health, text="LOOKING IDs", font=label, fg=MUTED, bg=PANEL).pack(
            anchor="w", padx=16, pady=(16, 2))
        self.ids_lbl = tk.Label(health, text="—", font=mono, fg=GREEN, bg=PANEL,
                                anchor="w", justify="left", wraplength=220)
        self.ids_lbl.pack(anchor="w", padx=16)

        # --- Right: attention stats (same as gaze-detection) ---
        stats = tk.Frame(body, bg=PANEL, width=220)
        stats.pack(side="right", fill="y", padx=(4, 0))
        stats.pack_propagate(False)

        def stat(title: str, color: str) -> tk.Label:
            tk.Label(stats, text=title, font=label, fg=MUTED, bg=PANEL).pack(
                anchor="w", padx=12, pady=(14, 0))
            v = tk.Label(stats, text="—", font=huge, fg=color, bg=PANEL)
            v.pack(anchor="w", padx=12)
            return v

        self.looking_val = stat("LOOKING NOW", GREEN)
        self.tracked_val = stat("TRACKED",     TEXT)
        self.peak_val    = stat("PEAK",        BLUE)

    # --- Button handlers -----------------------------------------------------

    def _start(self) -> None:
        self.engine.start()

    def _stop(self) -> None:
        self.engine.request_stop()

    def _quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        if self._tick_id:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.engine.stop()
        self.root.destroy()

    # --- Render loop ---------------------------------------------------------

    def _tick(self) -> None:
        if self._quitting:
            return
        s = self.engine.snapshot()

        # Connection indicator
        if not s.running:
            dot_color, conn_text = RED, "NOT STARTED"
        elif s.connected and s.server_reachable:
            dot_color, conn_text = GREEN, "CONNECTED"
        elif s.connected:
            dot_color, conn_text = AMBER, "NO SERVER RESPONSE"
        else:
            dot_color, conn_text = AMBER, "RECONNECTING…"

        self.conn_dot.config(fg=dot_color)
        self.conn_lbl.config(text=conn_text, fg=dot_color)

        # Stream metrics
        if s.running:
            self.fps_lbl.config(text=f"{s.fps:.1f} fps")
            self.metrics_lbl.config(
                text=f"sent {s.sent:,}  ·  dropped {s.dropped:,}")
            self.queue_lbl.config(text=f"queued {s.queued}")
        else:
            self.fps_lbl.config(text="")
            self.metrics_lbl.config(text="")
            self.queue_lbl.config(text="")

        # Attention counts (from server)
        if s.server_reachable:
            self.looking_val.config(text=str(s.looking_total))
            self.tracked_val.config(text=str(s.tracked_total))
            self.peak_val.config(text=str(s.peak_looking))
            ids_text = ", ".join(str(i) for i in s.looking_ids) or "—"
            self.ids_lbl.config(text=ids_text)
        else:
            self.looking_val.config(text="—")
            self.tracked_val.config(text="—")
            self.peak_val.config(text="—")
            self.ids_lbl.config(text="—")

        # Status bar
        if s.error:
            self.status.config(text=s.message, fg=RED)
        elif s.message:
            self.status.config(text=s.message, fg=MUTED)
        elif s.running:
            self.status.config(
                text=f"Streaming  ·  {s.tracked_total} face(s) tracked on server",
                fg=MUTED)

        self._tick_id = self.root.after(33, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def launch(server_url: Optional[str] = None,
           fps: float = config.DEFAULT_FPS,
           res: str = "640x480",
           test_video: Optional[Path] = None) -> None:
    engine = StreamEngine(server_url=server_url, fps=fps, res=res,
                          test_video=test_video)
    StreamerGUI(engine).run()

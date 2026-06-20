"""Fullscreen touch GUI for the ad-attention device.

Tkinter on the main thread; the Engine runs the camera/inference in its own thread.
The GUI polls Engine.snapshot() ~30x/s and renders the annotated frame via Pillow.
All controls are large touch targets — no keyboard or terminal needed.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk

from . import config
from .engine import Engine

# Palette
BG      = "#101418"
PANEL   = "#1b2127"
TEXT    = "#e8eef2"
MUTED   = "#8a97a3"
GREEN   = "#28c76f"
RED     = "#ea5455"
BLUE    = "#3a7afe"
AMBER   = "#ff9f43"


class AdAttentionGUI:
    def __init__(self, engine: Engine, settings: config.Settings) -> None:
        self.engine   = engine
        self.settings = settings
        self._imgtk   = None      # keep a reference so Tk doesn't GC the image

        self.root = tk.Tk()
        self.root.title("Ad Attention")
        self.root.configure(bg=BG)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self._tick_id: str | None = None   # after() handle so we can cancel it
        self._quitting = False

        self._build_widgets()
        self._tick_id = self.root.after(50, self._tick)

    # --- Layout --------------------------------------------------------------

    def _build_widgets(self) -> None:
        big   = tkfont.Font(family="DejaVu Sans", size=13, weight="bold")
        huge  = tkfont.Font(family="DejaVu Sans", size=24, weight="bold")
        label = tkfont.Font(family="DejaVu Sans", size=10)
        btn   = tkfont.Font(family="DejaVu Sans", size=12, weight="bold")

        # Header — packed first so it's always at top
        header = tk.Frame(self.root, bg=BG)
        header.pack(side="top", fill="x", padx=8, pady=(4, 2))
        tk.Label(header, text="AD ATTENTION", font=big, fg=TEXT, bg=BG).pack(side="left")
        self.fps_lbl = tk.Label(header, text="", font=label, fg=MUTED, bg=BG)
        self.fps_lbl.pack(side="right")

        # Controls + status — packed at bottom BEFORE the body so they are
        # always reserved and never pushed off-screen when the preview fills.
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(side="bottom", fill="x", padx=8, pady=(2, 6))

        def make_btn(parent, text, color, cmd, width=6):
            b = tk.Button(parent, text=text, font=btn, fg="#0b0f12", bg=color,
                          activebackground=color, relief="flat", width=width,
                          height=1, command=cmd, takefocus=0)
            b.pack(side="left", padx=3)
            return b

        self.start_btn = make_btn(controls, "START",     GREEN, self._start)
        self.stop_btn  = make_btn(controls, "STOP",      RED,   self._stop)
        make_btn(controls, "CALIBRATE", AMBER, self._calibrate, width=9)

        # Yaw offset nudger
        off = tk.Frame(controls, bg=PANEL)
        off.pack(side="left", padx=10)
        tk.Label(off, text="YAW", font=label, fg=MUTED, bg=PANEL).grid(
            row=0, column=0, columnspan=3, pady=(2, 0))
        tk.Button(off, text="−", font=btn, fg=TEXT, bg=PANEL, relief="flat",
                  width=2, command=lambda: self._nudge(-2), takefocus=0).grid(row=1, column=0, padx=3, pady=2)
        self.offset_lbl = tk.Label(off, text="0°", font=btn, fg=TEXT, bg=PANEL, width=4)
        self.offset_lbl.grid(row=1, column=1)
        tk.Button(off, text="+", font=btn, fg=TEXT, bg=PANEL, relief="flat",
                  width=2, command=lambda: self._nudge(+2), takefocus=0).grid(row=1, column=2, padx=3, pady=2)

        make_btn(controls, "RESET", BLUE,      self._reset, width=6)
        make_btn(controls, "QUIT",  "#5a6573", self._quit,  width=5)

        self.status = tk.Label(self.root, text="Tap START to begin", font=label,
                               fg=MUTED, bg=BG, anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 2))

        # Body: preview (left) + stats (right) — packed last, takes remaining space
        body = tk.Frame(self.root, bg=BG)
        body.pack(side="top", fill="both", expand=True, padx=8, pady=2)

        self.preview = tk.Label(body, bg="#000000")
        self.preview.pack(side="left", fill="both", expand=True)

        stats = tk.Frame(body, bg=PANEL)
        stats.pack(side="right", fill="y", padx=(8, 0))

        def stat(title: str, color: str) -> tk.Label:
            tk.Label(stats, text=title, font=label, fg=MUTED, bg=PANEL).pack(
                anchor="w", padx=12, pady=(10, 0))
            v = tk.Label(stats, text="0", font=huge, fg=color, bg=PANEL)
            v.pack(anchor="w", padx=12)
            return v

        self.looking_val = stat("LOOKING NOW", GREEN)
        self.total_val   = stat("TOTAL VIEWERS", TEXT)
        self.dwell_val   = stat("AVG DWELL", BLUE)

        # Calibration overlay (hidden unless calibrating)
        self.overlay = tk.Label(self.preview, text="", font=huge, fg=AMBER, bg="#000000")

    # --- Button handlers -----------------------------------------------------

    def _start(self) -> None:
        # If a previous stop is still cleaning up, wait for the thread to finish
        # before starting a new one (engine.start() checks this internally).
        self.engine.start()

    def _stop(self)      -> None: self.engine.request_stop()   # non-blocking; UI stays live
    def _calibrate(self) -> None: self.engine.calibrate(5.0)
    def _reset(self)     -> None: self.engine.reset_session()

    def _nudge(self, delta: float) -> None:
        self.engine.nudge_yaw_offset(delta)
        self.offset_lbl.config(text=f"{self.settings.yaw_offset:+.0f}°")

    def _quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        # Cancel the pending _tick before touching the engine or destroying the window.
        if self._tick_id:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.engine.stop()        # blocks until the worker thread finishes cleanly
        self.root.destroy()

    # --- Render loop ---------------------------------------------------------

    def _tick(self) -> None:
        if self._quitting:
            return
        s = self.engine.snapshot()

        self.looking_val.config(text=str(s.looking_now))
        self.total_val.config(text=str(s.total_unique))
        self.dwell_val.config(text=f"{s.avg_dwell:.1f}s")
        self.fps_lbl.config(text=f"{s.fps:.1f} fps" if s.running else "stopped")
        self.offset_lbl.config(text=f"{self.settings.yaw_offset:+.0f}°")

        if s.message:
            self.status.config(text=s.message.split("\n")[0],
                               fg=RED if s.error else MUTED)
        elif s.running:
            self.status.config(text=f"Running — {s.tracked_now} face(s) tracked", fg=MUTED)

        if s.frame is not None:
            self._show_frame(s.frame)

        if s.calibrating:
            self.overlay.config(text=f"Look at the ad\n{s.calib_remaining:.0f}")
            self.overlay.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.overlay.place_forget()

        self._tick_id = self.root.after(33, self._tick)

    def _show_frame(self, bgr) -> None:
        w = self.preview.winfo_width()  or 640
        h = self.preview.winfo_height() or 480
        fh, fw = bgr.shape[:2]
        scale  = min(w / fw, h / fh)
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(rgb).resize(
            (max(1, int(fw * scale)), max(1, int(fh * scale))))
        self._imgtk = ImageTk.PhotoImage(img)
        self.preview.config(image=self._imgtk)

    def run(self) -> None:
        self.root.mainloop()


def launch(settings: config.Settings | None = None) -> None:
    settings = settings or config.Settings.load()
    engine   = Engine(settings)
    AdAttentionGUI(engine, settings).run()

#!/usr/bin/env python3
"""Touch-GUI entry point for the ad-attention device.

Run on the Pi (or laptop with a desktop session):

    python app.py

Loads settings.json, opens the fullscreen GUI, and runs the camera/inference engine
in a background thread. Everything else is driven from on-screen touch buttons.
"""
from attention import config

config.load_dotenv()           # must run before depthai is imported (Hub API key)

from attention.gui import launch   # noqa: E402 — import after load_dotenv


def main() -> None:
    launch(config.Settings.load())


if __name__ == "__main__":
    main()

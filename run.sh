#!/usr/bin/env bash
# Launch the fullscreen touch GUI. Used by the desktop icon and runnable directly.
cd "$(dirname "$0")"
exec ./.venv/bin/python app.py

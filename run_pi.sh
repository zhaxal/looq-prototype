#!/usr/bin/env bash
# Launch the fullscreen Pi streamer GUI.
# Used by the desktop icon and runnable directly: ./run_pi.sh
cd "$(dirname "$0")"
exec ./.venv/bin/python pi_gui.py "$@"

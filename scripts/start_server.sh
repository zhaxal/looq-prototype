#!/usr/bin/env bash
# Start the GPU attention server (FastAPI + uvicorn).
# Run from the repo root on the server box (not the Pi).
#
#     bash scripts/start_server.sh
#
# Env overrides (or set in .env):
#   LOOQ_SERVER_HOST  (default 0.0.0.0)
#   LOOQ_SERVER_PORT  (default 8000)
#   LOOQ_AGE_GENDER   1/0 (default 1)
#   LOOQ_EMOTION      1/0 (default 1)
#   LOOQ_LOG          path for CSV log; empty string = auto-named
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load .env if present (same logic as attention/config.py load_dotenv).
if [ -f .env ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

HOST="${LOOQ_SERVER_HOST:-0.0.0.0}"
PORT="${LOOQ_SERVER_PORT:-8000}"

echo "Starting looq attention server on ${HOST}:${PORT} …"
exec ./.venv/bin/uvicorn server.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1

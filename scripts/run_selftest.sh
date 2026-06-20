#!/usr/bin/env bash
# Download the ONNX models for the GPU server, then run the smoke-test.
# Run on the server box (not the Pi) after `git clone`:
#
#     bash scripts/run_selftest.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Downloading ONNX models…"
./.venv/bin/python scripts/download_models_onnx.py

echo ""
echo "==> Running server self-test…"
./.venv/bin/python -m server.selftest

echo ""
echo "All done. Start the server with:  bash scripts/start_server.sh"

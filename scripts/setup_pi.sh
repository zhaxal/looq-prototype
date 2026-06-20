#!/usr/bin/env bash
# One-shot provisioning for the looq Pi streamer on Raspberry Pi OS (64-bit).
# Idempotent — safe to re-run. Run once after `git clone`:
#
#     bash scripts/setup_pi.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$1"; }

# 1. System packages
say "Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-tk libusb-1.0-0 git

# 2. OAK udev rule so the VPU is reachable without sudo.
RULE=/etc/udev/rules.d/80-movidius.rules
if ! grep -qs '03e7' "$RULE" 2>/dev/null; then
    say "Installing OAK udev rule…"
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
        | sudo tee "$RULE" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
else
    say "OAK udev rule already present."
fi

# 3. Virtualenv + Python deps (Pi-side only — no server deps here).
say "Creating virtualenv and installing Python deps…"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements-pi.txt -q

# 4. Make the launcher executable.
chmod +x run_pi.sh

# 5. Install the desktop launcher.
say "Installing desktop launcher…"
DESKTOP_FILE="$ROOT/looq-pi.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Looq Streamer
Comment=Stream camera to the attention-counting GPU server
Exec=$ROOT/run_pi.sh
Icon=camera-video
Terminal=false
Categories=Utility;
EOF
chmod +x "$DESKTOP_FILE"
mkdir -p "$HOME/Desktop" "$HOME/.local/share/applications"
cp "$DESKTOP_FILE" "$HOME/Desktop/looq-pi.desktop"
cp "$DESKTOP_FILE" "$HOME/.local/share/applications/looq-pi.desktop"
gio set "$HOME/Desktop/looq-pi.desktop" metadata::trusted true 2>/dev/null || true

say "Done."
cat <<EOF

Next steps:
  1. Set the server address in .env:
         echo 'LOOQ_SERVER_URL=ws://SERVER_IP:8000/ingest' >> .env
  2. Plug the OAK-D-Lite into a powered USB hub / Y-cable (Pi USB alone is unreliable).
  3. Double-tap "Looq Streamer" on the desktop, OR run:
         ./run_pi.sh

Verify the camera is detected first:
  ./.venv/bin/python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
EOF

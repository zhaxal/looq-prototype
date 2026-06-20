#!/usr/bin/env bash
# One-shot provisioning for the ad-attention device on Raspberry Pi OS (64-bit).
# Idempotent — safe to re-run. Run once after `git clone`:
#
#     bash scripts/setup_pi.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$1"; }

# 1. System packages (Tkinter GUI, USB, git). Pillow/opencv come from pip.
say "Installing system packages…"
sudo apt-get update
sudo apt-get install -y python3-venv python3-tk libusb-1.0-0 git

# 2. OAK udev rule so the VPU is reachable without sudo.
RULE=/etc/udev/rules.d/80-movidius.rules
if ! grep -qs '03e7' "$RULE" 2>/dev/null; then
    say "Installing OAK udev rule…"
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee "$RULE" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
else
    say "OAK udev rule already present."
fi

# 3. Virtualenv + Python deps.
say "Creating virtualenv and installing Python deps…"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# 4. Restore the two model archives from git if missing.
say "Ensuring models are present…"
for m in yunet-320x240 yunet-640x480 head-pose-60x60; do
    f="models/${m}.rvc2.tar.xz"
    [ -f "$f" ] || git checkout HEAD -- "$f" || true
done

# 5. Install the desktop launcher (tap to start).
say "Installing desktop launcher…"
chmod +x run.sh
DESKTOP_FILE="$ROOT/adwatch.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Ad Attention
Comment=Count who looks at the advertisement
Exec=$ROOT/run.sh
Icon=camera-video
Terminal=false
Categories=Utility;
EOF
chmod +x "$DESKTOP_FILE"
mkdir -p "$HOME/Desktop" "$HOME/.local/share/applications"
cp "$DESKTOP_FILE" "$HOME/Desktop/adwatch.desktop"
cp "$DESKTOP_FILE" "$HOME/.local/share/applications/adwatch.desktop"
# Mark the desktop copy trusted on PIXEL desktop (best-effort).
gio set "$HOME/Desktop/adwatch.desktop" metadata::trusted true 2>/dev/null || true

say "Done."
cat <<EOF

Next steps:
  1. Plug the OAK-D-Lite into a *powered* USB hub / Y-cable (Pi USB alone is unreliable).
  2. Double-tap the "Ad Attention" icon on the desktop, OR run:  ./run.sh
  3. Tap CALIBRATE once (stand where viewers will, looking at the ad) to set the offset.

Verify the camera is seen first:
  ./.venv/bin/python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
EOF

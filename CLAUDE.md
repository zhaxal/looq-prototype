# looq-prototype — Ad Attention

Portable device that measures how many people look at an advertisement and for how
long. OAK-D-Lite + Raspberry Pi 4 with a **touchscreen**, operated entirely from a
fullscreen touch GUI (no terminal).

## Hardware & Deploy
- Camera: OAK-D-Lite (RVC2 VPU), sitting **beside** the ad on a fixed side,
  mounted **upside down** — the sensor is rotated 180° on-device
  (`Settings.flip_180` → `CameraImageOrientation.ROTATE_180_DEG`) so the whole
  pipeline and preview see an upright image; pose signs match a normal mount.
- Host: Raspberry Pi 4 with touchscreen (desktop session — the GUI is Tkinter).
- Power: powered USB hub or Y-cable (Pi USB alone is unreliable).
- Provision: `bash scripts/setup_pi.sh` (apt deps, OAK udev rule, venv, models, icon).

## SDK
DepthAI **v3**. Do not use v2 patterns.

## Key Design Decisions
- "Looking" = **head pose** (yaw/pitch), NOT gaze estimation.
- Camera is beside the ad, so "looking at the ad" is an **angular offset** from the
  camera axis. The **Calibrate** button measures that offset empirically (and resolves
  its sign) — see `Settings.yaw_offset` / `is_looking_at_ad()`.
- Metrics only: **unique viewer count + dwell time**. No age/gender/emotion.
- Settings persist to `settings.json` (offsets, tolerances, face_res, fps, log,
  flip_180).

## Architecture
GUI on the main thread; the DepthAI pipeline + host matching loop run in a worker
thread (`attention/engine.py`) publishing a thread-safe `SharedState` the GUI polls.
Frames render via Pillow `ImageTk`, so the Pi keeps `opencv-python-headless`.

```
Camera(BGR888p 320x240) → PNN[YuNet] → ObjectTracker(SHORT_TERM_IMAGELESS+UNIQUE_ID)
                                      └→ FrameCropper[60x60] → PNN[head-pose] → GatherData → poses
```

## Project Layout
```
app.py                  touch-GUI entry point  →  python app.py
main.py                 headless/SSH CLI (--calibrate, --log, offset flags)
attention/
  config.py             constants + Settings (settings.json) + load_dotenv()
  processing.py         LookState, geometry, is_looking_at_ad(), pose parser, VERIFY probes
  pipeline.py           DepthAI v3 pipeline (single head-pose branch)
  engine.py             worker thread, host loop, SharedState, calibrate, CSV log
  gui.py                Tkinter fullscreen touch UI
scripts/
  setup_pi.sh           one-shot Pi provisioning
  download_models.py    fetch YuNet + head-pose from the zoo (models restored from git otherwise)
run.sh, adwatch.desktop tap-to-launch on the Pi desktop
models/                 yunet-{320x240,640x480} + head-pose-60x60 archives
```

## Confirmed v3 Model Zoo Slugs
- Face detect: `luxonis/yunet:<res>` (320x240 default; also 640x360/640x480/960x720)
- Head pose: `luxonis/head-pose-estimation:60x60`

## Known Risks / VERIFY Markers
`# VERIFY` comments flag accessor/enum names that may drift across depthai-nodes
releases — they print `[VERIFY]` warnings on the first real-hardware run:
- Pose message accessor + angle order (`extract_pose`)
- GatherData field names (`reference_data` / `detections` / `items`)
- Tracklet `.roi` accessor; tracker type / id-policy enums

## Docs
- Luxonis docs index: https://docs.luxonis.com/llms.txt
- v3 SDK overview: https://docs.luxonis.com/software-v3/depthai.md
- All nodes: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes.md
- Model Zoo: https://docs.luxonis.com/software-v3/ai-inference/model-source/zoo.md

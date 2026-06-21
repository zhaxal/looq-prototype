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
main.py                 field/headless CLI: --doctor, --simulate-poses, --privacy-safe,
                        --calibrate center, field run + TUI, summary.json + events.csv
attention/
  config.py             constants + Settings (settings.json) + load_dotenv()
  processing.py         LookState, geometry, is_looking_at_ad(), pose parser, VERIFY probes
  pipeline.py           DepthAI v3 pipeline (single head-pose branch)
  engine.py             worker thread, host loop, SharedState (incl. dwell buckets), CSV/events
  gui.py                Tkinter fullscreen touch UI
  metrics.py            pure dwell-bucket counting (total_passed/looked_0_3/0_5/1_0) — testable
  calibration.py        Calibration JSON profile (billboard direction) load/save/validate
  report.py             summary.json schema builder (privacy block + limitations)
  privacy.py            assert_privacy_safe() — reject --age-gender/--emotion/--upload/…
  doctor.py             --doctor checks (software/hardware/privacy/calibration/output)
  simulate.py           --simulate-poses offline counter verification (no hardware)
  field_wizard.py       `main.py field <doctor|preview|calibrate|controlled|middle|high|bundle>`
scripts/
  setup_pi.sh           one-shot Pi provisioning
  download_models.py    fetch YuNet + head-pose from the zoo (models restored from git otherwise)
  collect_debug_bundle.py  privacy-safe runs/debug_bundle_*.zip (text/JSON only, no images)
  field_wizard.py       thin wrapper → same as `python main.py field <phase>`
configs/                calibration profiles (example_calibration.json committed)
runs/                   per-session summary.json + events.csv (gitignored)
docs/FIELD_GUIDE.md     single operator guide (setup, steps, results, privacy, troubleshooting)
run.sh, adwatch.desktop tap-to-launch on the Pi desktop
models/                 yunet-{320x240,640x480} + head-pose-60x60 archives
```

## Field MVP (privacy-safe billboard counter)
- Metric = **LIKELY_ATTENTION**: unique local face/head track whose head pose stayed
  near the **calibrated billboard direction** for a dwell threshold. Not gaze/identity/
  demographics. The looking decision is `is_looking_at_ad()` against `yaw_offset/pitch_offset`,
  which the calibration file sets (camera sits *beside* the ad).
- Reported numbers: `total_passed, looked_total, looked_0_3s, looked_0_5s, looked_1_0s`.
  Bucket logic lives in `attention/metrics.py` and is verified offline by `--simulate-poses`.
- Privacy by construction: no age/gender/emotion, no crops/frames/video saved, no upload.
  `--privacy-safe` makes that strict (rejects unsafe flags). Runbook: `docs/`.
- **Counting ROI** (`--counting-roi x1,y1,x2,y2`, normalized): `is_track_inside_roi()`
  gates counting/dwell to the billboard exposure zone (background filter). Optional;
  warns if absent. **Field phases** (`--test-phase`) + `report.compute_field_decision()`
  produce `field_decision: pass|warning|fail|not_evaluated` for controlled sanity tests.
- **Results persistence**: each phase writes `runs/<ts>_<phase>/` (summary.json, events.csv,
  command.txt, README.txt, calibration.json copy); calibration also keeps a timestamped copy.
  Billboard for this PoC is **1.0m × 2.0m**, camera height ~1.6m.

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

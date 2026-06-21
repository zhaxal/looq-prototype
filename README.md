# LOOQ — Privacy-Safe Billboard Attention Counter

Counts **likely attention** to a billboard: how many unique people look at it, and
for how long — locally, anonymously, on a Raspberry Pi 4 + OAK-D Lite. No age,
gender, emotion, identity, or cloud. Head pose only.

> Status: software-level checks pass and offline counters are verified. The OAK-D
> Lite + DepthAI pipeline is unchanged from the working prototype but has **not**
> been re-verified on hardware in this change — it needs a field run on the Pi.

**Prompt / implementation version:**
- **V1.0**: privacy-safe counter, calibration, doctor, simulation, summary, debug bundle
- **V1.2**: field wizard, operator runbook, phase validation, ROI support, safer result persistence

- **⭐ The one doc the field team needs:** [docs/FIELD_GUIDE.md](docs/FIELD_GUIDE.md) — setup, steps, results, privacy, and troubleshooting in one place.

## Tomorrow Metro Field Test: One Page Flow

Run these in order. Each writes results to its own `runs/<timestamp>_<phase>/` folder.
Press **Ctrl+C** to finish a live phase. Add `--print` to see the long command.

```bash
python main.py field doctor       # PHASE 0 — OAK detected? privacy ok? output writable?
python main.py field preview      # PHASE 1 — faces visible, 2–5 m, camera beside billboard
python main.py field calibrate    # PHASE 2 — ONE person looks at billboard centre 5s
python main.py field controlled --counting-roi 0.15,0.20,0.85,0.95 \
    --manual-total 10 --manual-lookers 5     # PHASE 3 — 5 look / 5 no-look → PASS/WARN/FAIL
python main.py field middle  --counting-roi 0.15,0.20,0.85,0.95   # PHASE 4 — 3–5 min moderate
python main.py field high    --counting-roi 0.15,0.20,0.85,0.95   # PHASE 5 — 3–5 min stress only
python main.py field bundle       # if broken: makes runs/debug_bundle_*.zip to send back
```
Only continue past PHASE 3 if the controlled test is **PASS** (or acceptable **WARNING**).
Every `field` command also works as `python scripts/field_wizard.py <phase>`.

## The five numbers

| number | meaning |
|---|---|
| `total_passed` | unique valid face/head tracks in the session |
| `looked_total` | tracks with calibrated looking dwell ≥ 0.3 s (= `looked_0_3s`) |
| `looked_0_3s`  | tracks with looking dwell ≥ 0.3 s |
| `looked_0_5s`  | tracks with looking dwell ≥ 0.5 s |
| `looked_1_0s`  | tracks with looking dwell ≥ 1.0 s |

Metric tier: **`LIKELY_ATTENTION`** = local unique track whose head pose stayed near
the calibrated billboard direction for a dwell threshold. Not gaze, not identity,
not demographics, not final reach.

## Hardware
- **OAK-D Lite** (primary camera, DepthAI v3) — use a stable USB3 cable; powered hub
  if possible.
- **Raspberry Pi 4 8GB** (host).
- Do **not** use the Raspberry Pi Camera Module as the primary path.

## Install
```bash
python -m pip install -r requirements.txt
```

## Commands

Doctor (safe without a camera):
```bash
python main.py --doctor
```

Calibrate (one person looking at the billboard):
```bash
python main.py --privacy-safe \
  --calibrate center --seconds 5 \
  --camera-id oak_d_lite_01 --camera-height-m 1.6 \
  --billboard-id metro_billboard_001 --billboard-width-m 1.0 --billboard-height-m 2.0 \
  --calibration-file configs/metro_billboard_calibration.json --tui
```

Privacy-safe field run (with a counting ROI to ignore background faces):
```bash
python main.py --privacy-safe \
  --calibration-file configs/metro_billboard_calibration.json \
  --counting-roi 0.15,0.20,0.85,0.95 --test-phase controlled \
  --tui --summary-out runs/metro_test/summary.json
```
Press **Ctrl+C** to finish → writes `summary.json` (+ `events.csv`).

Local debug preview (frames shown locally, never stored/uploaded):
```bash
python main.py --privacy-safe \
  --debug-local-preview --preview \
  --calibration-file configs/metro_billboard_calibration.json
```

Offline simulation (no hardware — verifies the counters):
```bash
python main.py --simulate-poses --privacy-safe \
  --calibration-file configs/example_calibration.json \
  --summary-out runs/sim_test/summary.json
```

Debug bundle (privacy-safe zip to send back):
```bash
python scripts/collect_debug_bundle.py     # → runs/debug_bundle_<ts>.zip
```

Touch GUI (unchanged; for the kiosk build):
```bash
python app.py
```

## Counting ROI (strongly recommended in metro)
Metro scenes have lots of background faces. The optional `--counting-roi "x1,y1,x2,y2"`
(normalized 0..1, e.g. `0.15,0.20,0.85,0.95`) counts a track only when its **center**
enters the box, and accumulates looking dwell **only while inside** it. Without a ROI
you get a `WARNING: no counting ROI provided` and background faces may be counted.
The ROI is stored in `summary.json` and drawn in the local debug preview.

## Field phases & pass/fail
`--test-phase controlled|middle_traffic|high_traffic` tags the run and drives a
**field sanity check** written to `summary.json` as `field_decision` (status =
`pass|warning|fail|not_evaluated`). With `--manual-total`/`--manual-lookers`, the
controlled phase checks `total_passed` (7–13 PASS, 5–15 WARNING for a target of 10)
and `looked_0_5s` (3–7 PASS, 2–8 WARNING for a target of 5). High traffic is always
`not_evaluated` (stress test only). These are field sanity checks, not science.

## What to report
`total_passed`, `looked_total`, `looked_0_3s`, `looked_0_5s`, `looked_1_0s`

## What NOT to report (not produced)
age, gender, emotion, identity, "verified attention", "verified reach", demographics

## Privacy (enforced by code)
No raw video / frame / screenshot upload, no face crop storage, no embeddings, no
identity, no cross-location tracking, no cloud, no age/gender/emotion/demographics.
Local track IDs reset every session. `--privacy-safe` rejects unsafe flags loudly.

## Known limitations
- `total_passed` = unique valid face/head **tracks**, not final unique pedestrian reach.
- `LIKELY_ATTENTION` = calibrated head-pose **direction**, not verified geometric gaze.
- Track loss may **overcount** the same person (no re-identification, by design).
- Calibration is **placement-specific** — redo it per billboard/camera setup.

---

## Fallback matrix (when something looks wrong)

**OAK-D Lite not detected**
- check USB cable (use a USB3 data cable, not charge-only)
- use a powered USB hub
- run `lsusb` (look for Movidius / Luxonis / Intel)
- reboot the Pi

**Calibration fails with zero faces**
- move closer
- improve lighting
- make sure the face is clearly visible

**Calibration fails with multiple faces**
- clear the scene
- only one calibration person visible

**FPS is low**
- use `--face-res 320x240`
- use `--fps 8` or `--fps 10`
- disable preview (don't pass `--preview`)
- (age/gender/emotion are already off — nothing to disable)

**`total_passed` is too low**
- camera may be too far → move it closer
- face boxes too small → reduce distance / angle
- crowd is occluding faces

**`looked_total` is too high**
- the calibration cone may be too wide
- the camera may be aligned with the walking direction
- recalibrate
- prefer `looked_0_5s` / `looked_1_0s` as the stronger signal

**`looked_total` is too low**
- calibration may be wrong → recalibrate
- people are looking with their eyes, not their head (this measures head pose)
- the face/head pose isn't visible from the camera angle
- widen tolerance slightly (`--yaw-tol` / `--pitch-tol`) only **after** a controlled test

# LOOQ Field Guide — everything you need for the metro test

This is the **only document the field team needs.** It tells you what the system
measures, how to set it up, the exact steps to run, how to read the results, and
what to do when something goes wrong. You do **not** need to be a CV/ML expert.

---

## 1. What this measures (in plain words)

LOOQ counts how many people **look at the billboard**, and for how long — locally,
anonymously, on a Raspberry Pi 4 + OAK-D Lite camera. It uses **head pose** (which
way the head is turned), not faces, names, or eyes.

You will collect **five numbers** per session:

| number | meaning |
|---|---|
| `total_passed` | how many unique people (face/head tracks) were seen |
| `looked_total` | how many looked at the billboard for **≥ 0.3 s** (same as `looked_0_3s`) |
| `looked_0_3s`  | looked **≥ 0.3 s** |
| `looked_0_5s`  | looked **≥ 0.5 s**  ← the most useful "real look" number |
| `looked_1_0s`  | looked **≥ 1.0 s**  ← cleaner, but lower |

The metric is called **`LIKELY_ATTENTION`**: *a unique person whose head pointed at
the billboard direction long enough.*

**What it is NOT** (don't claim these): not eye-gaze, not identity/face recognition,
not age/gender/emotion, not demographics, not "verified reach". Track loss can
slightly **over-count** the same person — that's expected for this PoC.

### Privacy promise
Nothing personal ever leaves the device. **No** video/photos saved, **no** face
crops, **no** uploads, **no** cloud, **no** age/gender/emotion. Output is only
anonymous counters in local files. (Running with `--privacy-safe` even refuses
unsafe options instead of ignoring them.)

---

## 2. Hardware setup (physical)

- **OAK-D Lite = the camera. Raspberry Pi 4 = the computer. Do NOT use the Pi Camera.**
- Connect the OAK-D Lite with a **good USB3 cable**; use a **powered USB hub** if you have one (the Pi's own USB power is unreliable).
- Mount the camera **beside the billboard edge**, about **1.5–1.8 m** high.
- Aim it at the **faces of people walking up to the billboard** — not at the billboard itself.
- Best face distance is **2–5 m**. Pick **one** main walking direction for the first test.
- The billboard doesn't need to be fully in view; the **faces in front of it** do.
- This billboard is **1.0 m wide × 2.0 m tall** (the "3 m" figure some people quote is the *height of the top edge from the floor*, not the board size).

Install once:
```bash
python -m pip install -r requirements.txt
```

---

## 3. The test flow — run these in order

Each command saves its results to its own timestamped folder under `runs/`.
Press **Ctrl+C** to finish a live phase. Add `--print` to any command to *see* the
full command without running it. (Everything also works as
`python scripts/field_wizard.py <phase>`.)

### Step 0 — Doctor (is the rig OK?)
```bash
python main.py field doctor
```
You want `✅ OAK device detected`, privacy checks passing, output folder writable.
If the camera isn't found, follow the printed fixes (see Troubleshooting below).

### Step 1 — Preview (is the framing OK?)
```bash
python main.py field preview
```
Check: faces are clearly visible, not mostly ceiling/signs/background, people 2–5 m
away. Frames are shown **locally only** and never stored. Close the window when happy.
*(Tip: add a counting box, e.g. `python main.py field preview --counting-roi 0.15,0.20,0.85,0.95`.)*

### Step 2 — Calibrate (teach it the "looking" direction) — ONE person
```bash
python main.py field calibrate
```
One person stands where viewers stand and looks at the **centre** of the billboard
for 5 seconds. It **fails on purpose** if it sees zero or more than one face — clear
the scene and retry. On success it saves the calibration **and** a timestamped copy.

### Step 3 — Controlled test (does it roughly agree with you?)
```bash
python main.py field controlled --counting-roi 0.15,0.20,0.85,0.95 \
  --manual-total 10 --manual-lookers 5
```
Do **10 passes — 5 people look, 5 don't.** (Only 2 people? Repeat passes.) Each
person fully **leaves the frame** before the next one enters, so tracking resets.
At the end it prints **PASS / WARNING / FAIL**:
- `total_passed` near 10 → **7–13 PASS**, 5–15 WARNING.
- `looked_0_5s` near 5 → **3–7 PASS**, 2–8 WARNING.

**Only continue if this is PASS (or an acceptable WARNING).**

### Step 4 — Middle traffic (3–5 minutes, moderate flow)
```bash
python main.py field middle --counting-roi 0.15,0.20,0.85,0.95
```
Watch the screen: stable FPS, no crashes, numbers that look sane.

### Step 5 — High traffic (3–5 minutes, dense flow)
```bash
python main.py field high --counting-roi 0.15,0.20,0.85,0.95
```
This is a **stress test only** — do not claim accuracy from this alone.

### If anything breaks
```bash
python main.py field bundle
```
Creates `runs/debug_bundle_<timestamp>.zip` with anonymous text/JSON only (no images).
**Send that zip back to the CV team.**

> **The counting box (`--counting-roi`) is strongly recommended in the metro.** It's
> four numbers `x1,y1,x2,y2` from 0 to 1 (fractions of the frame). Only people whose
> head-center is inside the box are counted, which keeps background faces out. Without
> it you'll see `WARNING: no counting ROI provided`.

---

## 4. Reading the results

Every phase writes a folder like `runs/20260622_101500_controlled/` containing:
- `summary.json` — the five numbers, the PASS/WARNING/FAIL decision, privacy block, limitations.
- `events.csv` — anonymous per-track events (`timestamp,track_id,event,looking,dwell_sec,yaw_deg,pitch_deg,reason`). No age/gender/emotion.
- `command.txt`, `README.txt`, and a copy of the calibration used.

Old results are never overwritten — each run is its own timestamped folder.

**Report these five numbers** (from each `summary.json`), plus your manual counts and the PASS/WARNING/FAIL:
```
total_passed   looked_total   looked_0_3s   looked_0_5s   looked_1_0s
```

**Do NOT claim:** age, gender, emotion, identity, demographics, "verified attention",
"verified reach". The system does not produce them.

---

## 5. Troubleshooting

| Symptom | What to do |
|---|---|
| **OAK-D Lite not detected** | Use a USB3 **data** cable (not charge-only); use a powered USB hub; run `lsusb` (look for Movidius/Luxonis/Intel); reboot the Pi. |
| **Calibration fails — zero faces** | Move closer; improve lighting; make sure the face is clearly visible. |
| **Calibration fails — multiple faces** | Clear the scene; only **one** person visible during calibration. |
| **FPS is low** | Use `--face-res 320x240`; use `--fps 10` or `--fps 8`; don't run preview during a real count. |
| **`total_passed` too low** | Camera too far / face boxes too small → move closer; reduce the angle; crowd may be occluding faces; widen the counting box. |
| **`looked_total` too high** | Counting box / cone too wide, or the camera is aligned with the walking direction → recalibrate; trust `looked_0_5s` / `looked_1_0s` more. |
| **`looked_total` too low** | Calibration likely wrong → recalibrate; people may be looking with eyes not head; the head pose isn't visible from this angle. Widen tolerance (`--yaw-tol`/`--pitch-tol`) only **after** a good controlled test. |
| **Anything else / a crash** | `python main.py field bundle` and send the zip. |

---

## 6. Notes & appendix

**Don't move the camera after calibrating.** Any nudge invalidates calibration — re-run Step 2.

**No hardware yet?** You can prove the counting logic works offline (no camera):
```bash
python main.py --simulate-poses --privacy-safe \
  --calibration-file configs/example_calibration.json \
  --summary-out runs/sim_test/summary.json
```

**The long commands** (the wizard just runs these for you):
```bash
# Doctor
python main.py --doctor

# Calibrate
python main.py --privacy-safe --calibrate center --seconds 5 \
  --camera-id oak_d_lite_01 --camera-height-m 1.6 \
  --billboard-id metro_billboard_001 --billboard-width-m 1.0 --billboard-height-m 2.0 \
  --calibration-file configs/metro_billboard_calibration.json --tui

# Field run (controlled phase, with counting box)
python main.py --privacy-safe \
  --calibration-file configs/metro_billboard_calibration.json \
  --counting-roi 0.15,0.20,0.85,0.95 --test-phase controlled \
  --manual-total 10 --manual-lookers 5 \
  --tui --summary-out runs/controlled_test/summary.json

# Debug bundle
python scripts/collect_debug_bundle.py
```

**How it works under the hood** (for the curious): the OAK-D Lite detects faces
(YuNet), tracks them with stable IDs, and estimates each head's yaw/pitch. The Pi
checks whether each head points near the *calibrated billboard direction*, times how
long, and counts unique people over the thresholds. Inference runs on the camera's
chip; the counting math runs on the Pi and is verified offline by `--simulate-poses`.

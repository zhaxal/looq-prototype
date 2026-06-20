# looq-prototype — Attention Counter

OAK-D-Lite app: count how many people are looking at the camera, plus age/gender/emotion per person.

## Hardware & Deploy Target
- Camera: OAK-D-Lite (RVC2 VPU)
- Deploy: Raspberry Pi 4, headless (`opencv-python-headless`)
- Power: powered USB hub or Y-cable (Pi USB alone is unreliable)
- Dev: laptop with OAK confirmed working in oakviewer

## SDK
DepthAI v3. Do not use v2 patterns.

## Key Design Decisions
- "Looking" = **head pose** (yaw/pitch thresholds), NOT full gaze estimation — simpler and more robust
- Age/gender cached per track ID (re-run only on new faces)
- Emotion throttled periodically (heavy net, Pi 4 budget)

## Confirmed v3 Model Zoo Slugs
- Face detect: `luxonis/yunet:640x480` (confirmed zoo slugs: 320x240, 640x360, 640x480, 960x720)
- Head pose: `luxonis/head-pose-estimation:60x60`
- Age/gender: classic OpenVINO `age-gender-recognition-retail` (~62x62) — needs conversion via HubAI/ModelConverter, not yet a confirmed v3 zoo slug
- Emotion: classic OpenVINO `emotions-recognition-retail` (64x64, 5 classes) — same status

## v3 Wiring Idiom
```python
# Stage-1 detection
ParsingNeuralNetwork.build(input=, nnSource=SLUG)

# Per-face crop
FrameCropper.fromImgDetections(...).build(inputImage=)

# Sync stage-2 output to detections
GatherData.build(inputData=, inputReference=, cameraFps=)

# Tracking
ObjectTracker  # SHORT_TERM_IMAGELESS + UNIQUE_ID strategy

# Host loop — block on an output queue (NO pipeline.processTasks(); it is not
# a v3 API. Host nodes/parsers run in their own threads after start()).
pipeline.start()
while pipeline.isRunning():
    msg = some_queue.get()  # blocking; drives the loop one msg per frame

# Queues
node.out.createOutputQueue()
```

## Deployment Modes
Two ways to run, sharing `attention/config.py` constants and `attention/calc.py`:

1. **All-on-device (legacy)** — `python main.py [flags]`. Everything on the
   OAK/Pi (the original design below).
2. **Pi/server split** — the OAK VPU isn't powerful enough for 4 nets/face, so:
   - **Pi** (`pi_streamer.py`): OAK captures RGB + hardware-encodes **H.264**;
     `attention/netclient.py` ships frames to the server over WebSocket. No NN on
     the Pi.
   - **Server** (`server/`, GPU): decodes H.264 (PyAV) and runs the **whole**
     vision pipeline on the GPU via **ONNX Runtime (CUDA)** — YuNet detect →
     host IoU tracker → head pose / age-gender / emotion — then the attention
     calc (debounce/dwell/counting), CSV log, and session summary.
   - Wire format: binary WS message = `struct "<dI"` (capture_ts, seq) + H.264
     bytes. The server uses **capture_ts as `now`** so debounce/dwell are immune
     to network jitter.

## Project Layout
```
attention/          Python package (shared by both modes)
  config.py         constants, model paths, networking env vars, load_dotenv()
  calc.py           PURE calc: LookState, iou, is_looking, best_match, total_dwell
  processing.py     Pi-only NN parsers + tracklet geometry (re-exports calc)
  pipeline.py       DAI pipeline construction (legacy on-device)
  display.py        LiveDisplay (TUI) + draw_preview (OpenCV)
  netclient.py      Pi → server resilient H.264 WebSocket sender
main.py             legacy all-on-device entry  →  python main.py [flags]
pi_streamer.py      Pi streamer entry           →  python pi_streamer.py [--server URL]
server/             GPU server (no depthai)
  app.py            FastAPI: WS /ingest + GET /status,/health
  decode.py         PyAV H.264 → BGR frames
  models.py         ONNX Runtime (CUDA) sessions + pre/post for the 4 nets
  tracker.py        host IoU tracker (replaces OAK ObjectTracker)
  session.py        AttentionSession: per-track state + calc (ported from main.py)
  csvlog.py         CSV writer + session summary
  pipeline.py       orchestrates decode → detect → track → nets → calc
  selftest.py       python -m server.selftest  (smoke-test the ONNX models)
  requirements.txt
scripts/
  download_models.py        fetch RVC2 zoo models (legacy on-device)
  download_models_onnx.py   fetch ONNX models for the server → models/onnx/
  replay_to_server.py       stream a local video to the server (test w/o a Pi)
models/
  age_gender/       config.json + superblob (HubAI output)
  emotion/          config.json + superblob
  *.rvc2.tar.xz     zoo archives (yunet, head-pose)  — RVC2/VPU only
  onnx/             host-runnable ONNX (yunet, head-pose, age-gender, emotion)
```

## Build Phases
- [x] Phase 1-3: camera → YuNet → ObjectTracker → FrameCropper → head pose → prints id/yaw/pitch/LOOKING
- [x] Phase 4: debounced counter + deduplicate by track ID; `--preview` overlay; `--tui` dashboard; CSV log
- [x] Phase 5: age/gender (cache per track ID) + emotion (throttled); `--looking-gate`; `--test-video`
- [x] Phase 6: Pi/server split — Pi streams H.264; GPU server runs vision (ONNX/CUDA) + calc

## Known Risks / VERIFY Markers
Code has `# VERIFY` comments on accessor/enum names that may drift across depthai-nodes releases:
- Pose message accessor and angle order
- GatherData field names (`.gathered` / `.reference` / `.data`)
- Tracklet `.roi` accessor

Server (Pi/server split) VERIFY markers — confirm on first GPU run:
- ONNX output tensor names for head-pose (yaw/pitch/roll) and age-gender (age/prob)
  in `server/models.py` (resolved by name hint with positional fallback)
- ONNX model **sources/URLs** in `scripts/download_models_onnx.py` (head-pose &
  age-gender have no canonical ONNX — convert their OpenVINO IR or supply a URL)
- Emotion preprocessing (HSEmotion enet_b2: RGB + ImageNet normalize, 260×260)
- v3 `VideoEncoder` API (`setDefaultProfilePreset`, profile enum, `.bitstream`)
  and `ReplayVideo` methods in `pi_streamer.py`
- Run `python -m server.selftest` to validate the models before full wiring

## Docs
- Luxonis docs index: https://docs.luxonis.com/llms.txt
- v3 SDK overview: https://docs.luxonis.com/software-v3/depthai.md
- All nodes: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes.md
- Model Zoo: https://docs.luxonis.com/software-v3/ai-inference/model-source/zoo.md
- HubAI (model conversion): https://docs.luxonis.com/cloud/hubai.md
- v2→v3 porting guide: https://docs.luxonis.com/software-v3/depthai/tutorials/v2-vs-v3.md

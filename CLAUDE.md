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
- Face detect: `luxonis/yunet:640x480` (also 320x240, 640x360, 960x720, 1280x960)
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

# Host loop
pipeline.start()
while pipeline.isRunning():
    pipeline.processTasks()

# Queues
node.out.createOutputQueue()
```

## Build Phases
- [x] Phase 1-3: camera → YuNet → ObjectTracker → FrameCropper → head pose → prints id/yaw/pitch/LOOKING
- [ ] Phase 4: debounced counter + deduplicate by track ID; optional `--preview` overlay
- [ ] Phase 5: add age/gender (cache per track ID) then emotion (throttled); both need model conversion

## Known Risks / VERIFY Markers
Code has `# VERIFY` comments on accessor/enum names that may drift across depthai-nodes releases:
- Pose message accessor and angle order
- GatherData field names (`.gathered` / `.reference` / `.data`)
- Tracklet `.roi` accessor

## Docs
- Luxonis docs index: https://docs.luxonis.com/llms.txt
- v3 SDK overview: https://docs.luxonis.com/software-v3/depthai.md
- All nodes: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes.md
- Model Zoo: https://docs.luxonis.com/software-v3/ai-inference/model-source/zoo.md
- HubAI (model conversion): https://docs.luxonis.com/cloud/hubai.md
- v2→v3 porting guide: https://docs.luxonis.com/software-v3/depthai/tutorials/v2-vs-v3.md

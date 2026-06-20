"""Runtime constants and model path resolution."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"


def load_dotenv() -> None:
    """Load DEPTHAI_HUB_API_KEY (and other vars) from .env into the environment."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# --- Model sources -----------------------------------------------------------

# Zoo slugs (downloaded by scripts/download_models.py).
FACE_MODEL_SLUG = "luxonis/yunet"                    # resolution appended at runtime
POSE_MODEL_SLUG = "luxonis/head-pose-estimation:60x60"

# Local model archives — HubAI / ModelConverter output, placed in models/.
# Used via dai.NNArchive(str(path)) when present.
AGE_GENDER_ARCHIVE = MODELS_DIR / "age_gender-62x62.rvc2.tar.xz"
EMOTION_ARCHIVE    = MODELS_DIR / "enet_b2_8_best.rvc2.tar.xz"

# --- NN input sizes ----------------------------------------------------------

POSE_INPUT       = (60, 60)
AGE_GENDER_INPUT = (62, 62)
EMOTION_INPUT    = (260, 260)

# --- Pipeline ----------------------------------------------------------------

FACE_RESOLUTIONS = ("320x240", "640x360", "640x480", "960x720")

DEFAULT_FPS     = 12
FACE_CONFIDENCE = 0.80
MIN_FACE_AREA   = 0.003  # normalized bbox area; filters distant blobs (≈ 55×42 px at 640×480)
YAW_LIMIT       = 20.0
PITCH_LIMIT     = 15.0
DEBOUNCE_SECS = 0.20   # commit looking/not-looking after this many seconds stable
IOU_THRESHOLD   = 0.2
POSE_UNSEEN     = (90.0, 90.0)   # sentinel used when no pose is cached yet

# Minimum seconds between per-track age/gender updates (re-runs to correct bad initial estimates).
AGE_GENDER_INTERVAL = 2.0
# Minimum seconds between per-track emotion updates (heavy net on Pi 4 budget).
EMOTION_INTERVAL = 2.0

# --- Networking (Pi streamer → GPU server) -----------------------------------

# WebSocket endpoint on the server that ingests the Pi's H.264 stream.
LOOQ_SERVER_URL = os.environ.get("LOOQ_SERVER_URL", "ws://localhost:8000/ingest")
# Bind host/port for the server (uvicorn).
LOOQ_SERVER_HOST = os.environ.get("LOOQ_SERVER_HOST", "0.0.0.0")
LOOQ_SERVER_PORT = int(os.environ.get("LOOQ_SERVER_PORT", "8000"))
# Max encoded frames buffered on the Pi before dropping oldest (resilience).
LOOQ_SEND_QUEUE_MAX = int(os.environ.get("LOOQ_SEND_QUEUE_MAX", "30"))

# Directory holding the host-runnable ONNX models used by the server.
ONNX_MODELS_DIR = Path(os.environ.get("ONNX_MODELS_DIR", str(MODELS_DIR / "onnx")))

# H.264 encoder settings on the OAK (Pi side).
ENCODER_BITRATE_KBPS = int(os.environ.get("ENCODER_BITRATE_KBPS", "4000"))
ENCODER_KEYFRAME_FREQ = int(os.environ.get("ENCODER_KEYFRAME_FREQ", "15"))


# --- Labels ------------------------------------------------------------------

EMOTION_CLASSES = ["Anger", "Contempt", "Disgust", "Fear",
                   "Happiness", "Neutral", "Sadness", "Surprise"]
GENDER_CLASSES  = ["Female", "Male"]

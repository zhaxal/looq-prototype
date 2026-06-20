"""FastAPI server: ingest the Pi's H.264 stream, run vision + attention calc.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints:
    WS  /ingest   binary frames: struct "<dI" (capture_ts, seq) + H.264 bytes
    GET /status   latest counts + aggregates (JSON)
    GET /health   liveness
"""
from __future__ import annotations

import os
import struct
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from attention.config import load_dotenv

load_dotenv()

from .models import VisionModels       # noqa: E402 — after load_dotenv
from .pipeline import VisionPipeline   # noqa: E402

_HEADER = struct.Struct("<dI")


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes")


class AppState:
    models: VisionModels | None = None
    sessions: dict[int, VisionPipeline] = {}
    next_id: int = 1


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.models = VisionModels(
        age_gender=_flag("LOOQ_AGE_GENDER", True),
        emotion=_flag("LOOQ_EMOTION", True),
    )
    print("[server] models ready")
    yield
    state.models = None


app = FastAPI(title="looq attention server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "models_loaded": state.models is not None,
            "active_sessions": len(state.sessions)}


@app.get("/status")
async def status() -> dict:
    sessions = {sid: p.latest_counts for sid, p in state.sessions.items()}
    latest = next(reversed(state.sessions.values()), None)
    return {
        "active_sessions": len(state.sessions),
        "latest": latest.latest_counts if latest else {},
        "sessions": sessions,
    }


@app.websocket("/ingest")
async def ingest(ws: WebSocket) -> None:
    await ws.accept()
    log_path = os.environ.get("LOOQ_LOG")   # None disables; "" → auto-named file
    pipeline = VisionPipeline(state.models, log_path=log_path)
    sid = state.next_id
    state.next_id += 1
    state.sessions[sid] = pipeline
    peer = ws.client.host if ws.client else "?"
    print(f"[server] session {sid} connected from {peer}")
    try:
        while True:
            msg = await ws.receive_bytes()
            if len(msg) < _HEADER.size:
                continue
            capture_ts, _seq = _HEADER.unpack_from(msg, 0)
            pipeline.ingest(capture_ts, msg[_HEADER.size:])
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[server] session {sid} error: {exc}")
    finally:
        pipeline.finalize()
        state.sessions.pop(sid, None)
        print(f"[server] session {sid} disconnected")

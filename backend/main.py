import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.api.websocket import ws_manager
from backend.database.db import init_db


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
        if origins:
            return origins
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://127.0.0.1",
        "http://neuroevolve.local",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ws_manager.set_event_loop(asyncio.get_running_loop())
    yield


app = FastAPI(
    title="NEUROEVOLVE — Self-Evolving Neural Architecture Search Engine",
    description="Real PyTorch Neural Architecture Search powered by Genetic Evolution, Reinforcement Learning, and Pareto Multi-Objective Optimization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.websocket("/ws")
async def websocket_global(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep socket alive and allow optional client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/experiments/{exp_id}")
async def websocket_experiment(websocket: WebSocket, exp_id: str):
    await ws_manager.connect(websocket, experiment_id=exp_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, experiment_id=exp_id)
    except Exception:
        ws_manager.disconnect(websocket, experiment_id=exp_id)


@app.get("/")
def root():
    return {
        "name": "NEUROEVOLVE API",
        "docs": "/docs",
        "health": "/api/health",
    }

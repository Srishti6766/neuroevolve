import asyncio
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket


class ConnectionManager:
    """Manages global and per-experiment WebSocket connections for real-time NAS streaming."""

    def __init__(self) -> None:
        self._global_connections: Set[WebSocket] = set()
        self._exp_connections: Dict[str, Set[WebSocket]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket, experiment_id: Optional[str] = None) -> None:
        await websocket.accept()
        if experiment_id:
            self._exp_connections.setdefault(experiment_id, set()).add(websocket)
        else:
            self._global_connections.add(websocket)

    def disconnect(self, websocket: WebSocket, experiment_id: Optional[str] = None) -> None:
        if experiment_id and experiment_id in self._exp_connections:
            self._exp_connections[experiment_id].discard(websocket)
            if not self._exp_connections[experiment_id]:
                self._exp_connections.pop(experiment_id, None)
        self._global_connections.discard(websocket)

    async def broadcast(self, event: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        targets: Set[WebSocket] = set(self._global_connections)
        if experiment_id and experiment_id in self._exp_connections:
            targets |= set(self._exp_connections[experiment_id])

        dead = []
        for ws in targets:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, experiment_id)

    def broadcast_threadsafe(self, event: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(event, experiment_id), self._loop)
        except Exception:
            pass


ws_manager = ConnectionManager()

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress over WebSocket (docs/SPEC.md section 12).

The workers are separate processes and do not talk to the server: they write
their progress into the catalogue, which is the only thing all sixteen of them
share. So the server polls that one table and pushes what changed.

Polling sounds worse than it is. The query is a grouped count over an indexed
column -- microseconds -- and it runs four times a second, which is the rate
section 12 caps the messages at anyway. The alternative, a pipe from every
worker, would have to be drained by the event loop, survive a worker dying, and
still be batched at the other end. This is four lines and cannot lose a message,
because the message is a snapshot rather than an event.

Two consequences worth knowing. A client that connects halfway through gets the
current state immediately, not a replay. And a client that misses a frame misses
nothing: the next one is a complete picture.

The endpoint is ``/ws``. The service worker of section 4 must never cache it,
which is a rule the frontend enforces and test 19 checks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..db.models import JobState
from ..db.session import get_sessionmaker

__all__ = ["ProgressHub", "hub", "router"]

_log = logging.getLogger(__name__)

router = APIRouter(tags=["progress"])

#: Four a second, the cap of section 12. Also the slowest rate at which a
#: progress bar still looks alive.
_INTERVAL = 0.25


def _snapshot(project_id: int | None = None) -> dict[str, Any]:
    """Everything a progress display needs, in one query per table."""
    from sqlalchemy import func, select

    from .. import problems
    from ..db.models import Job, Photo

    maker = get_sessionmaker()
    session = maker()
    try:
        counts = {state.value: 0 for state in JobState}
        statement = select(Job.state, func.count(Job.id)).group_by(Job.state)
        if project_id is not None:
            statement = statement.where(Job.project_id == project_id)
        for state, count in session.execute(statement):
            counts[state.value if hasattr(state, "value") else str(state)] = int(count)

        running = session.scalars(
            select(Job)
            .where(Job.state == JobState.RUNNING)
            .order_by(Job.id)
            .limit(32)
        ).all()

        photos_statement = select(func.count(Photo.id))
        with_proxy = select(func.count(Photo.id)).where(Photo.proxy_path.is_not(None))
        culled = select(func.count(Photo.id)).where(Photo.culling_features.is_not(None))
        if project_id is not None:
            photos_statement = photos_statement.where(Photo.project_id == project_id)
            with_proxy = with_proxy.where(Photo.project_id == project_id)
            culled = culled.where(Photo.project_id == project_id)

        active = counts[JobState.QUEUED.value] + counts[JobState.RUNNING.value]
        total = sum(counts.values())
        return {
            "type": "progress",
            "project_id": project_id,
            "counts": counts,
            "active": active,
            "total": total,
            "progress": 1.0 if total == 0 else (total - active) / total,
            "running": [
                {
                    "id": job.id,
                    "kind": job.kind.value,
                    "progress": job.progress,
                    "photo_id": (job.payload or {}).get("photo_id"),
                }
                for job in running
            ],
            "photos": int(session.scalar(photos_statement) or 0),
            "photos_with_proxy": int(session.scalar(with_proxy) or 0),
            #: Photos whose culling analysis is done: the progress of section 7.
            "photos_analysed": int(session.scalar(culled) or 0),
            #: Section 19: "il conteggio dei falliti è sempre visibile".
            "problems": problems.count(session, project_id),
        }
    finally:
        session.close()


@dataclass
class ProgressHub:
    """Every open WebSocket, and the one task that feeds them all.

    One poll serves every client watching the same project, which is what keeps
    ten open tabs from being ten times the database traffic.
    """

    clients: dict[WebSocket, int | None] = field(default_factory=dict)
    _task: asyncio.Task | None = None
    _last: dict[int | None, dict[str, Any]] = field(default_factory=dict)

    async def connect(self, websocket: WebSocket, project_id: int | None) -> None:
        await websocket.accept()
        self.clients[websocket] = project_id
        # The state as it is now, before the first tick: a client that connects
        # to a quiet system must not wait 250 ms to learn that it is quiet.
        await websocket.send_json(await asyncio.to_thread(_snapshot, project_id))
        self._ensure_running()

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.pop(websocket, None)

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while self.clients:
            await asyncio.sleep(_INTERVAL)
            scopes = set(self.clients.values())
            payloads: dict[int | None, dict[str, Any]] = {}
            for scope in scopes:
                payloads[scope] = await asyncio.to_thread(_snapshot, scope)

            for websocket, scope in list(self.clients.items()):
                payload = payloads.get(scope)
                # Nothing changed since the last tick: say nothing. An idle
                # system should not be sending four messages a second for ever.
                if payload is None or payload == self._last.get(scope):
                    continue
                try:
                    await websocket.send_json(payload)
                except Exception:  # noqa: BLE001 - a closed socket is not an error
                    self.disconnect(websocket)
            self._last = payloads
        self._task = None


#: One per server process.
hub = ProgressHub()


@router.websocket("/ws")
async def progress_socket(websocket: WebSocket, project_id: int | None = None) -> None:
    """Live progress. ``?project_id=`` narrows it to one project."""
    await hub.connect(websocket, project_id)
    try:
        while True:
            # The client has nothing to say; this is here to notice when it goes
            # away, which is the only message that ever arrives.
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception:  # noqa: BLE001
        hub.disconnect(websocket)

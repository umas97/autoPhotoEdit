# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The program as a program: its window, its preferences, and how it stops.

Three things live here that are not about photographs.

``GET /api/window`` tells the interface what it is being displayed in. It is the
only way the UI can know that it ended up in an ordinary tab instead of the
dedicated window, which section 21.2 requires it to say out loud.

``POST /api/window/close-intent`` receives the answer to the closing dialog of
section 21.3. It arrives through ``navigator.sendBeacon`` while the window is
being torn down, so it must be cheap, must not need a response, and must not
care that the socket dies underneath it.

``POST /api/app/quit`` is "Esci dall'applicazione". Stopping the server from
inside a request is a small dance: the response has to leave first, so the
shutdown is handed to a background task and the user gets a 202.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Job, JobState, Setting
from ..launcher import current_session
from .deps import get_session

__all__ = ["CloseIntent", "close_intent", "router"]

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["app"])

#: Preference key of section 21.3, stored in ``Setting``.
ON_WINDOW_CLOSE = "on_window_close"

Decision = Literal["continue", "pause", "stop"]

#: What the user last chose in the closing dialog, for the launcher to read when
#: the window process exits. ``None`` means the window died without saying
#: anything -- a crash or a kill -- which section 21.3 defines as "continue".
_close_intent: Decision | None = None


def close_intent() -> Decision | None:
    """The decision the interface sent while closing, if it sent one."""
    return _close_intent


def set_close_intent(decision: Decision | None) -> None:
    global _close_intent
    _close_intent = decision


class CloseIntent(BaseModel):
    """The body of the closing beacon."""

    decision: Decision = "continue"
    #: "Ricorda la scelta": persist it as the default for next time.
    remember: bool = False


class WindowOut(BaseModel):
    mode: str
    browser: str | None = None
    window_pid: int | None = None
    alive: bool = False
    #: Present only when the experience is degraded, and shown as it is.
    note: str | None = None
    url: str | None = None
    #: Whether "Esci dall'applicazione" can actually stop this server.
    can_quit: bool = False
    on_window_close: Decision | None = None


class SettingIn(BaseModel):
    value: Any = Field(default=None)


def _setting(session: Session, key: str) -> Setting | None:
    return session.scalars(select(Setting).where(Setting.key == key)).first()


@router.get("/window", response_model=WindowOut)
def read_window(request: Request, session: Session = Depends(get_session)) -> WindowOut:
    """How the interface is being shown, and what it may offer because of it."""
    live = current_session()
    stored = _setting(session, ON_WINDOW_CLOSE)
    remembered = stored.value if stored and stored.value in ("continue", "pause", "stop") else None
    return WindowOut(
        **live.as_dict(),  # type: ignore[arg-type]
        can_quit=callable(getattr(request.app.state, "shutdown_hook", None)),
        on_window_close=remembered,  # type: ignore[arg-type]
    )


@router.post("/window/close-intent", status_code=status.HTTP_204_NO_CONTENT)
def receive_close_intent(
    payload: CloseIntent,
    request: Request,
    session: Session = Depends(get_session),
) -> None:
    """Record what to do with the running jobs, then act on it (section 21.3).

    Called from ``navigator.sendBeacon``, so it answers 204 and never anything
    the caller could read: by the time this returns, the window is gone.
    """
    set_close_intent(payload.decision)
    if payload.remember:
        stored = _setting(session, ON_WINDOW_CLOSE)
        if stored is None:
            session.add(Setting(key=ON_WINDOW_CLOSE, value=payload.decision))
        else:
            stored.value = payload.decision

    pool = getattr(request.app.state, "pool", None)
    if payload.decision == "pause" and pool is not None and pool.running:
        # The queue is untouched: the jobs stay queued and the running ones go
        # back on it when the workers stop. Resuming is starting the pool again.
        _log.info("pausa richiesta alla chiusura della finestra")
        pool.stop()
    elif payload.decision == "stop":
        cancelled = session.execute(select(Job).where(Job.state == JobState.QUEUED)).scalars().all()
        for job in cancelled:
            job.state = JobState.CANCELLED
        _log.info("interruzione richiesta: %d job annullati", len(cancelled))


@router.get("/jobs/active", response_model=dict)
def active_jobs(session: Session = Depends(get_session)) -> dict[str, int]:
    """How many jobs are still moving. The closing dialog asks before it opens."""
    from sqlalchemy import func

    count = int(
        session.scalar(
            select(func.count(Job.id)).where(Job.state.in_((JobState.QUEUED, JobState.RUNNING)))
        )
        or 0
    )
    return {"active": count}


@router.post("/window/open", response_model=dict)
def open_window_here(request: Request) -> dict[str, object]:
    """Open a window on this server (section 21.1).

    This is how a second launch gets a window when the first one lost hers --
    closed in tab mode, or started with ``--no-window``. The window belongs to
    the process that is already serving, not to the command that asked for it,
    so closing it still stops the program that owns it.

    Raises:
        HTTPException 409: this server has no way to open a window, which means
            it was not started by the launcher.
    """
    hook = getattr(request.app.state, "open_window_hook", None)
    if not callable(hook):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "questo server non è stato avviato dal launcher e non può aprire finestre",
        )
    return dict(hook())


@router.post("/app/pause", response_model=dict)
def pause_workers(request: Request) -> dict[str, bool]:
    """Stop the worker pool without touching the queue."""
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "questo server non ha worker")
    if pool.running:
        pool.stop()
    return {"running": bool(pool.running)}


@router.post("/app/resume", response_model=dict)
def resume_workers(request: Request) -> dict[str, bool]:
    """Start the pool again. Interrupted jobs are requeued by the pool itself."""
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "questo server non ha worker")
    if not pool.running:
        pool.start()
    return {"running": bool(pool.running)}


@router.post("/app/quit", status_code=status.HTTP_202_ACCEPTED)
def quit_app(request: Request, background: BackgroundTasks) -> dict[str, str]:
    """Close the window, stop the workers, stop the server (section 21.3).

    Raises:
        HTTPException 409: this server was not started by the launcher, so there
            is nothing to hand the shutdown to -- a development ``uvicorn``, or
            the test client. Better a refusal than a process that will not die.
    """
    hook = getattr(request.app.state, "shutdown_hook", None)
    if not callable(hook):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "questo server non è stato avviato dal launcher e non può spegnersi da solo",
        )
    background.add_task(hook)
    return {"detail": "chiusura in corso"}


@router.get("/settings", response_model=dict)
def read_settings(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Every global preference, as one object."""
    return {row.key: row.value for row in session.scalars(select(Setting)).all()}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError("un testo di al massimo 500 caratteri")
    return value.strip() or None


def _quota(value: Any) -> float:
    # Section 20.3: 20 GB by default. Under 1 GB the proxies of one project
    # would not fit, and an eviction at every job is not a cache.
    if not isinstance(value, int | float) or isinstance(value, bool) or not 1 <= value <= 10_000:
        raise ValueError("un numero di GB fra 1 e 10000")
    return float(value)


def _close(value: Any) -> str | None:
    if value not in (None, "continue", "pause", "stop"):
        raise ValueError("continue, pause, stop o nessuna scelta")
    return value


#: The preferences a screen may write, each with what it accepts. Everything
#: else in the table -- the schema version first of all -- is the program's own.
_WRITABLE = {"artist": _text, "copyright": _text, "cache_max_gb": _quota, ON_WINDOW_CLOSE: _close}


@router.put("/settings/{key}", response_model=dict)
def write_setting(
    key: str, payload: SettingIn, session: Session = Depends(get_session)
) -> dict[str, Any]:
    check = _WRITABLE.get(key)
    if check is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"impostazione sconosciuta: {key}")
    try:
        payload = SettingIn(value=check(payload.value))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{key}: serve {exc}") from exc
    stored = _setting(session, key)
    if stored is None:
        session.add(Setting(key=key, value=payload.value))
    else:
        stored.value = payload.value
    session.flush()
    return {key: payload.value}

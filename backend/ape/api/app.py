# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The local server: routes, lifespan, and the worker pool it owns.

The server is not a service. It is the back half of a desktop program that
happens to speak HTTP to its own window, and everything here follows from that:
it binds ``127.0.0.1`` and has no option not to (section 21.1), it has no
authentication because there is no one else on the socket, and it starts and
stops the worker pool with itself. The other pages open in the user's browser
*can* reach a loopback socket; ``local_only`` turns them away.

The pool is started in the lifespan rather than lazily so that the jobs a
previous run left in the queue start moving the moment the program is up --
which is half of what test 6 asks for; the other half, putting crashed jobs back
in the queue, happens in ``init_db`` just before.

Errors become JSON with a message in Italian. The interface never shows a
traceback: section 19 gives the user a Problems panel and a diagnostics bundle,
and a stack trace in a dialog is neither.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from .. import __version__
from ..config import get_settings
from ..db.session import init_db
from ..jobs.pool import WorkerPool
from ..safety import SourceWriteError
from ..window_id import WINDOW_PATH
from . import (
    routes_analysis,
    routes_app,
    routes_culling,
    routes_export,
    routes_folders,
    routes_jobs,
    routes_masks,
    routes_merges,
    routes_photos,
    routes_problems,
    routes_projects,
    routes_retouch,
    routes_review,
    routes_snapshots,
    routes_storage,
    routes_styles,
    ws,
)
from .local_only import LocalOnlyMiddleware
from .spa import mount_frontend

__all__ = ["create_app"]

_log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_db()
    from ..analysis.service import retire_stale_proposals
    from ..db.session import session_scope
    from ..style.profile import ensure_builtins

    with session_scope() as session:
        ensure_builtins(session)
        retire_stale_proposals(session)

    pool: WorkerPool | None = None
    if app.state.start_workers:
        pool = WorkerPool(db_path=settings.db_path, workers=app.state.worker_count)
        pool.start()
    app.state.pool = pool

    try:
        yield
    finally:
        await ws.hub.stop()
        if pool is not None:
            pool.stop()
        # Whatever started this process gets its last word here, and only here.
        # ``uvicorn`` re-raises the signal that stopped it once its own handlers
        # are back in place, so a ``finally`` around ``server.run()`` never runs
        # on a SIGTERM -- and the window would outlive the program that opened
        # it, with a lockfile pointing at a PID that is gone (section 21.3).
        teardown = getattr(app.state, "teardown", None)
        if callable(teardown):
            teardown()


def create_app(*, start_workers: bool = True, worker_count: int = 0) -> FastAPI:
    """Build the application.

    Args:
        start_workers: start the process pool with the server. Off in tests,
            which drive the workers themselves so that a test never races a
            background process.
        worker_count: override the count of section 12. Zero means the rule.

    Returns:
        A configured :class:`fastapi.FastAPI`.
    """
    app = FastAPI(
        title="autoPhotoEdit",
        version=__version__,
        summary="Post-produzione automatica di RAW Sony, in locale e non distruttiva",
        lifespan=_lifespan,
    )
    app.state.start_workers = start_workers
    app.state.worker_count = worker_count
    # Set by the launcher when it owns the process (section 21.3). Without it
    # "Esci dall'applicazione" refuses rather than half-stopping the program.
    app.state.shutdown_hook = None
    #: Called once, on the way out, by the lifespan above. The launcher uses it
    #: to close the window and release the lockfile.
    app.state.teardown = None
    app.add_middleware(LocalOnlyMiddleware)

    app.include_router(routes_projects.router)
    app.include_router(routes_folders.router)
    app.include_router(routes_photos.router)
    app.include_router(routes_culling.router)
    app.include_router(routes_analysis.router)
    app.include_router(routes_styles.router)
    app.include_router(routes_review.router)
    app.include_router(routes_export.router)
    app.include_router(routes_masks.router)
    app.include_router(routes_retouch.router)
    app.include_router(routes_merges.router)
    app.include_router(routes_snapshots.router)
    app.include_router(routes_problems.router)
    app.include_router(routes_storage.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_app.router)
    app.include_router(ws.router)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, Any]:
        """Liveness, and the numbers the launcher of section 21 checks."""
        settings = get_settings()
        pool = getattr(app.state, "pool", None)
        return {
            "status": "ok",
            "version": __version__,
            "database": str(settings.db_path),
            "workers": pool.workers if pool else 0,
            "workers_running": bool(pool and pool.running),
        }

    @app.get(WINDOW_PATH, include_in_schema=False)
    def window_entry() -> RedirectResponse:
        """Where the window opens (``launcher.WINDOW_PATH``): on to the interface.

        The path only gives the window an identity of its own in the dock;
        Chromium fixes it when the window opens, so the redirect keeps it.
        """
        return RedirectResponse("/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    # Last, deliberately: the interface is a catch-all route and would shadow
    # every API path registered after it.
    app.state.frontend_mounted = mount_frontend(app)

    @app.exception_handler(SourceWriteError)
    async def _source_write(_request: Request, exc: SourceWriteError) -> JSONResponse:
        """A refused write to the source folder is a 409, and it is logged loudly.

        Reaching this handler means a code path tried to write where section 2
        forbids it. The guard did its job, but the attempt is a defect and the
        log line is how it gets found.
        """
        _log.error("scrittura rifiutata nella cartella sorgente: %s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app

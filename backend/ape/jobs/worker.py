# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The worker pool: processes that take jobs off the queue and run them.

Shape, from docs/SPEC.md section 12: ``max(2, cpu_count - 2)`` processes, each with
BLAS and OpenCV pinned to a single thread. Pinning is not a detail. NumPy and
OpenCV both parallelise internally, and sixteen processes each starting sixteen
threads on sixteen cores is two hundred and fifty-six runnable threads fighting
over the caches -- measurably slower than doing nothing about it.

Processes rather than threads because the work is CPU-bound in Python, and
``forkserver`` rather than ``fork`` because the server that starts this pool has
threads: uvicorn's, OpenCV's, SQLAlchemy's pool. A child forked from a threaded
process inherits their locks without inheriting the threads that would release
them, and the first allocation inside NumPy hangs on a mutex nobody will ever
unlock. The forkserver is a small, thread-free process started before any of
that, which forks the workers on request -- as cheap as ``fork`` after the first
one, and safe. The modules a worker needs are preloaded into it, so starting a
worker does not re-import NumPy, OpenCV and rawpy.

Each worker builds its own SQLite engine: connections are the one thing that
must never be shared across a process boundary, forked or not.

A worker is a loop, not a callback: claim, run, record, repeat. That is what
makes a crash survivable -- the job it was holding is a row in state ``running``
with a PID that no longer exists, and the next startup puts it back in the
queue. No job is ever lost, and none is run twice, which is test 6.

Failures never kill the loop. A RAW that LibRaw refuses, a file that disappeared
between the import and now, a worker that runs out of memory on one photo: the
job is marked failed with a message a person can read, and the pool moves on
(test 11).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..db.models import JobKind, JobState
from . import queue as jobq
from . import wake
from .limits import release_idle_models, trim_heap

__all__ = [
    "HANDLERS",
    "default_worker_count",
    "pool_worker",
    "register_handler",
    "run_job",
    "worker_loop",
]

_log = logging.getLogger(__name__)

#: How long a worker without a doorbell (``wake.py``) waits before asking for
#: work again: a test or a script driving ``worker_loop`` directly. The pool's
#: workers never poll -- fourteen of them at this interval were 25% of a core.
_IDLE_SLEEP = 0.15

#: Slice of one job after which progress is written back. Writing per photo
#: would have sixteen processes committing constantly; per batch is what
#: section 12's "at most four messages a second" needs upstream of it.
_PROGRESS_EVERY = 0.5


#: Preloaded into the forkserver. One module, which pins the threads *and then*
#: imports the heavy libraries -- the order is the whole point, and importing
#: numpy or rawpy directly here would undo it. See ``_preload.py``.
_PRELOAD = ("ape.jobs._preload",)


def _pool_context() -> Any:
    """A multiprocessing context safe to start workers from a threaded server.

    ``forkserver`` where the platform has it, ``spawn`` otherwise. Never
    ``fork``: see the module docstring.
    """
    methods = mp.get_all_start_methods()
    if "forkserver" in methods:
        context = mp.get_context("forkserver")
        # Preloading is an optimisation; a platform that will not do it still
        # gets safe workers, just slower ones.
        with contextlib.suppress(AttributeError, ValueError):  # pragma: no cover
            context.set_forkserver_preload(list(_PRELOAD))
        return context
    return mp.get_context("spawn")


def default_worker_count() -> int:
    """``max(2, cpu_count - 2)``, the rule of section 12."""
    return max(2, (os.cpu_count() or 4) - 2)


def _pin_threads() -> None:
    """One thread per library, per process.

    The forkserver has already done this before importing anything (see
    ``_preload.py``); this covers the ``spawn`` fallback, where the worker
    process starts empty and nothing heavy has been imported yet -- which is
    exactly the window in which setting these still works.
    """
    from ._preload import THREAD_VARIABLES

    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"


#: ``kind -> handler``. A handler takes the claimed job and a progress callback
#: and either returns (success) or raises (failure, with the message shown to
#: the user). Registered here rather than imported so that a phase can add its
#: own without this module knowing about it.
HANDLERS: dict[JobKind, Callable[[jobq.JobRecord, Callable[[float], None]], Any]] = {}


def register_handler(
    kind: JobKind,
) -> Callable[
    [Callable[[jobq.JobRecord, Callable[[float], None]], Any]],
    Callable[[jobq.JobRecord, Callable[[float], None]], Any],
]:
    def decorate(
        function: Callable[[jobq.JobRecord, Callable[[float], None]], Any],
    ) -> Callable[[jobq.JobRecord, Callable[[float], None]], Any]:
        HANDLERS[kind] = function
        return function

    return decorate


def _load_handlers() -> None:
    """Import the modules that register handlers. Idempotent."""
    from . import (  # noqa: F401
        handlers,
        handlers_analysis,
        handlers_export,
        handlers_masks,
        handlers_merge,
        handlers_retouch,
        handlers_review,
        handlers_style,
    )


def run_job(record: jobq.JobRecord, progress: Callable[[float], None]) -> Any:
    """Dispatch one claimed job to its handler.

    Raises:
        KeyError: if nothing handles this kind, which is a programming error
            rather than a user-facing failure -- but it still lands in the job's
            error message rather than taking the worker down.
    """
    _load_handlers()
    handler = HANDLERS.get(record.kind)
    if handler is None:
        raise KeyError(f"nessun gestore per i job di tipo {record.kind.value}")
    return handler(record, progress)


def pool_worker(*args: Any, **kwargs: Any) -> None:
    """A worker of the server's pool: the log file first (section 19), then the loop."""
    from ..logs import configure_worker

    configure_worker()
    worker_loop(*args, **kwargs)


def worker_loop(
    db_path: str | Path,
    stop: Any,
    *,
    kinds: Sequence[JobKind] | None = None,
    idle_exit: bool = False,
    doorbell: wake.Doorbell | None = None,
    limits: dict[JobKind, int] | None = None,
) -> None:
    """Claim and run jobs until ``stop`` is set (or the queue empties).

    Args:
        db_path: catalogue to work against. Passed explicitly because a forked
            child must build its own engine rather than inherit one.
        stop: a :class:`multiprocessing.Event`, or anything with ``is_set()``.
        kinds: restrict this worker to some job kinds. Used to keep a worker
            free for interactive previews while a batch runs.
        idle_exit: return as soon as the queue is empty. For tests and for the
            one-shot CLI, never for the server's pool.
        doorbell: the pool's bell, rung when a job becomes claimable. Without
            one the worker polls every ``_IDLE_SLEEP``.
        limits: at most this many jobs of a kind running at once, pool-wide
            (``queue.claim_job``).
    """
    _pin_threads()

    from sqlalchemy.orm import sessionmaker

    from ..db.session import engine_for, session_scope

    engine = engine_for(db_path)
    maker = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    pid = os.getpid()
    # The forkserver is the parent, and it leaves with the server. A worker
    # whose parent changed is an orphan of a server that was killed outright.
    parent = os.getppid()
    wake.install(doorbell)

    # The pool stops workers with SIGTERM; finishing the job in hand and then
    # leaving is the difference between a clean stop and a half-written proxy.
    interrupted = {"flag": False}

    def _handle_term(_signum: int, _frame: Any) -> None:
        interrupted["flag"] = True

    # Not the main thread (in tests, and in the API's own executor): there is
    # no signal to install, and there is nothing to do about it.
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _handle_term)

    # Trim once per stretch of idleness, not on every empty poll.
    worked_since_trim = False
    try:
        while not (stop.is_set() or interrupted["flag"]):
            with session_scope(maker) as session:
                record = jobq.claim_job(session, kinds, worker_pid=pid, limits=limits)

            if record is None:
                if idle_exit:
                    return
                if release_idle_models() or worked_since_trim:
                    trim_heap()
                    worked_since_trim = False
                if doorbell is None:
                    time.sleep(_IDLE_SLEEP)
                elif not doorbell.wait() and os.getppid() != parent:
                    return
                continue

            _run_one(maker, record)
            worked_since_trim = True
            if limits and record.kind in limits and doorbell is not None:
                # A capped job just finished: a worker that found only capped
                # work may be asleep waiting for exactly this slot.
                doorbell.ring()
    finally:
        engine.dispose()


#: The jobs that write into the cache, after which its quota is checked (section 20.3).
_FILLS_CACHE = frozenset({JobKind.PROXY, JobKind.CULL, JobKind.RENDER_PREVIEW})


def _run_one(maker: Any, record: jobq.JobRecord) -> None:
    """Run a claimed job and record its outcome. Never raises."""
    from ..db.session import session_scope

    last_write = 0.0

    def report(fraction: float) -> None:
        nonlocal last_write
        now = time.monotonic()
        if now - last_write < _PROGRESS_EVERY:
            return
        last_write = now
        with session_scope(maker) as session:
            jobq.update_progress(session, record.id, fraction)

    try:
        run_job(record, report)
    except Exception as exc:  # noqa: BLE001 - a worker must survive anything
        message = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "job %d (%s) fallito: %s", record.id, record.kind.value, message, exc_info=True
        )
        with session_scope(maker) as session:
            state = jobq.fail_job(session, record.id, message, details=traceback.format_exc())
        if state is JobState.QUEUED:
            _log.info("job %d rimesso in coda (tentativo %d)", record.id, record.attempts)
    else:
        with session_scope(maker) as session:
            jobq.complete_job(session, record.id)
        if record.kind in _FILLS_CACHE:
            from ..cache import maybe_enforce

            with session_scope(maker) as session:
                maybe_enforce(session)

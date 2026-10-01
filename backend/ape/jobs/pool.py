# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The server's pool of worker processes (section 12): started and stopped as one.

The loop each process runs is ``worker.py``; this module owns the processes,
the doorbell that wakes them and the cap on concurrent exports.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..db.models import JobKind, JobState
from . import queue as jobq
from . import wake
from .limits import default_export_limit
from .worker import _pool_context, default_worker_count, pool_worker

__all__ = ["WorkerPool"]

_log = logging.getLogger(__name__)

#: Full-resolution merges at once. A panorama's composition and a stack's
#: tiles are budgeted as the one large job of a worker (section 26), and they
#: already use two threads each (``merge/rebuild.py``).
MERGE_LIMIT = 1


@dataclass
class WorkerPool:
    """A group of worker processes, started and stopped as one.

    Owned by the server. Starting it twice is a no-op; stopping it waits for the
    jobs in flight rather than killing them, because a job killed mid-write is
    how a truncated proxy gets into the cache.
    """

    db_path: Path
    workers: int = 0
    kinds: Sequence[JobKind] | None = None
    #: Full-resolution exports allowed at once; 0 computes it from the memory.
    export_limit: int = 0
    _processes: list[mp.Process] = None  # type: ignore[assignment]
    _stop: Any = None
    _doorbell: wake.Doorbell | None = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.workers = self.workers or default_worker_count()
        from ..config import get_settings

        self.export_limit = (
            self.export_limit
            or get_settings().export_concurrency
            or default_export_limit(self.workers)
        )
        self._processes = []

    @property
    def running(self) -> bool:
        return any(p.is_alive() for p in self._processes)

    def start(self) -> None:
        if self.running:
            return
        context = _pool_context()
        self._stop = context.Event()
        self._doorbell = wake.Doorbell.create(context, self.workers)
        # Jobs enqueued by the server itself ring the same bell.
        wake.install(self._doorbell)
        self._processes = []
        for index in range(self.workers):
            process = context.Process(
                target=pool_worker,
                args=(str(self.db_path), self._stop),
                kwargs={
                    "kinds": self.kinds,
                    "doorbell": self._doorbell,
                    "limits": {JobKind.EXPORT: self.export_limit, JobKind.MERGE: MERGE_LIMIT},
                },
                name=f"ape-worker-{index}",
                daemon=True,
            )
            process.start()
            self._processes.append(process)
        _log.info(
            "pool avviato: %d worker, al massimo %d export insieme",
            self.workers,
            self.export_limit,
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Ask every worker to finish what it is doing and exit."""
        if self._stop is not None:
            self._stop.set()
        if self._doorbell is not None:
            self._doorbell.ring()  # the idle ones are asleep on it
        deadline = time.monotonic() + timeout
        for process in self._processes:
            remaining = max(0.1, deadline - time.monotonic())
            process.join(remaining)
        for process in self._processes:
            if process.is_alive():
                _log.warning("worker %s non si è fermato: lo termino", process.name)
                process.terminate()
                process.join(5.0)
        self._processes = []
        # Whatever was in flight when a worker had to be killed goes back into
        # the queue, exactly as after a crash.
        self._requeue_orphans()

    def _requeue_orphans(self) -> None:
        from sqlalchemy.orm import sessionmaker

        from ..db.session import engine_for, session_scope

        engine = engine_for(self.db_path)
        maker = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        try:
            with session_scope(maker) as session:
                requeued = jobq.requeue_running(session)
            if requeued:
                _log.info("%d job rimessi in coda alla chiusura del pool", requeued)
        finally:
            engine.dispose()

    def drain(self, timeout: float = 600.0, poll: float = 0.1) -> bool:
        """Wait until no job is queued or running. Returns False on timeout.

        For the CLI and for tests: the server never blocks on this.
        """
        from sqlalchemy.orm import sessionmaker

        from ..db.session import engine_for, session_scope

        engine = engine_for(self.db_path)
        maker = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                with session_scope(maker) as session:
                    counts = jobq.counts_by_state(session)
                if counts[JobState.QUEUED.value] == 0 and counts[JobState.RUNNING.value] == 0:
                    return True
                time.sleep(poll)
            return False
        finally:
            engine.dispose()

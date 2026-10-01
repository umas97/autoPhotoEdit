# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A worker process the resume test can kill.

Test 6 of docs/SPEC.md section 13 says to kill the process halfway through a batch
and restart, so the test needs a real process with a real queue -- not a thread
it can politely ask to stop. This module is that process: it registers one
handler, whose only job is to leave a trace on disk that outlives it, and then
runs the ordinary ``worker_loop``.

The trace is one file per unit of work, appended to. A unit that ran twice
therefore has two lines, which is precisely what "resumes without duplicating"
forbids, and a unit that never ran has no file at all, which is what "without
losing" forbids. The assertions in the test are counts of those lines.

Run as::

    python -m jobrunner <db_path> [--idle-exit]
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from ape.db.models import JobKind
from ape.jobs.queue import JobRecord
from ape.jobs.worker import _load_handlers, register_handler, worker_loop

__all__ = ["record_unit", "run"]

# The production handlers first, so that the one below replaces theirs for the
# kind it borrows rather than being replaced by it at the first job: the queue
# dispatches on kind, and every kind is meant to have a real handler sooner or
# later (``analyze`` got one in phase 5 and this test broke silently).
_load_handlers()


@register_handler(JobKind.ANALYZE)
def record_unit(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Sleep a little, then leave a line saying this unit ran.

    The sleep is what gives the test a window in which to kill the process while
    a job is genuinely in flight; without it the batch would be over before the
    signal arrived.
    """
    payload = record.payload
    marker_dir = Path(payload["marker_dir"])
    unit = int(payload["unit"])
    duration = float(payload.get("sleep", 0.2))

    progress(0.1)
    time.sleep(duration)
    if payload.get("fail_once") and record.attempts == 1:
        raise RuntimeError("guasto simulato al primo tentativo")

    marker_dir.mkdir(parents=True, exist_ok=True)
    with open(marker_dir / f"{unit:03d}.done", "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} tentativo {record.attempts}\n")
    progress(1.0)


class _StopNever:
    @staticmethod
    def is_set() -> bool:
        return False


def run(db_path: str, idle_exit: bool = False) -> None:
    worker_loop(db_path, _StopNever(), idle_exit=idle_exit)


if __name__ == "__main__":
    run(sys.argv[1], idle_exit="--idle-exit" in sys.argv)

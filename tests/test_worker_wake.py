# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Idle workers sleep, and wake when there is work (section 26, ``jobs/wake.py``).

"A idle il programma consuma ~0% di CPU: nessun polling." Workers that asked
the queue every 150 ms were 25% of a core with fourteen of them, for as long as
the server ran. These tests hold the two halves of the replacement: a worker
with nothing to do costs nothing, and a job enqueued while it sleeps still
starts at once rather than at the next timeout.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

from ape.db.models import Job, JobKind, JobState
from ape.jobs import wake
from ape.jobs.queue import enqueue


@pytest.fixture
def bell():
    doorbell = wake.Doorbell.create(mp.get_context("spawn"), 3)
    wake.install(doorbell)
    yield doorbell
    wake.install(None)


def test_a_ring_before_the_wait_is_not_lost(bell):
    bell.ring()
    started = time.perf_counter()
    assert bell.wait(timeout=5.0)
    assert time.perf_counter() - started < 0.5


def test_rings_are_topped_up_not_piled_up(bell):
    for _ in range(1000):
        bell.ring()
    woken = 0
    while bell.wait(timeout=0.01):
        woken += 1
    # One permit per worker at most: a burst of rings while every worker is
    # busy must not turn into a thousand needless looks at the queue.
    assert woken == 3


def test_an_unrung_bell_times_out(bell):
    assert not bell.wait(timeout=0.05)


def test_the_bell_rings_on_commit_not_before(catalog, bell):
    with catalog() as session:
        enqueue(session, JobKind.PROXY, {"photo_id": 1})
        enqueue(session, JobKind.PROXY, {"photo_id": 2})
        # Not yet: a worker woken now would find nothing and go back to sleep.
        assert not bell.wait(timeout=0.01)
        session.commit()
    assert bell.wait(timeout=0.01)


def _cpu_seconds(pids: list[int]) -> float:
    ticks = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for pid in pids:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        total += (int(fields[11]) + int(fields[12])) / ticks
    return total


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="legge /proc")
def test_an_idle_pool_sleeps_and_wakes_at_once(catalog):
    from ape.config import get_settings
    from ape.jobs.pool import WorkerPool

    pool = WorkerPool(db_path=get_settings().db_path, workers=3)
    pool.start()
    try:
        pids = [process.pid for process in pool._processes]
        time.sleep(2.0)  # past start-up and the first look at an empty queue
        before = _cpu_seconds(pids)
        time.sleep(3.0)
        idle = _cpu_seconds(pids) - before
        # Polling at 150 ms was about 2% of a core per worker; asleep on the
        # bell it is nothing the kernel can count in 3 s.
        assert idle < 0.03, f"{idle:.3f} s di CPU in 3 s di inattività"

        # A job for a photo that does not exist: it fails at once, which is all
        # this needs -- proof that a sleeping worker picked it up.
        with catalog() as session:
            job = enqueue(session, JobKind.PROXY, {"photo_id": 999_999})
            session.commit()
            job_id = job.id
        deadline = time.monotonic() + 10.0
        state = JobState.QUEUED
        while time.monotonic() < deadline and state is not JobState.FAILED:
            time.sleep(0.05)
            with catalog() as session:
                state = session.get(Job, job_id).state
        assert state is JobState.FAILED, state
        # Well inside the one-minute fallback of the bell.
        assert deadline - time.monotonic() > 5.0
    finally:
        pool.stop()
        wake.install(None)

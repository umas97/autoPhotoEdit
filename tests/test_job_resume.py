# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 6 of docs/SPEC.md section 13: a killed batch resumes.

"Kill the process halfway through a batch, restart, the batch resumes without
duplicating or losing photos."

The kill is a real ``SIGKILL`` to real worker processes, because the failure
mode being tested only exists in real processes: a job in state ``running``
whose worker no longer exists. Nothing in Python notices that on its own -- it
is noticed at the next startup, which is where ``init_db`` puts those rows back
into the queue.

Each unit of work leaves a line in a file when it completes (see
``jobrunner.py``). Afterwards the assertions are arithmetic: every unit has a
file, and no file has two lines.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import select

from ape.db.models import Job, JobKind, JobState
from ape.db.session import get_engine, init_db
from ape.jobs.queue import claim_job, enqueue, requeue_running

UNITS = 12
UNIT_SECONDS = 0.25


def _enqueue_batch(catalog, marker_dir: Path, *, units: int = UNITS, **extra) -> None:
    with catalog() as session:
        for unit in range(units):
            enqueue(
                session,
                JobKind.ANALYZE,
                {
                    "marker_dir": str(marker_dir),
                    "unit": unit,
                    "sleep": UNIT_SECONDS,
                    **extra,
                },
                dedupe_key=f"unit:{unit}",
            )
        session.commit()


def _spawn_workers(db_path: Path, count: int, *, idle_exit: bool = False) -> list:
    environment = dict(os.environ)
    here = Path(__file__).parent
    backend = here.parent / "backend"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(here), str(backend), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    command = [sys.executable, "-m", "jobrunner", str(db_path)]
    if idle_exit:
        command.append("--idle-exit")
    return [
        subprocess.Popen(command, env=environment, cwd=str(here)) for _ in range(count)
    ]


def _markers(marker_dir: Path) -> dict[int, list[str]]:
    return {
        int(path.stem): path.read_text(encoding="utf-8").strip().splitlines()
        for path in sorted(marker_dir.glob("*.done"))
    }


def _wait_for(predicate, timeout: float = 60.0, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


@pytest.mark.slow
def test_a_killed_batch_resumes_without_losing_or_repeating_work(catalog, tmp_path: Path):
    from ape.config import get_settings

    db_path = get_settings().db_path
    markers = tmp_path / "markers"
    _enqueue_batch(catalog, markers)

    workers = _spawn_workers(db_path, 3)
    try:
        # Halfway is enough: some done, some in flight, some still queued.
        started = _wait_for(lambda: len(_markers(markers)) >= 3, timeout=60.0)
        assert started, "nessun job è partito: il runner non sta lavorando"
        done_before = len(_markers(markers))
        assert done_before < UNITS, "il batch è finito prima di poterlo interrompere"
    finally:
        for worker in workers:
            os.kill(worker.pid, signal.SIGKILL)
        for worker in workers:
            worker.wait(timeout=10)

    # Some job was in flight when the workers died: its row says "running" and
    # the process that claimed it is gone.
    get_engine().dispose()
    with catalog() as session:
        stranded = session.scalars(
            select(Job).where(Job.state == JobState.RUNNING)
        ).all()
        assert stranded, "nessun job era in corso alla morte dei worker"
        # What the catalogue already calls done must never run again. A job in
        # flight may have finished its work and died before its commit: running
        # it again is the only safe answer (at least once), as for every real
        # handler, which writes by content name and so repeats harmlessly.
        done_at_kill = {
            job.payload["unit"]
            for job in session.scalars(select(Job).where(Job.state == JobState.DONE))
        }
        in_flight = {job.payload["unit"] for job in stranded}

    # Restart. This is the whole test: startup puts the stranded rows back.
    init_db()
    with catalog() as session:
        still_running = session.scalars(select(Job).where(Job.state == JobState.RUNNING)).all()
        assert not still_running, "un job è rimasto 'running' dopo il riavvio"

    workers = _spawn_workers(db_path, 3, idle_exit=True)
    for worker in workers:
        worker.wait(timeout=120)

    found = _markers(markers)
    assert sorted(found) == list(range(UNITS)), "unità perdute alla ripresa"
    repeated = {unit: lines for unit, lines in found.items() if len(lines) > 1}
    assert not set(repeated) & done_at_kill, f"unità già fatte rieseguite: {repeated}"
    assert set(repeated) <= in_flight, f"unità eseguite più di una volta: {repeated}"

    with catalog() as session:
        jobs = session.scalars(select(Job).where(Job.kind == JobKind.ANALYZE)).all()
        assert len(jobs) == UNITS, "la ripresa ha duplicato delle righe in coda"
        assert all(job.state is JobState.DONE for job in jobs)


def test_enqueueing_the_same_work_twice_adds_one_job(catalog):
    """The dedupe key names the work, so the UI may ask as often as it likes."""
    with catalog() as session:
        first = enqueue(session, JobKind.PROXY, {"photo_id": 1}, dedupe_key="proxy:1")
        second = enqueue(session, JobKind.PROXY, {"photo_id": 1}, dedupe_key="proxy:1")
        session.commit()
        assert first is not None
        assert second is None
        assert len(session.scalars(select(Job)).all()) == 1


def test_a_finished_job_does_not_block_the_same_work_later(catalog):
    """Dedupe covers *live* jobs only: a proxy asked for again must be rebuilt."""
    with catalog() as session:
        enqueue(session, JobKind.PROXY, {"photo_id": 1}, dedupe_key="proxy:1")
        session.commit()
    with catalog() as session:
        record = claim_job(session)
        assert record is not None
        from ape.jobs.queue import complete_job

        complete_job(session, record.id)
        session.commit()
    with catalog() as session:
        again = enqueue(session, JobKind.PROXY, {"photo_id": 1}, dedupe_key="proxy:1")
        session.commit()
        assert again is not None


def test_two_claims_never_get_the_same_job(catalog):
    """The claim is one statement; the second claimant must see an empty queue."""
    with catalog() as session:
        enqueue(session, JobKind.ANALYZE, {"unit": 0}, dedupe_key="solo")
        session.commit()

    with catalog() as first_session, catalog() as second_session:
        first = claim_job(first_session, worker_pid=1001)
        second = claim_job(second_session, worker_pid=1002)

    assert first is not None
    assert second is None


def test_a_failing_job_is_retried_and_then_given_up_on(catalog):
    from ape.jobs.queue import MAX_ATTEMPTS, fail_job

    with catalog() as session:
        enqueue(session, JobKind.ANALYZE, {"unit": 0}, dedupe_key="fragile")
        session.commit()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        with catalog() as session:
            record = claim_job(session)
            assert record is not None, f"tentativo {attempt}: job non ripreso"
            state = fail_job(session, record.id, "guasto simulato")
            session.commit()
        expected = JobState.QUEUED if attempt < MAX_ATTEMPTS else JobState.FAILED
        assert state is expected

    with catalog() as session:
        job = session.scalars(select(Job)).one()
        assert job.state is JobState.FAILED
        assert job.attempts == MAX_ATTEMPTS
        assert "guasto simulato" in job.error


def test_requeue_running_is_what_startup_does(catalog):
    with catalog() as session:
        enqueue(session, JobKind.ANALYZE, {"unit": 0}, dedupe_key="orfano")
        session.commit()
    with catalog() as session:
        claim_job(session, worker_pid=4242)
        session.commit()
    with catalog() as session:
        assert requeue_running(session) == 1
        session.commit()
        job = session.scalars(select(Job)).one()
        assert job.state is JobState.QUEUED
        assert job.claimed_by is None
        # The attempt is not forgotten: a job that strands repeatedly must
        # eventually stop being retried rather than loop for ever.
        assert job.attempts == 1

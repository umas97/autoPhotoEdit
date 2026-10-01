# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Waking idle workers when there is work, instead of having them ask.

Section 26: at idle the program uses about 0% of the CPU -- no polling, no
background timers. Workers that asked the queue every 150 ms measured 25% of a
core with fourteen of them, from the moment the server started until it closed.

So a worker that finds the queue empty waits on a *doorbell*, and whoever makes
a job claimable rings it once the transaction that did so has committed: the
server when the user imports or retries, a worker when a job it ran enqueues
the next one (proxy -> analysis). The bell is a semaphore topped up to one
permit per worker. Permits outlive the ring, so a job committed between "the
queue is empty" and "wait" is not missed: the wait returns at once. Not a
``multiprocessing.Condition``, whose ``notify_all`` waits -- with no timeout --
for every sleeper to acknowledge: one worker killed in its sleep and the
server would hang at the next enqueue.

The wait still times out, once a minute, for three things that have no bell to
ring: noticing that the server died (the worker is then orphaned and leaves),
dropping a model nobody used for five minutes, and a job enqueued by some
process outside the pool. Fourteen workers waking once a minute is nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import event
from sqlalchemy.orm import Session

__all__ = ["Doorbell", "install", "ring_after_commit"]

#: The longest a worker sleeps with nothing to do. Bounded by the five minutes
#: after which an unused model is dropped (section 26): checked once a minute,
#: it goes within six.
IDLE_TIMEOUT = 60.0

_SESSION_FLAG = "ape_ring_on_commit"


@dataclass
class Doorbell:
    """A semaphore shared by the pool and its workers, and the pool's size.

    Built by the pool from its multiprocessing context and handed to each
    worker at start: synchronisation primitives cross a process boundary only
    that way, not through a queue or a pipe later on.
    """

    permits: Any
    size: int

    @classmethod
    def create(cls, context: Any, size: int) -> Doorbell:
        return cls(context.Semaphore(0), max(1, size))

    def ring(self) -> None:
        """Wake the waiting workers: top the permits up to one each.

        Topped up rather than added to, so a thousand rings while every worker
        is busy leave at most ``size`` spare permits -- as many needless looks
        at the queue -- instead of a thousand. The count is read without a
        lock; a race only means a permit more or less, and a permit less is
        still at least one.
        """
        try:
            missing = self.size - self.permits.get_value()
        except NotImplementedError:  # macOS has no sem_getvalue
            missing = 1
        for _ in range(missing):
            self.permits.release()

    def wait(self, timeout: float = IDLE_TIMEOUT) -> bool:
        """Sleep until rung. Returns whether there was a ring."""
        return bool(self.permits.acquire(True, timeout))


#: The bell of this process's pool: set by the pool in the server, and by
#: ``worker_loop`` in each worker. ``None`` where there is no pool to wake --
#: a script, a test calling ``worker_loop`` without one.
_bell: Doorbell | None = None


def install(bell: Doorbell | None) -> None:
    global _bell
    _bell = bell


def ring_after_commit(session: Session) -> None:
    """Ring this process's bell when ``session`` commits.

    After, not now: a worker woken before the commit would look at the queue,
    find nothing, and go back to sleep with the job still waiting. Once per
    transaction however many jobs it adds.
    """
    if _bell is None or session.info.get(_SESSION_FLAG):
        return
    session.info[_SESSION_FLAG] = True
    bell = _bell

    def _ring(committed: Session) -> None:
        committed.info.pop(_SESSION_FLAG, None)
        bell.ring()

    event.listen(session, "after_commit", _ring, once=True)

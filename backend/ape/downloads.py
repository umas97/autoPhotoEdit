# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The only two network calls the program makes, and how they are made.

Section 19 allows exactly two: downloading an ONNX model (section 17) and
updating the lensfun data, both on an explicit action of the user. Both go
through :class:`Download`, so there is one place to read to know what the
program sends over the network: a plain ``GET`` of a public URL, no cookies, no
identifying headers beyond a ``User-Agent`` naming the program, nothing about
the user's photos.

A download runs in a thread of the server process -- it is waiting on a socket,
not on a CPU, and the workers are busy with photos -- and it reports progress
through the object itself, which the API reads when the interface asks. No timer
and no polling of its own (section 26).

The file lands next to its destination as ``.part`` and is renamed only after
its size and SHA-256 have been checked, so a crash, a cancellation or a
tampered mirror can never leave a half-written or wrong file where the program
would load it (section 17, test 18).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .safety import assert_outside_source

__all__ = ["Download", "DownloadCancelled", "DownloadError", "DownloadState"]

_log = logging.getLogger(__name__)

#: Long enough for a slow mirror to answer, short enough that a machine without
#: a network says so in seconds rather than minutes.
_TIMEOUT_S = 20.0

_CHUNK = 1 << 16


class DownloadState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadError(RuntimeError):
    """The download did not produce the file it promised. The message is Italian."""


class DownloadCancelled(DownloadError):
    pass


@dataclass
class Download:
    """One file from one URL, verified before it is kept.

    ``sha256`` may be ``None`` only for data whose integrity is checked another
    way -- the lensfun archive, which is validated by loading it -- never for a
    model (section 17 makes the check mandatory there, and the registry refuses
    an unpinned entry before a download is even offered).
    """

    url: str
    destination: Path
    sha256: str | None = None
    expected_size: int | None = None
    state: DownloadState = DownloadState.IDLE
    received: int = 0
    total: int | None = None
    error: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def fraction(self) -> float | None:
        total = self.total or self.expected_size
        if not total:
            return None
        return min(1.0, self.received / total)

    def cancel(self) -> None:
        self._cancel.set()

    def start(self, on_done: Callable[[Download], None] | None = None) -> None:
        """Run :meth:`run` in a background thread. Idempotent while running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._cancel.clear()

        def target() -> None:
            # A failure is recorded on the object; the thread has nobody to raise to.
            with contextlib.suppress(DownloadError):
                self.run()
            if on_done is not None:
                on_done(self)

        self._thread = threading.Thread(target=target, name="ape-download", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run(self) -> Path:
        """Fetch, verify, and move into place. Blocking.

        Raises:
            DownloadCancelled: :meth:`cancel` was called.
            DownloadError: no network, a server error, a short file or a
                checksum that does not match. Nothing is left on disk.
        """
        self.state = DownloadState.RUNNING
        self.received, self.error = 0, None
        partial = self.destination.with_name(self.destination.name + ".part")
        assert_outside_source(partial)
        assert_outside_source(self.destination)
        partial.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(
                self.url, headers={"User-Agent": "autoPhotoEdit"}
            )
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                length = response.headers.get("Content-Length")
                self.total = int(length) if length and length.isdigit() else None
                with open(partial, "wb") as handle:
                    while chunk := response.read(_CHUNK):
                        if self._cancel.is_set():
                            raise DownloadCancelled("download annullato")
                        handle.write(chunk)
                        digest.update(chunk)
                        self.received += len(chunk)
            self._verify(digest.hexdigest())
            os.replace(partial, self.destination)
        except DownloadCancelled as exc:
            self._fail(partial, DownloadState.CANCELLED, str(exc))
            raise
        except DownloadError as exc:
            self._fail(partial, DownloadState.FAILED, str(exc))
            raise
        except OSError as exc:
            # urllib's URLError is an OSError: no network, DNS, refused, timeout.
            reason = getattr(exc, "reason", None) or exc
            message = f"download non riuscito, rete non raggiungibile? ({reason})"
            self._fail(partial, DownloadState.FAILED, message)
            raise DownloadError(message) from exc
        self.state = DownloadState.DONE
        _log.info("scaricato %s (%d byte)", self.destination.name, self.received)
        return self.destination

    def _verify(self, actual: str) -> None:
        if self.expected_size is not None and self.received != self.expected_size:
            raise DownloadError(
                f"file incompleto: {self.received} byte invece di {self.expected_size}"
            )
        if self.sha256 is not None and actual != self.sha256:
            raise DownloadError(
                "il file scaricato non corrisponde al checksum atteso ed è stato cancellato"
            )

    def _fail(self, partial: Path, state: DownloadState, message: str) -> None:
        self.state, self.error = state, message
        partial.unlink(missing_ok=True)
        _log.warning("download di %s: %s", self.url, message)

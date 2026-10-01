# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One running program at a time (docs/SPEC.md section 21.1).

The lockfile in ``$XDG_RUNTIME_DIR`` holds the server's PID, its port and the
PID of the window. That is enough to answer the three questions a second launch
asks, in this order:

1. is there a lockfile, and is the process it names still alive? A file naming a
   dead PID is stale -- the machine was rebooted, or the program was killed --
   and is removed without ceremony;
2. does that process answer on its port? A live PID whose port is silent is not
   this program: PIDs are recycled, and refusing to start because some unrelated
   process inherited the number would be a bug the user could not diagnose;
3. is its window still open? If it is, the second launch raises it and exits 0.
   If it is not -- closed in tab mode, or started with ``--no-window`` -- the
   second launch opens a window **on the existing server** rather than starting
   a second one.

Raising the window needs a tool that can talk to the window manager. ``wmctrl``
and ``xdotool`` are the two that are usually there; when neither is, the launch
still exits 0 and says why, because failing to raise a window is a nuisance and
exiting non-zero from the desktop icon is a fault.

Nothing here uses ``fcntl`` locks. An advisory lock dies with the file
descriptor, so it cannot answer question 1 after a crash, which is the only
question that matters.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_settings

__all__ = [
    "AlreadyRunning",
    "LockInfo",
    "PortBusy",
    "acquire",
    "port_is_free",
    "raise_existing_window",
    "read_lock",
    "server_responds",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockInfo:
    """What the lockfile says. ``window_pid`` is ``None`` in tab or headless mode."""

    pid: int
    port: int
    window_pid: int | None = None
    started_at: float = 0.0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "port": self.port,
            "window_pid": self.window_pid,
            "started_at": self.started_at,
        }


class AlreadyRunning(RuntimeError):
    """A server of ours is up. Carries the lock so the caller can talk to it."""

    def __init__(self, info: LockInfo) -> None:
        super().__init__(f"autoPhotoEdit è già in esecuzione su {info.url}")
        self.info = info


class PortBusy(RuntimeError):
    """The port is taken by something that is not us (section 21.1)."""

    def __init__(self, port: int) -> None:
        super().__init__(
            f"la porta {port} è occupata da un altro programma; "
            f"usa --port per sceglierne un'altra"
        )
        self.port = port


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process with our PID: alive, but not ours. Treated as
        # alive here; ``server_responds`` is what decides whether it is us.
        return True
    except OSError as exc:  # pragma: no cover - defensive
        if exc.errno == errno.ESRCH:
            return False
    return True


def read_lock(path: Path | None = None) -> LockInfo | None:
    """Parse the lockfile, or ``None`` if it is absent or unreadable.

    A truncated or corrupt file counts as absent: it can only come from a crash
    between ``open`` and ``write``, and refusing to start over it would leave the
    user with a program that will not launch and no way to find out why.
    """
    target = path or get_settings().lock_path
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return LockInfo(
            pid=int(payload["pid"]),
            port=int(payload["port"]),
            window_pid=int(payload["window_pid"]) if payload.get("window_pid") else None,
            started_at=float(payload.get("started_at", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def server_responds(port: int, *, timeout: float = 1.0) -> bool:
    """Does *our* server answer on this port? Asks ``/api/health``.

    A stranger on the port answers something else, or nothing; either way this
    returns False and the caller reports the port as busy rather than pretending
    to have found an existing instance.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(  # noqa: S310 - loopback, fixed scheme
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    return payload.get("status") == "ok"


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Can we bind this port? Asked before uvicorn does, to fail with a sentence."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


@dataclass
class LockFile:
    """The lock this process holds. Released on exit, and only by its owner."""

    path: Path
    info: LockInfo

    def update(self, **fields: Any) -> None:
        """Rewrite the lock with new facts -- in practice, the window's PID."""
        self.info = LockInfo(**(self.info.as_dict() | fields))
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and moved into place: a reader never sees half a file.
        temporary = self.path.with_suffix(".lock.tmp")
        temporary.write_text(json.dumps(self.info.as_dict()), encoding="utf-8")
        temporary.replace(self.path)

    def release(self) -> None:
        """Remove the lockfile, unless another process has taken it over."""
        current = read_lock(self.path)
        if current is not None and current.pid != self.info.pid:
            _log.debug("lockfile appartiene ora al pid %s: non lo rimuovo", current.pid)
            return
        self.path.unlink(missing_ok=True)


def acquire(port: int, *, path: Path | None = None) -> LockFile:
    """Take the lock for this process.

    Args:
        port: the port the server is about to bind.
        path: override the lockfile location. Tests use it; nothing else should.

    Returns:
        The held lock. The caller must call :meth:`LockFile.release` on the way
        out -- ``cmd_serve`` does it in a ``finally``.

    Raises:
        AlreadyRunning: a server of ours is already up and answering.
        PortBusy: the port is taken by a process that is not ours.
    """
    target = path or get_settings().lock_path
    existing = read_lock(target)

    if existing is not None:
        if _pid_alive(existing.pid) and server_responds(existing.port):
            raise AlreadyRunning(existing)
        _log.info("lockfile stantio (pid %s): lo rimuovo", existing.pid)
        target.unlink(missing_ok=True)

    # Nothing of ours is running, so the port must be ours to take. Section 21.1
    # forbids falling back to another port: the address has to stay predictable.
    if not port_is_free(port):
        if server_responds(port):
            # Ours, but without a lockfile -- the file was deleted under a live
            # server. Reconstruct enough of it to raise the window and exit.
            raise AlreadyRunning(LockInfo(pid=0, port=port))
        raise PortBusy(port)

    lock = LockFile(path=target, info=LockInfo(pid=os.getpid(), port=port, started_at=time.time()))
    lock._write()
    return lock


def raise_existing_window(info: LockInfo) -> bool:
    """Bring the running instance's window to the front (section 21.1).

    Returns:
        True if a tool was found and reported success. False means either that
        there is no window to raise or that this desktop has neither ``wmctrl``
        nor ``xdotool``; in both cases the caller still exits 0, and says what
        happened.
    """
    if info.window_pid is None or not _pid_alive(info.window_pid):
        return False

    # The class the window was started with, the one ``StartupWMClass`` points
    # at in the desktop entry (section 18).
    from .window_id import window_class

    wm_class = window_class()

    wmctrl = shutil.which("wmctrl")
    if wmctrl:
        result = subprocess.run(  # noqa: S603 - fixed argv, resolved path
            [wmctrl, "-x", "-a", wm_class], capture_output=True, check=False
        )
        if result.returncode == 0:
            return True

    xdotool = shutil.which("xdotool")
    if xdotool:
        result = subprocess.run(  # noqa: S603 - fixed argv, resolved path
            [xdotool, "search", "--class", wm_class, "windowactivate", "%1"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True

    _log.info("nessuno strumento per attivare la finestra (wmctrl o xdotool)")
    return False

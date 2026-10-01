# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Starting the program, and stopping it (docs/SPEC.md sections 21.1 to 21.3).

``__main__`` parses the command line; this is what that command line *does* when
it is the bare ``autophotoedit`` the desktop entry runs. The order is fixed and
every step depends on the one before:

1. take the lockfile, which is also how a second launch discovers the first and
   raises its window instead of starting a second server;
2. build the application and hand it two callbacks -- one that stops everything
   ("Esci dall’applicazione") and one that opens a window on this server;
3. start a thread that waits for the server to answer and only then opens the
   window. A browser launched before the port is listening shows its own
   connection-refused page and does not come back;
4. serve, and on the way out close the window and release the lock.

The interesting decision is what closing the window means. With an empty queue
it means quit, as in any desktop program. With jobs still moving it means
whatever the interface said in its closing beacon -- and saying nothing, which
is what a crash looks like, means carry on (section 21.3).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def want_window(args: argparse.Namespace) -> bool:
    """Should this launch open a window?

    Opening one is the default: the bare command is what the desktop icon runs
    (section 18). Two things turn it off -- ``--no-window``, and a session with
    no display at all, which is the SSH case section 21.2 mentions. ``--window``
    overrides the display check, for a session where the variables lie.
    """
    if getattr(args, "no_window", False):
        return False
    if getattr(args, "window", False):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_server(args: argparse.Namespace) -> int:
    """Start the local server and, unless told otherwise, its window.

    The order is fixed and each step depends on the one before: take the lock so
    that a second launch cannot start a second server (section 21.1), build the
    application, arrange for the window to open once the server answers, and
    only then start serving. The window is opened from a thread because uvicorn
    owns the main one, and it is opened *after* health rather than before
    because a browser that arrives early shows its own error page and does not
    come back.
    """
    import uvicorn

    from . import single_instance as si
    from .api.app import create_app
    from .api.routes_app import close_intent
    from .config import get_settings
    from .launcher import WindowWatcher, current_session

    settings = get_settings()
    settings.ensure_dirs()
    port = args.port or settings.port
    url = f"http://{settings.host}:{port}/"
    with_window = want_window(args)

    try:
        lock = si.acquire(port)
    except si.AlreadyRunning as exc:
        # Section 21.1: a second launch is a request to *see* the program, not
        # to start it again. Raise the window if there is one; open one on the
        # running server if there is not; exit 0 either way.
        running = exc.info
        print(f"autoPhotoEdit è già in esecuzione su {running.url}")
        if si.raise_existing_window(running):
            print("finestra esistente portata in primo piano")
        elif with_window:
            answer = _ask_running_server_for_window(running.port)
            if answer is None:
                print("l'istanza in esecuzione non ha potuto aprire una finestra")
            elif answer.get("opened"):
                print("nuova finestra aperta sull'istanza esistente")
            else:
                print(
                    "la finestra è già aperta: cercala fra le finestre aperte "
                    "(installa wmctrl o xdotool perché venga portata in primo piano)"
                )
        return 0
    except si.PortBusy as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 2

    from .logs import configure_server

    configure_server()
    logging.getLogger(__name__).info("avvio del server sulla porta %d", port)
    app = create_app(start_workers=not args.no_workers, worker_count=args.workers)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=port,
        log_level=str(args.log_level or "warning").lower(),
        # The window talks to this server over a loopback socket; access logs of
        # every thumbnail request are noise, and section 19 has real logging.
        access_log=False,
    )
    server = uvicorn.Server(config)

    def stop_everything() -> None:
        """"Esci dall'applicazione": close the window, then end the server."""
        current_session().close()
        server.should_exit = True

    def teardown() -> None:
        """Release what this process owns outside itself. Runs exactly once.

        Hung on the application's shutdown rather than on a ``finally`` around
        ``server.run()``: uvicorn re-raises the signal that stopped it once its
        own handlers are back in place, so the code after ``run()`` is never
        reached on a SIGTERM -- and the window would outlive the program that
        opened it, behind a lockfile naming a process that no longer exists.

        Both halves are safe to run twice: closing a process that has already
        gone is a no-op, and a lock is only released by the process that holds
        it.
        """
        current_session().close()
        lock.release()

    app.state.shutdown_hook = stop_everything
    app.state.teardown = teardown

    def window_closed() -> None:
        """The user closed the window. Whether that ends the program (21.3).

        With nothing in the queue, closing the window closes the program, as in
        any desktop application. With jobs still moving it follows the decision
        the interface sent while closing -- and the absence of one, which is
        what a crash or a kill looks like, means continue.
        """
        decision = close_intent()
        if _active_jobs() > 0 and decision != "stop":
            print(
                "finestra chiusa: i job in corso proseguono in background; "
                f"riapri con «autophotoedit» o su {url}"
            )
            lock.update(window_pid=None)
            return
        stop_everything()

    if with_window:
        watcher = WindowWatcher(
            port=port,
            browser=getattr(args, "browser", None),
            on_open=lambda session: lock.update(window_pid=session.pid),
            on_close=window_closed,
        )
        app.state.open_window_hook = lambda: _open_another_window(port, args, lock)
        watcher.start()
        print(f"autoPhotoEdit in ascolto su {url} — apertura della finestra…")
    else:
        print(f"autoPhotoEdit in ascolto su {url}")

    try:
        server.run()
    finally:
        # Belt and braces: on a clean return the lifespan has already run this.
        teardown()
    return 0


def _active_jobs() -> int:
    """Queued plus running, over the whole catalogue."""
    from sqlalchemy import func, select

    from .db.models import Job, JobState
    from .db.session import session_scope

    with session_scope() as session:
        return int(
            session.scalar(
                select(func.count(Job.id)).where(
                    Job.state.in_((JobState.QUEUED, JobState.RUNNING))
                )
            )
            or 0
        )


def _open_another_window(port: int, args: argparse.Namespace, lock: object) -> dict[str, object]:
    """Open a window on this already-running server (section 21.1)."""
    from .launcher import WindowWatcher, current_session

    live = current_session()
    if live.alive:
        # There is already a window on this server; opening a second one would
        # be exactly what section 21.1 forbids.
        return live.as_dict() | {"opened": False}

    watcher = WindowWatcher(
        port=port,
        browser=getattr(args, "browser", None),
        on_open=lambda session: lock.update(window_pid=session.pid),  # type: ignore[attr-defined]
    )
    watcher.start()
    watcher.started.wait(timeout=25.0)
    return current_session().as_dict() | {"opened": True}


def _ask_running_server_for_window(port: int) -> dict[str, object] | None:
    """Ask the instance that holds the lock to open a window of its own.

    Returns:
        What that instance answered, including whether it opened anything, or
        ``None`` if it could not be asked at all.
    """
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 - loopback, fixed scheme
        f"http://127.0.0.1:{port}/api/window/open", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return dict(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None

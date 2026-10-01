# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The application window (docs/SPEC.md section 21.2).

The user must never see a browser. A Chromium in ``--app`` mode is a window with
no address bar, no tabs and no bookmarks, and it is the whole of the plan; what
this module adds is the four details that make it behave like a desktop program
rather than like a browser someone stripped:

* a **dedicated profile** under ``$XDG_DATA_HOME``. Without it the window
  inherits the user's extensions, their session and their enterprise policy, and
  opens differently depending on whether their browser happened to be running;
* ``--class=autophotoedit``, which is what ``StartupWMClass`` in the desktop
  entry points at (section 18). Without it the dock shows a Chromium icon with
  the wrong name;
* the browser process is a **child of this one**, so closing the window is a
  signal the program can act on (section 21.3) and "Esci" can close the window;
* a fallback that says so. On a machine with only Firefox there is no
  application mode, so the UI opens in an ordinary tab and the interface shows a
  one-line notice -- degraded, but never mysterious.

The chain is: an installed PWA in our own profile, then application mode, then a
plain tab. It stops at the first that works and reports which one it took, and
``/api/window`` hands that answer to the interface.

This module never installs and never downloads a browser. If there is none, it
says so and the program still runs -- the server is the program; the window is
how it is looked at.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .config import get_settings
from .single_instance import WM_CLASS, server_responds

__all__ = [
    "BROWSER_CANDIDATES",
    "FLATPAK_CANDIDATES",
    "WindowMode",
    "WindowSession",
    "current_session",
    "find_browser",
    "open_window",
    "wait_for_server",
    "web_app_id",
]

_log = logging.getLogger(__name__)

#: Executables tried in order, as section 21.2 lists them.
BROWSER_CANDIDATES: tuple[str, ...] = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "brave-browser",
    "microsoft-edge",
)

#: The same browsers as Flatpak application ids, tried after the native ones.
FLATPAK_CANDIDATES: tuple[str, ...] = (
    "org.chromium.Chromium",
    "com.google.Chrome",
    "com.brave.Browser",
    "com.microsoft.Edge",
)

#: Geometry of a fresh window. The profile remembers it afterwards.
WINDOW_SIZE = "1400,900"


class WindowMode(StrEnum):
    """How the interface ended up on screen. The UI shows the last two."""

    PWA = "pwa"
    """An installed PWA, opened by its app id in our profile."""

    APP = "app"
    """A Chromium in application mode: the dedicated window of section 21.2."""

    TAB = "tab"
    """The default browser, in an ordinary tab. Everything works, nothing hides."""

    NONE = "none"
    """``--no-window``: the server alone."""


@dataclass
class WindowSession:
    """The window this process opened, and how.

    Attributes:
        mode: which link of the chain answered.
        browser: the executable or Flatpak id behind it, when there is one.
        process: the child process, when the window is ours to close.
        note: a sentence in Italian for the interface, set only when the
            experience is degraded (:attr:`WindowMode.TAB`).
    """

    mode: WindowMode = WindowMode.NONE
    browser: str | None = None
    process: subprocess.Popen[bytes] | None = None
    note: str | None = None
    url: str | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self, timeout: float = 5.0) -> None:
        """Close the window. Politely first, then not."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "browser": self.browser,
            "window_pid": self.pid,
            "alive": self.alive,
            "note": self.note,
            "url": self.url,
        }


#: The session of this process. One window per server, so one global is enough.
_session = WindowSession()
_session_lock = threading.Lock()


def current_session() -> WindowSession:
    """What the interface is being shown in, right now."""
    with _session_lock:
        return _session


def _set_session(session: WindowSession) -> WindowSession:
    global _session
    with _session_lock:
        _session = session
    return session


def profile_dir() -> Path:
    """The window's own browser profile (section 21.2)."""
    return get_settings().data_dir / "window-profile"


def find_browser(
    explicit: str | None = None,
    *,
    candidates: tuple[str, ...] | None = None,
    flatpaks: tuple[str, ...] | None = None,
) -> list[str] | None:
    """The command that opens a Chromium window, or ``None`` if there is none.

    Args:
        explicit: ``--browser PATH``. Falls back to ``APE_BROWSER``.
        candidates: override the executable list. Tests force it empty to
            exercise the tab fallback of section 21.2; nothing else should.
        flatpaks: override the Flatpak list, for the same reason.

    Returns:
        The argv prefix to run, so that a Flatpak and a native binary are the
        same thing to the caller.
    """
    chosen = explicit or os.environ.get("APE_BROWSER")
    if chosen:
        path = Path(chosen).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return [str(path)]
        resolved = shutil.which(chosen)
        if resolved:
            return [resolved]
        _log.warning("browser indicato non trovato: %s", chosen)
        return None

    for name in BROWSER_CANDIDATES if candidates is None else candidates:
        resolved = shutil.which(name)
        if resolved:
            return [resolved]

    flatpak = shutil.which("flatpak")
    if flatpak:
        for app_id in FLATPAK_CANDIDATES if flatpaks is None else flatpaks:
            probe = subprocess.run(  # noqa: S603 - fixed argv, resolved path
                [flatpak, "info", app_id], capture_output=True, check=False
            )
            if probe.returncode == 0:
                return [flatpak, "run", app_id]
    return None


def web_app_id(start_url: str) -> str:
    """The id Chromium gives a web app installed from ``start_url``.

    The algorithm is Chromium's own: SHA-256 of the manifest id, the first
    sixteen bytes, each nibble mapped onto ``a``..``p``. It is reproduced here
    only to *recognise* an installed app in our profile -- opening ``--app-id``
    for something that is not installed gives a window that closes itself.
    """
    digest = hashlib.sha256(start_url.encode("utf-8")).hexdigest()[:32]
    return "".join(chr(ord("a") + int(nibble, 16)) for nibble in digest)


def installed_app_id(start_url: str, profile: Path | None = None) -> str | None:
    """The app id if the user has installed the PWA in our profile, else ``None``.

    Chromium keeps one directory per installed web app under the profile. Its
    presence is the cheapest reliable answer, and it costs no browser launch.
    """
    app_id = web_app_id(start_url)
    root = (profile or profile_dir()) / "Default" / "Web Applications"
    for parent in (root / "Manifest Resources", root):
        if (parent / app_id).is_dir():
            return app_id
    return None


def _app_argv(command: list[str], url: str, profile: Path) -> list[str]:
    return [
        *command,
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--class={WM_CLASS}",
        f"--name={WM_CLASS}",
        f"--window-size={WINDOW_SIZE}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def _pwa_argv(command: list[str], app_id: str, profile: Path) -> list[str]:
    return [
        *command,
        f"--app-id={app_id}",
        f"--user-data-dir={profile}",
        f"--class={WM_CLASS}",
        f"--name={WM_CLASS}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def open_window(
    url: str,
    *,
    browser: str | None = None,
    candidates: tuple[str, ...] | None = None,
    flatpaks: tuple[str, ...] | None = None,
    prefer_installed: bool = True,
) -> WindowSession:
    """Open the interface, taking the first link of the chain that works.

    Args:
        url: where the server is listening.
        browser: an explicit browser, from ``--browser`` or ``APE_BROWSER``.
        candidates: see :func:`find_browser`.
        flatpaks: see :func:`find_browser`.
        prefer_installed: try the installed PWA first. Off in tests that want
            application mode specifically.

    Returns:
        The session, already recorded as the current one so that ``/api/window``
        can report it.
    """
    command = find_browser(browser, candidates=candidates, flatpaks=flatpaks)
    profile = profile_dir()

    if command is not None:
        profile.mkdir(parents=True, exist_ok=True)
        app_id = installed_app_id(url, profile) if prefer_installed else None
        argv = (
            _pwa_argv(command, app_id, profile)
            if app_id
            else _app_argv(command, url, profile)
        )
        mode = WindowMode.PWA if app_id else WindowMode.APP
        try:
            process = subprocess.Popen(  # noqa: S603 - argv built above, no shell
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=False,  # the window stays our child (section 21.2)
            )
        except OSError as exc:
            _log.warning("impossibile avviare %s: %s", command[0], exc)
        else:
            _log.info("finestra aperta con %s in modalità %s", command[0], mode.value)
            return _set_session(
                WindowSession(mode=mode, browser=command[0], process=process, url=url)
            )

    # Nothing with an application mode. The tab still works, and the interface
    # has to say why it looks like a browser (section 21.2, point 3).
    _log.info("nessun browser Chromium: apro una scheda del browser predefinito")
    try:
        webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001 - a missing browser is not a crash
        _log.warning("nessun browser disponibile: %s", exc)
    return _set_session(
        WindowSession(
            mode=WindowMode.TAB,
            browser=None,
            process=None,
            url=url,
            note=(
                "Nessun browser compatibile con la finestra dedicata: autoPhotoEdit "
                "è aperto in una scheda. Installa Chromium, oppure usa «Installa come "
                "app» dal menu del browser, per avere una finestra propria."
            ),
        )
    )


def wait_for_server(port: int, *, timeout: float = 20.0, poll: float = 0.1) -> bool:
    """Block until the server answers, or the timeout runs out.

    The window is opened after this, never before: a Chromium that arrives first
    shows its own connection-refused page and does not retry.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_responds(port, timeout=0.5):
            return True
        time.sleep(poll)
    return False


@dataclass
class WindowWatcher:
    """Opens the window once the server is up, and notices when it is closed.

    Runs in a thread because ``uvicorn`` owns the main one. Two jobs, in order:
    wait for health then open the window; then wait for the child to exit and
    call ``on_close``, which is how closing the window stops the program
    (section 21.3).
    """

    port: int
    on_close: object = None  # Callable[[], None]
    on_open: object = None  # Callable[[WindowSession], None]
    browser: str | None = None
    prefer_installed: bool = True
    started: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ape-window", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if not wait_for_server(self.port):
            _log.error("il server non ha risposto: non apro la finestra")
            self.started.set()
            return

        session = open_window(f"http://127.0.0.1:{self.port}/", browser=self.browser,
                              prefer_installed=self.prefer_installed)
        if callable(self.on_open):
            self.on_open(session)
        self.started.set()

        if session.process is None:
            # A tab is not a reliable signal of intent: closing it must not stop
            # the server (section 21.3).
            return
        session.process.wait()
        _log.info("finestra chiusa: spengo il server")
        if callable(self.on_close):
            self.on_close()

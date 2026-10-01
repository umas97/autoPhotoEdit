# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 19 of section 13, first half: the window and the single instance.

The parts that can be automated are automated here, with a fake browser -- a
script that records the command line it was given and then sits still. That is
enough to check everything the specification asks about the launcher, because
what section 21.2 requires is a *command line*: application mode, a dedicated
profile, and the ``WM_CLASS`` the desktop entry points at. Whether Chromium
honours it is Chromium's business, and it is checked by hand once.

The end-to-end cases start the real program in a subprocess, because that is the
only way to see the thing under test: a second launch talking to the first one
through a lockfile and a port. In process, there is no first one.

Not automated, and checked by hand: that the resulting window has no
address bar. No test on this machine can see a window.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from ape import launcher
from ape import single_instance as si

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_browser(tmp_path: Path) -> tuple[Path, Path]:
    """A script that records its argv and then stays alive, like a window."""
    log = tmp_path / "browser-argv.jsonl"
    script = tmp_path / "fake-browser"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv) + '\\n')\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, log


def _launches(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


# --- the command line (section 21.2) ----------------------------------------


def test_the_window_is_a_chromium_in_application_mode(tmp_path, xdg_home, monkeypatch):
    script, log = _fake_browser(tmp_path)
    session = launcher.open_window(
        "http://127.0.0.1:8787/", browser=str(script), prefer_installed=False
    )
    try:
        # The child has to get as far as writing its line before we read it.
        for _ in range(100):
            if _launches(log):
                break
            time.sleep(0.05)

        assert session.mode is launcher.WindowMode.APP
        assert session.alive
        (argv,) = _launches(log)
        assert "--app=http://127.0.0.1:8787/" in argv
        assert f"--class={si.WM_CLASS}" in argv
        assert f"--name={si.WM_CLASS}" in argv
        assert f"--user-data-dir={launcher.profile_dir()}" in argv
        assert "--no-first-run" in argv
        # The profile is a real directory by now: Chromium would not create it
        # before writing into it, and section 21.2 wants the geometry kept.
        assert launcher.profile_dir().is_dir()
    finally:
        session.close()


def test_a_browser_that_is_not_there_is_not_invented(tmp_path):
    assert launcher.find_browser(str(tmp_path / "nessun-browser")) is None


def test_the_explicit_browser_wins_over_the_search(tmp_path, monkeypatch):
    script, _ = _fake_browser(tmp_path)
    monkeypatch.setenv("APE_BROWSER", str(script))
    assert launcher.find_browser() == [str(script)]


def test_with_no_chromium_at_all_it_falls_back_to_a_tab_and_says_so(monkeypatch, xdg_home):
    """Fallback 3 of section 21.2: it works, it is degraded, and it declares it."""
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.delenv("APE_BROWSER", raising=False)

    session = launcher.open_window(
        "http://127.0.0.1:8787/", candidates=(), flatpaks=()
    )

    assert session.mode is launcher.WindowMode.TAB
    assert session.process is None
    assert opened == ["http://127.0.0.1:8787/"]
    # The interface has to be able to explain the situation, so the sentence
    # comes from here rather than from a string table the server cannot reach.
    assert session.note is not None and "scheda" in session.note


def test_the_tab_fallback_is_visible_through_the_api(monkeypatch, catalog):
    from fastapi.testclient import TestClient

    from ape.api.app import create_app

    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: None)
    monkeypatch.delenv("APE_BROWSER", raising=False)
    launcher.open_window("http://127.0.0.1:8787/", candidates=(), flatpaks=())

    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as client:
        body = client.get("/api/window").json()
    assert body["mode"] == "tab"
    assert "scheda" in body["note"]
    # Nothing to quit: this server was not started by the launcher.
    assert body["can_quit"] is False


def test_an_installed_pwa_is_only_used_when_it_is_installed(tmp_path, xdg_home):
    url = "http://127.0.0.1:8787/"
    app_id = launcher.web_app_id(url)
    assert len(app_id) == 32 and app_id.isalpha()

    assert launcher.installed_app_id(url) is None
    installed = launcher.profile_dir() / "Default" / "Web Applications" / "Manifest Resources"
    (installed / app_id).mkdir(parents=True)
    assert launcher.installed_app_id(url) == app_id


# --- the lockfile (section 21.1) --------------------------------------------


def test_a_stale_lockfile_is_removed_and_the_launch_continues(xdg_home, tmp_path):
    lock_path = tmp_path / "stale.lock"
    lock_path.write_text(json.dumps({"pid": 999_999, "port": 8787}), encoding="utf-8")

    lock = si.acquire(_free_port(), path=lock_path)
    try:
        assert lock.info.pid == os.getpid()
        assert si.read_lock(lock_path).pid == os.getpid()
    finally:
        lock.release()
    assert not lock_path.exists()


def test_a_port_held_by_a_stranger_is_an_error_not_another_port(xdg_home, tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stranger:
        stranger.bind(("127.0.0.1", 0))
        stranger.listen(1)
        port = stranger.getsockname()[1]
        with pytest.raises(si.PortBusy) as error:
            si.acquire(port, path=tmp_path / "x.lock")
    assert "--port" in str(error.value)


def test_a_lock_is_not_released_by_a_process_that_does_not_own_it(xdg_home, tmp_path):
    lock_path = tmp_path / "held.lock"
    lock = si.acquire(_free_port(), path=lock_path)
    # Another process took the file over in the meantime.
    lock_path.write_text(
        json.dumps({"pid": os.getpid() + 1, "port": 1234}), encoding="utf-8"
    )
    lock.release()
    assert lock_path.exists()


# --- the program, started for real ------------------------------------------


@pytest.fixture
def server_env(tmp_path: Path) -> Iterator[dict[str, str]]:
    """An environment with its own XDG home, for a real subprocess launch."""
    home = tmp_path / "xdg"
    environment = dict(os.environ)
    environment.update(
        {
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_STATE_HOME": str(home / "state"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_RUNTIME_DIR": str(home / "run"),
            "PYTHONPATH": str(BACKEND),
        }
    )
    environment.pop("APE_BROWSER", None)
    (home / "run").mkdir(parents=True, exist_ok=True)
    yield environment


def _serve(environment: dict[str, str], *arguments: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "ape", "serve", "--no-workers", *arguments],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _await_health(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if si.server_responds(port, timeout=0.5):
            return True
        time.sleep(0.1)
    return False


@pytest.mark.slow
def test_one_launch_opens_one_window_and_the_second_opens_none(server_env, tmp_path):
    """The heart of test 19, end to end."""
    script, log = _fake_browser(tmp_path)
    port = _free_port()
    server = _serve(server_env, "--port", str(port), "--window", "--browser", str(script))
    try:
        assert _await_health(port), "il server non ha risposto"

        for _ in range(200):
            if _launches(log):
                break
            time.sleep(0.05)
        assert len(_launches(log)) == 1, "la prima esecuzione deve aprire una finestra sola"

        lock = si.read_lock(Path(server_env["XDG_RUNTIME_DIR"]) / "autophotoedit.lock")
        assert lock is not None and lock.port == port
        assert lock.window_pid is not None, "il lockfile registra il pid della finestra"

        # The second launch. It must not start a server, must not open a second
        # window, and must exit 0 -- clicking the icon twice is not an error.
        second = subprocess.run(
            [
                sys.executable,
                "-m",
                "ape",
                "serve",
                "--port",
                str(port),
                "--window",
                "--browser",
                str(script),
            ],
            env=server_env,
            capture_output=True,
            timeout=60,
        )
        assert second.returncode == 0, second.stderr.decode()
        assert "già in esecuzione" in second.stdout.decode()
        time.sleep(0.5)
        assert len(_launches(log)) == 1, "la seconda esecuzione non apre una seconda finestra"
    finally:
        server.terminate()
        server.wait(timeout=30)
        for launch in _launches(log):
            _ = launch  # the fake browser dies with its parent's shutdown


@pytest.mark.slow
def test_stopping_the_server_closes_the_window_and_frees_the_lock(server_env, tmp_path):
    """Section 21.3, and a bug this test exists because of.

    ``uvicorn`` puts the original signal handlers back and re-raises the signal
    that stopped it, so a ``finally`` around ``server.run()`` never executes on
    a SIGTERM: the window outlived the program and the lockfile kept naming a
    process that was gone, which made the *next* launch refuse to start. The
    cleanup hangs on the application's shutdown instead, which does run.
    """
    script, log = _fake_browser(tmp_path)
    port = _free_port()
    lock_path = Path(server_env["XDG_RUNTIME_DIR"]) / "autophotoedit.lock"
    server = _serve(server_env, "--port", str(port), "--window", "--browser", str(script))

    assert _await_health(port)
    for _ in range(200):
        if si.read_lock(lock_path) and si.read_lock(lock_path).window_pid:
            break
        time.sleep(0.05)
    window_pid = si.read_lock(lock_path).window_pid
    assert window_pid is not None

    server.terminate()
    server.wait(timeout=30)
    time.sleep(1.0)

    assert not lock_path.exists(), "il lockfile deve sparire con il programma"
    with pytest.raises(ProcessLookupError):
        os.kill(window_pid, 0)


@pytest.mark.slow
def test_no_window_starts_no_child_at_all(server_env, tmp_path):
    script, log = _fake_browser(tmp_path)
    port = _free_port()
    server = _serve(
        server_env, "--port", str(port), "--no-window", "--browser", str(script)
    )
    try:
        assert _await_health(port)
        time.sleep(1.0)
        assert _launches(log) == []
        body = json.loads(
            __import__("urllib.request", fromlist=["request"])
            .urlopen(f"http://127.0.0.1:{port}/api/window", timeout=5)
            .read()
        )
        assert body["mode"] == "none"
        assert body["can_quit"] is True, "il launcher ha avviato questo server"
    finally:
        server.terminate()
        server.wait(timeout=30)

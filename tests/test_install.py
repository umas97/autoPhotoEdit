# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Section 18: install.sh and uninstall.sh, in a home directory of their own.

What an automated test can say: the launcher, the icons and the menu entry
land where the desktop looks for them, the entry is valid and its
``StartupWMClass`` is the class the launcher gives the window; running the
installer twice changes nothing; uninstalling removes exactly that, and the
user's data only after an explicit "sì" typed on a terminal. What it cannot
say -- that the dock shows the right icon and name -- is a check by hand.
"""

from __future__ import annotations

import os
import pty
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "install.sh"
UNINSTALL = REPO / "uninstall.sh"

pytestmark = pytest.mark.skipif(
    not (REPO / "backend/ape/static/index.html").is_file() or shutil.which("uv") is None,
    reason="servono la build dell'interfaccia e uv",
)


def _env(home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("XDG_") and key not in ("VIRTUAL_ENV",)
    }
    # uv keeps its own caches under the real home: the test must not refill them.
    env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    env.update(
        HOME=str(home),
        APE_INSTALL_SKIP_SYNC="1",
        APE_INSTALL_SKIP_FRONTEND="1",
    )
    return env


def _run(script: Path, home: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env=_env(home),
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=False,
        **kwargs,
    )


def _entry(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[Desktop Entry]"
    return dict(line.split("=", 1) for line in lines[1:] if "=" in line)


def test_install_puts_an_application_in_the_menu_and_again_changes_nothing(tmp_path: Path):
    from ape.single_instance import WM_CLASS

    home = tmp_path / "casa"
    home.mkdir()
    first = _run(INSTALL, home)
    assert first.returncode == 0, first.stderr + first.stdout
    assert "autoPhotoEdit installato" in first.stdout

    share = home / ".local/share"
    launcher = home / ".local/bin/autophotoedit"
    desktop = share / "applications/autophotoedit.desktop"
    icons = [
        share / "icons/hicolor" / size / "apps" / name
        for size, name in (
            ("48x48", "autophotoedit.png"),
            ("128x128", "autophotoedit.png"),
            ("256x256", "autophotoedit.png"),
            ("scalable", "autophotoedit.svg"),
        )
    ]
    assert launcher.is_file() and os.access(launcher, os.X_OK)
    assert all(icon.is_file() for icon in icons)
    entry = _entry(desktop)
    assert entry["StartupWMClass"] == WM_CLASS == "autophotoedit"
    assert entry["Icon"] == "autophotoedit" and entry["Terminal"] == "false"
    assert entry["Categories"] == "Graphics;Photography;"
    assert entry["Exec"] == str(launcher)
    if shutil.which("desktop-file-validate"):
        check = subprocess.run(
            ["desktop-file-validate", str(desktop)], capture_output=True, text=True
        )
        assert check.returncode == 0, check.stdout + check.stderr

    version = subprocess.run(
        [str(launcher), "--version"], capture_output=True, text=True, env=_env(home), timeout=60
    )
    assert version.returncode == 0 and "autoPhotoEdit" in version.stdout

    before = {p: p.read_bytes() for p in [launcher, desktop, *icons]}
    second = _run(INSTALL, home)
    assert second.returncode == 0, second.stderr
    assert {p: p.read_bytes() for p in before} == before


def test_install_refuses_to_overwrite_a_launcher_it_did_not_write(tmp_path: Path):
    home = tmp_path / "casa"
    theirs = home / ".local/bin/autophotoedit"
    theirs.parent.mkdir(parents=True)
    theirs.write_text("#!/bin/sh\necho qualcun altro\n", encoding="utf-8")
    result = _run(INSTALL, home)
    assert result.returncode != 0 and "non l'ha scritto install.sh" in result.stderr
    assert theirs.read_text(encoding="utf-8") == "#!/bin/sh\necho qualcun altro\n"
    # ...and uninstalling leaves it too.
    assert _run(UNINSTALL, home).returncode == 0
    assert theirs.is_file()


def test_uninstall_removes_the_program_and_keeps_the_data_unless_told(tmp_path: Path):
    home = tmp_path / "casa"
    home.mkdir()
    assert _run(INSTALL, home).returncode == 0
    data = home / ".local/share/autophotoedit"
    (data / "masks").mkdir(parents=True)
    (data / "catalog.db").write_bytes(b"catalogo")
    (data / "masks/mano.png").write_bytes(b"maschera")

    # No terminal to ask on: the program goes, the data stays.
    result = _run(UNINSTALL, home)
    assert result.returncode == 0, result.stderr
    assert not (home / ".local/bin/autophotoedit").exists()
    assert not (home / ".local/share/applications/autophotoedit.desktop").exists()
    assert not list((home / ".local/share/icons/hicolor").rglob("autophotoedit.*"))
    assert (data / "catalog.db").read_bytes() == b"catalogo" and (data / "masks/mano.png").exists()
    assert "i dati restano" in result.stdout

    # On a terminal, anything but "sì" keeps them; "sì" removes them.
    for answer, kept in ((b"forse\n", True), (b"s\xc3\xac\n", False)):
        primary, secondary = pty.openpty()
        try:
            process = subprocess.Popen(
                ["bash", str(UNINSTALL)],
                env=_env(home),
                stdin=secondary,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            os.write(primary, answer)
            out, err = process.communicate(timeout=60)
        finally:
            os.close(primary)
            os.close(secondary)
        assert process.returncode == 0, err
        assert data.exists() is kept, out.decode()

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The window's identity in the dock (docs/SPEC.md section 18).

The dock matches a window to its desktop entry by an identity, and under
Wayland Chromium does not let us choose it: it ignores ``--class`` and derives
one from the *host and path* of the ``--app`` URL -- never the port. Every local
app opened at ``http://127.0.0.1:<port>/`` therefore gets the same identity, and
the dock gives all of them the icon and name of whichever desktop entry claims
it first.

So the window is opened at :data:`WINDOW_PATH`, which the server redirects to
``/``, and :func:`window_class` reproduces the name Chromium derives from it.
``install.sh`` writes that name into ``StartupWMClass``; the launcher passes it
to ``--class`` too, so that an X11 session ends up with the same string.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from .config import get_settings

__all__ = ["WINDOW_PATH", "window_class", "window_url"]

#: Where the window is opened; the server redirects it to ``/``. It exists only
#: to give the window an identity no other local app shares (module docstring).
WINDOW_PATH = "/autophotoedit"

#: Chromium names its app windows after its *own* desktop entry, which is not
#: always the name of the executable. Anything not listed keeps that name, which
#: is right for chromium, brave-browser and microsoft-edge; a Flatpak's entry is
#: its application id.
_DESKTOP_BASE_NAMES = {
    "google-chrome": "chrome",
    "google-chrome-stable": "chrome",
    "chromium-browser": "chromium",
}


def window_url(base: str) -> str:
    """The address the window is opened at, on the server at ``base``."""
    return base.rstrip("/") + WINDOW_PATH


def _desktop_base_name(command: list[str] | None) -> str:
    if command is None:
        # No browser yet (install.sh on a bare machine): Chrome is the most
        # common, and --class makes X11 agree whatever turns up later.
        return "chrome"
    if Path(command[0]).name == "flatpak" and len(command) >= 3:
        return command[2]
    name = Path(command[0]).name
    if Path(command[0]).parts[:2] == ("/", "snap"):
        # A snap's desktop entry is named <snap>_<app>.
        return f"{name}_{name}"
    return _DESKTOP_BASE_NAMES.get(name, name)


def window_class(command: list[str] | None = None, url: str | None = None) -> str:
    """The identity the window gets, as the dock and the window manager see it.

    Chromium's own formula: its desktop base name, then host and path of the
    ``--app`` URL with every other character made ``_``, then the profile's
    directory. The port is not part of it, which is the whole problem the
    :data:`WINDOW_PATH` solves. For Chrome this is
    ``chrome-127.0.0.1__autophotoedit-Default``.

    Args:
        command: the browser, as :func:`find_browser` returns it. Looked up when
            omitted.
        url: the ``--app`` URL. Any port gives the same answer.
    """
    if command is None:
        from .launcher import find_browser  # the launcher imports this module

        command = find_browser()
    parts = urlsplit(url or window_url(f"http://{get_settings().host}/"))
    app_name = f"{parts.hostname or ''}_{parts.path or '/'}"
    return f"{_desktop_base_name(command)}-{re.sub(r'[^A-Za-z0-9._-]', '_', app_name)}-Default"

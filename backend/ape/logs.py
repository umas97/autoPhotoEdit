# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The log on file (section 19): what makes the diagnostics bundle not empty.

``$XDG_STATE_HOME/autophotoedit/logs/autophotoedit.log``, rotated at 5 MB with
four old files kept -- five files of 5 MB at most, as the section asks. The
level is ``APE_LOG_LEVEL`` (``INFO`` by default); ``--log-level`` on the
command line only changes what reaches the terminal.

One file, several processes. Only the **server** rotates it
(``RotatingFileHandler``: its size check looks at the file, so the workers'
lines count too). The **workers** append with a ``WatchedFileHandler``, which
notices that the file it had open was renamed away by a rotation and reopens
the new one. Two rotating handlers on one file would each rename it under the
other; the command-line tools therefore write to the terminal only.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import get_settings

__all__ = [
    "BACKUPS",
    "FORMAT",
    "LOG_NAME",
    "MAX_BYTES",
    "configure_server",
    "configure_worker",
    "log_files",
]

LOG_NAME = "autophotoedit.log"
#: Section 19: "5 file × 5 MB" -- the current one and four rotated.
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 4
FORMAT = "%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"

_OURS = "_ape_file_handler"


def _level() -> int:
    return getattr(logging, str(get_settings().log_level).upper(), logging.INFO)


def _path() -> Path:
    directory = get_settings().log_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / LOG_NAME


def _install(handler: logging.Handler) -> logging.Handler:
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, _OURS, False)]:
        root.removeHandler(old)
        old.close()
    # The root is about to let through what the file wants: the terminal keeps
    # the level it had, by holding it on its own handler.
    before = root.level or logging.WARNING
    for other in root.handlers:
        if other.level == logging.NOTSET:
            other.setLevel(before)
    setattr(handler, _OURS, True)
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.setLevel(_level())
    root.addHandler(handler)
    root.setLevel(min(before, _level()))
    return handler


def configure_server() -> logging.Handler:
    """The server's handler: the one that rotates."""
    return _install(
        logging.handlers.RotatingFileHandler(
            _path(), maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8", delay=True
        )
    )


def configure_worker() -> logging.Handler:
    """A pool worker's handler: appends, and follows the server's rotations."""
    return _install(logging.handlers.WatchedFileHandler(_path(), encoding="utf-8", delay=True))


def log_files() -> list[Path]:
    """The log files on disk, current first."""
    directory = get_settings().log_dir
    names = [LOG_NAME] + [f"{LOG_NAME}.{n}" for n in range(1, BACKUPS + 1)]
    return [directory / name for name in names if (directory / name).is_file()]

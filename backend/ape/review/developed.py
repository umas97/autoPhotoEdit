# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small renders of each photo's current edit, for the review's grid and strips.

The browsing proxy is neutral: it shows the frame, not the edit. On the review
screen that hides the one error the confidence cannot see -- the exposure the
user would have chosen for *this* photo, which strays from the rest of its
scene by most of a stop on the worst photos of their event and which no
measured signal anticipates (phase 7). Shown developed, next to the
others of its scene, the photo that came out too dark is seen at a glance.

One JPEG per photo and *version*, named by both, so that a file on disk is
never out of date: a new version is a new name, rendered by a job
(``jobs/handlers_review.py``) and served immutable. Missing files are asked
for lazily, when the review is looked at (``service.refresh``), like
everything else there; the interface shows the neutral proxy until they come.
It is cache (section 20.3): deleting it costs one render per photo.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import EditVersion, JobKind, Photo

__all__ = [
    "DEVELOPED_EDGE",
    "DEVELOPED_PRIORITY",
    "developed_path",
    "enqueue_missing",
    "existing_version",
    "remove_older",
]

#: Long edge in pixels. The grid's tiles are about 240 css px wide, twice that
#: on a high-density screen; the strip's are smaller still.
DEVELOPED_EDGE = 480

#: Behind the renders the user is waiting on while dragging a slider (10),
#: ahead of any batch: these are what the review screen is showing.
DEVELOPED_PRIORITY = 20

#: Bumped when what the file shows changes, like ``PROXY_VERSION``.
_VERSION = 1


def _directory() -> Path:
    return get_settings().cache_dir / "developed"


def developed_path(photo_id: int, version_id: int) -> Path:
    return _directory() / f"{photo_id}-{version_id}-v{_VERSION}.jpg"


def existing_version(photo_id: int, version: EditVersion | None) -> int | None:
    """The version whose developed render is on disk, if it is the current one."""
    if version is None:
        return None
    return version.id if developed_path(photo_id, version.id).is_file() else None


def enqueue_missing(
    session: Session, photos: Iterable[Photo], versions: dict[int, EditVersion]
) -> int:
    """Ask for the render of every current version that has none.

    Returns how many are missing -- queued now or already waiting -- which the
    interface polls on until they are all there.
    """
    from ..jobs.queue import enqueue
    from ..raw.source import pixels_of

    missing = 0
    for photo in photos:
        version = versions.get(photo.id)
        if version is None or pixels_of(photo) is None or existing_version(photo.id, version):
            continue
        missing += 1
        enqueue(
            session,
            JobKind.RENDER_PREVIEW,
            {"photo_id": photo.id, "version_id": version.id, "purpose": "developed"},
            project_id=photo.project_id,
            dedupe_key=f"developed:{photo.id}:{version.id}",
            priority=DEVELOPED_PRIORITY,
        )
    return missing


def remove_older(photo_id: int, keep: Path) -> None:
    """Drop the renders of the photo's earlier versions."""
    from ..safety import assert_outside_source

    for stale in _directory().glob(f"{photo_id}-*-v*.jpg"):
        # A partial file is another job's render in progress: not stale yet.
        if stale != keep and not stale.name.endswith(".part.jpg"):
            assert_outside_source(stale).unlink(missing_ok=True)

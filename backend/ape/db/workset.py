# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The photos a project is working on: one definition for every screen.

Style, scenes, review and export all act on "the photos of the project", and
that has meant the same four conditions written out in each of them. From phase
11 there is a fifth -- a frame an accepted merge replaced is not a photo the
project is working on any more (section 25.1) -- and a fifth condition added in
four places is how one of them ends up exporting the frames of an HDR next to
the HDR. So it lives here.

Merged photos are in the working set like any other: nothing here asks what
kind of photo a row is.
"""

from __future__ import annotations

from ..db.catalog import Photo

__all__ = ["editing_set", "present"]


def present(project_id: int) -> tuple:
    """Photos of the project that exist and are not stood in for by another."""
    return (
        Photo.project_id == project_id,
        Photo.missing.is_(False),
        Photo.superseded.is_(False),
    )


def editing_set(project_id: int) -> tuple:
    """Photos being developed: present, and kept by the culling."""
    return (*present(project_id), Photo.culled.is_(False))

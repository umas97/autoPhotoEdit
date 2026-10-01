# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running a merge in a worker, and running it again when the cache lost it.

An intermediate is cache (section 20.3): the quota may remove it, and so may
the user. The members are still in the source folder, and the recipe in the
catalogue, so the merge runs again -- deterministically, into the same file
name -- the first time a worker needs the pixels. Nothing downstream notices,
except for the wait.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..db.models import MergeGroup, Photo
from ..db.session import session_scope
from .engine import Recipe, run_full
from .errors import MergeFailure
from .virtual import intermediate_path_for, recipe_for

__all__ = ["MERGE_THREADS", "merge_threads", "rebuild_intermediate", "run_recipe"]

_log = logging.getLogger(__name__)

#: Section 12 budgets the merges on two cores. The workers pin OpenCV to one
#: thread (``jobs/_preload.py``); a merge takes a second for its duration.
MERGE_THREADS = 2


class merge_threads:  # noqa: N801 - used as a context manager, like a function
    """Let OpenCV use :data:`MERGE_THREADS` threads inside the block."""

    def __enter__(self) -> None:
        import cv2

        self._previous = cv2.getNumThreads()
        cv2.setNumThreads(MERGE_THREADS)

    def __exit__(self, *_exc: Any) -> None:
        import cv2

        cv2.setNumThreads(self._previous)


def run_recipe(
    recipe: Recipe, progress: Callable[[float], None] | None = None
) -> tuple[Path, dict, tuple[int, int]]:
    """Run the full merge of ``recipe`` into its intermediate."""
    destination = intermediate_path_for(recipe)
    with merge_threads():
        return run_full(
            recipe, destination, reduced_long_edge=get_settings().proxy_long_edge,
            progress=progress,
        )


def rebuild_intermediate(maker: Any, photo_id: int) -> Path:
    """Regenerate the intermediate of a merged photo; returns its path.

    Raises:
        MergeFailure: the merge cannot be made any more -- a member left the
            source folder -- with the reason.
    """
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        group = session.get(MergeGroup, photo.merge_group_id) if photo else None
        if photo is None or group is None or group.result_photo_id != photo_id:
            raise MergeFailure("la fusione di questa foto non esiste più")
        recipe = recipe_for(session, group)

    _log.info("rigenero l'intermedio della fusione %d (foto %d)", recipe.group_id, photo_id)
    path, _report, _size = run_recipe(recipe)
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is not None:
            photo.intermediate_path = str(path)
    return path

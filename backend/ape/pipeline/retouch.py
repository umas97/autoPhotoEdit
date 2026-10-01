# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``retouch`` stage: the removals, in order, right after the lens (docs/SPEC_rimozione.md 4.1).

Why there: the frame the removals are written in exists from the lens
correction on; the frame is still linear and scene-referred, where section
6.2 puts everything physical; the denoise that follows passes evenly over a
fill and its surroundings, so the grain does not give the patch away; and a
fill depends only on the decode, the lens and the removals before it -- no
slider of the development (white balance, exposure, tone, colour, masks)
makes a saved fill stale.

Each item sees the result of the ones before it. A spot is healed here at
whatever resolution is being rendered (``ops/heal.py``); an erased area is
composed from its saved fill (``ops/erase.py``). An erase whose fill is not
there yet is skipped: the preview shows the photo as it was while the worker
computes, and the export computes the fill before it renders
(``retouch/fills.py``) -- it never exports without a removal.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from .ops.erase import AreaAlpha, area_alpha, erase_into
from .ops.heal import heal_into
from .retouch_params import EraseItem, HealItem

__all__ = ["apply", "area_for", "stage_key"]

_log = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def area_for(name: str, expand: float, feather: float) -> AreaAlpha:
    """The grown, faded area of a painted raster. Cached by name: a name is its content."""
    from .. import masks_store

    return area_alpha(masks_store.load(name), expand, feather)


def stage_key(items: list[Any]) -> list:
    """What the stage's output depends on: every item, fills included."""
    return [item.model_dump(mode="json") for item in items]


def apply(img: np.ndarray, items: list[Any]) -> np.ndarray:
    """Every visible removal applied to ``img``, in order.

    Args:
        img: ``(H, W, 3)`` float32, linear scene-referred Rec.2020, white at
            1.0: the lens-corrected frame. Never written -- it may be the
            decoded frame or a cached stage.
        items: ``EditParams.retouch``.

    Returns:
        ``img`` itself when nothing is visible (a photo with no removals
        renders to the bit as before), otherwise a new frame, identical to
        ``img`` outside the removals.
    """
    from .. import retouch_store

    active = [item for item in items if item.visible and item.opacity > 0.0]
    if not active:
        return img
    out = img.copy()
    for item in active:
        if isinstance(item, HealItem):
            heal_into(out, item)
            continue
        assert isinstance(item, EraseItem)
        if item.fill is None:
            continue
        try:
            patch = retouch_store.load(item.fill)
            area = area_for(item.area, item.expand, item.feather)
        except ValueError as exc:
            # A missing fill or raster: the photo shows without this removal,
            # and the removal's state says why (``retouch/service.py``).
            _log.warning("rimozione %s saltata: %s", item.id, exc)
            continue
        erase_into(out, item, patch, area)
    return out

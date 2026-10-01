# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running the per-pixel stages of a full-resolution render band by band.

The render is bound by memory traffic, not by arithmetic (measured in phase 5):
each NumPy operation of a stage reads and writes a whole frame, 290 MB at
24 MP, and a stage has a dozen of them. For the stages where each output pixel
depends on its own input pixel alone, doing all of them on a band of rows
before moving to the next band gives the same bits and keeps every temporary
in the cache. Measured on the user's looks: the render from 6.2 to 5.0 s on one
core, and none of those stages' temporaries at frame size (phase 8).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["BAND_MIN_PIXELS", "BAND_PIXELS", "PER_PIXEL", "run_banded"]


#: Stages whose every output pixel depends on the same input pixel alone (and
#: on the parameters). Run one after another on a band of rows, they give the
#: same bits as on the whole frame -- and their temporaries, a dozen per stage,
#: are a band's worth of cache instead of a frame's worth of fresh memory.
#: Not the highlight recovery: it learns the colour of the highlights from the
#: whole frame (``highlight_recovery._channel_ratios``).
PER_PIXEL = frozenset({"white_balance", "exposure", "tone", "tone_shaping", "color", "output"})

#: Below this a frame is a proxy, which the per-stage path already handles in
#: the time section 10 asks for; banding is for the full-resolution export.
BAND_MIN_PIXELS = 4_000_000

#: Pixels per band: about 3 MB of float32 RGB per buffer, so that a stage's
#: input, output and temporaries stay in the last-level cache even with
#: several workers sharing it.
BAND_PIXELS = 256 * 1024


Stage = Callable[[np.ndarray, Any], np.ndarray]


def run_banded(img: np.ndarray, ctx: Any, functions: list[Stage]) -> np.ndarray:
    """Run consecutive per-pixel stages band by band into one new frame.

    The input is never written (it may be the decoded frame the caller keeps);
    the output is allocated once, full size, and filled a band at a time.
    """
    height, width = img.shape[:2]
    rows = max(8, BAND_PIXELS // max(width, 1))
    out = np.empty((height, width, 3), dtype=np.float32)
    try:
        for top in range(0, height, rows):
            band = img[top : top + rows]
            # The masks' selections are frame-sized: a stage that blends with
            # one needs to know which of its rows this band is.
            ctx.rows = slice(top, top + rows)
            for function in functions:
                band = function(band, ctx)
            out[top : top + rows] = band
    finally:
        ctx.rows = None
    return out

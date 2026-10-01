# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Empty borders, and the largest crop without them (sections 6.4 and 25.5.5).

Some frames have pixels that hold nothing: the irregular edges of a stitched
panorama, around the frames that did not reach them. The crop proposal then
offers the largest rectangle with no empty pixel -- still a proposal, never
applied -- under the aspect code :data:`BORDERS_ASPECT`, which the analysis
leaves in place instead of replacing it with a composition.

Which pixels are empty is known only to whatever produced them, never guessed
from the picture: in a rendered proxy the crushed shadows in the corners of an
ordinary photo are as black as a panorama's missing wedge (measured on
DSC05634, which the guess cropped). So the producer passes the mask, and this
module only finds the rectangle in it.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["BORDERS_ASPECT", "inner_rectangle", "largest_rectangle"]

#: ``CropProposal.aspect`` of a crop that only removes empty borders.
BORDERS_ASPECT = "borders"

#: Long edge of the mask the rectangle is searched on: the search is quadratic
#: in the width, and a pixel of error at this size is below what shows.
_SEARCH_EDGE = 300


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest axis-aligned rectangle of True pixels, ``(x, y, w, h)``."""
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int64)
    best, found = 0, None
    for row in range(height):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(width + 1):
            current = int(heights[col]) if col < width else 0
            while stack and int(heights[stack[-1]]) >= current:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                area = h * (col - left)
                if area > best:
                    best, found = area, (left, row - h + 1, col - left, h)
            stack.append(col)
    return found


def inner_rectangle(full: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest rectangle of True in ``full``, searched on a reduced copy.

    The result is shrunk by one cell of the reduced grid, so rounding can
    never let an excluded pixel in.
    """
    height, width = full.shape
    factor = min(1.0, _SEARCH_EDGE / max(width, height))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = cv2.resize(full.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) >= 255
    rect = largest_rectangle(small)
    if rect is None:
        return None
    x, y, w, h = rect
    inv = 1.0 / factor
    cx, cy = int(np.ceil((x + 1) * inv)), int(np.ceil((y + 1) * inv))
    cw, ch = int((w - 2) * inv), int((h - 2) * inv)
    if factor == 1.0:
        cx, cy, cw, ch = x, y, w, h
    if cw <= 0 or ch <= 0:
        return None
    return cx, cy, cw, ch

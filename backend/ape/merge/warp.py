# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resampling a full-resolution frame into the reference's geometry, in bands.

A warped copy of a 24 MP frame is another 288 MB; with the reference, the
accumulators and the frame being decoded, that copy is what would push an HDR
merge past the worker budget of section 26. So the warp is done a band of rows
at a time, straight into whatever consumes it, and the full warped frame never
exists.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

__all__ = ["BAND_ROWS", "shrink", "warp_bands", "warp_whole"]

#: Rows per band: 256 rows of a 6000 px frame are 18 MB of float32, which keeps
#: a band, its weights and its temporaries in the last-level cache's
#: neighbourhood without making the per-call overhead of OpenCV visible.
BAND_ROWS = 256


def shrink(image: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    """Area-reduce to ``long_edge``; returns the image and the full/small scale."""
    height, width = image.shape[:2]
    scale = max(width, height) / long_edge
    if scale <= 1.0:
        return image, 1.0
    size = (max(1, round(width / scale)), max(1, round(height / scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return small, width / size[0]


def _band_matrix(matrix: np.ndarray, top: int) -> np.ndarray:
    """``matrix`` for output rows starting at ``top``: output (x, y) is (x, y + top)."""
    shift = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(top)], [0.0, 0.0, 1.0]])
    return matrix @ shift


def warp_bands(
    frame: np.ndarray, matrix: np.ndarray, size: tuple[int, int], *, rows: int = BAND_ROWS
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(top, band)`` of ``frame`` resampled into the output geometry.

    ``matrix`` maps output coordinates to ``frame`` coordinates (the direction
    :mod:`.align` estimates). Pixels that fall outside ``frame`` are NaN, which
    the consumer turns into zero weight: an edge a hand-held frame does not
    cover must not darken the merge.
    """
    width, height = size
    flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    for top in range(0, height, rows):
        count = min(rows, height - top)
        band = cv2.warpPerspective(
            frame, _band_matrix(matrix, top), (width, count), flags=flags,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan, np.nan, np.nan, np.nan),
        )
        yield top, band


def warp_whole(frame: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The same resampling, in one piece, for reduced frames."""
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    return cv2.warpPerspective(
        frame, matrix, size, flags=flags, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(np.nan, np.nan, np.nan, np.nan),
    )

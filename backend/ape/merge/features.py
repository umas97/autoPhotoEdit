# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What merge detection reads of one photo, measured on its embedded preview.

Culling computes these as part of its own analysis (``culling/features.py``).
A project that goes straight to editing never runs that analysis, and its
bracketings, focus stacks and panoramas must still be found (section 25.6: the
merges screen is reachable right after the import), so the proxy job computes
the same few numbers on its own -- a few milliseconds on the camera's preview,
never a decode (section 25.2).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

__all__ = ["SHARP_GRID", "grouping_features", "sharpness_map"]

#: Tiles of the sharpness map, along the long and the short side. Six by four
#: is coarse on purpose: what a focus stack changes from frame to frame is
#: *which part* of the picture is sharp, and a coarse map says that without
#: reacting to texture.
SHARP_GRID = (6, 4)


def sharpness_map(gray: np.ndarray) -> list[float]:
    """Local sharpness per tile, in the preview's upright orientation.

    The log of the mean absolute Laplacian, row by row. Comparable between the
    frames of one sequence -- same scene, same exposure -- which is all it is
    used for; its absolute value means nothing across scenes.
    """
    height, width = gray.shape[:2]
    columns, rows = SHARP_GRID if width >= height else SHARP_GRID[::-1]
    response = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    values = []
    for row in range(rows):
        top, bottom = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            left, right = column * width // columns, (column + 1) * width // columns
            energy = float(response[top:bottom, left:right].mean())
            values.append(round(float(np.log1p(1000.0 * energy)), 3))
    return values


def grouping_features(bgr: np.ndarray, meta: Any, flip: int = 0) -> dict[str, Any]:
    """Signature, camera details and sharpness map: detection's whole input.

    Stored in ``Photo.culling_features`` *without* a version: culling, if the
    user turns it on later, sees no analysis of its own and runs one.
    """
    from ..culling.burst import visual_signature
    from ..culling.features import camera_details
    from ..culling.sharpness import analysis_gray

    return {
        "merge_only": True,
        "signature": visual_signature(bgr),
        "camera": camera_details(meta, flip) if meta is not None else {},
        "sharp_map": sharpness_map(analysis_gray(bgr)),
    }

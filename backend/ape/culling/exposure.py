# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exposure, judged by what a RAW cannot recover (section 7.3).

The preview is the camera's JPEG, so a region that is white in it may still
hold data in the RAW, and a dark frame is usually just a dark frame. Section 7.3
gives the margins -- about 2 EV of highlight and 4 EV of shadow recovery -- and
the penalties computed from these measures (``technical.py``) only start past
them: a photo is overexposed when a large part of it is blown *and* the frame
as a whole is bright, underexposed when its mid-tones sit more than four stops
below a normal rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["ExposureMetrics", "exposure_metrics"]

#: Log-average luminance of a normally rendered frame, as a fraction of white.
#: The 24 A7 III previews of ``tests/fixtures`` average 1.6 stops under the
#: 18 % of a grey card, with a spread of about one stop: a real scene has more
#: shadow than mid-grey in it, and the camera's tone curve renders it so.
_TYPICAL_KEY = 0.18 * 2.0**-1.6

#: A pixel is blown when all three channels are at the top of the JPEG range,
#: crushed when all three are at the bottom. A pixel is *clipped* when any one
#: channel is at the top: on its own that is often just a saturated colour --
#: a red flower -- but over most of a bright frame it is a colour the camera
#: could not hold, which is what overexposing a colourful scene looks like.
_BLOWN_LEVEL = 250
_CLIPPED_LEVEL = 254
_CRUSHED_LEVEL = 4


@dataclass(frozen=True)
class ExposureMetrics:
    #: Stops between this frame's log-average luminance and a normal
    #: rendering's. Negative is darker.
    ev: float
    #: Fraction of the frame white in all three channels, specular points
    #: excluded.
    blown: float
    #: Fraction black in all three channels.
    crushed: float
    #: Fraction with at least one channel at the top of the range.
    clipped: float = 0.0


def _log_luminance_lut() -> np.ndarray:
    """``log(linear(code) + eps)`` for every 8-bit code value.

    The key is computed on the camera's luma rather than on luminance from
    linearised channels: the two differ by a fraction of a stop on saturated
    colours, which is far inside the one-stop spread of normal exposures, and
    the table lookup makes the whole measure a few hundred microseconds.
    """
    code = np.arange(256, dtype=np.float64) / 255.0
    linear = np.where(code <= 0.04045, code / 12.92, ((code + 0.055) / 1.055) ** 2.4)
    return np.log(linear + 1e-4)


_LOG_LUMINANCE = _log_luminance_lut()


def exposure_metrics(bgr: np.ndarray) -> ExposureMetrics:
    """Exposure of a camera JPEG, in the terms of section 7.3.

    Args:
        bgr: 8-bit BGR in sRGB. Any size; a quarter-size preview is plenty,
            since exposure is a property of areas and not of edges.
    """
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(luma.ravel(), minlength=256)
    key = float(np.exp(np.dot(histogram, _LOG_LUMINANCE) / max(int(luma.size), 1)))

    blue, green, red = cv2.split(bgr)
    lowest = cv2.min(cv2.min(blue, green), red)
    highest = cv2.max(cv2.max(blue, green), red)
    blown_mask = (lowest >= _BLOWN_LEVEL).astype(np.uint8)
    # Specular highlights and the sun are blown in every correct exposure; an
    # opening removes anything smaller than a few pixels at this size, so what
    # remains is area -- sky, snow, a window -- which is what data loss means.
    blown_mask = cv2.morphologyEx(blown_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    size = max(int(highest.size), 1)
    crushed = float(np.count_nonzero(highest <= _CRUSHED_LEVEL) / size)
    clipped = float(np.count_nonzero(highest >= _CLIPPED_LEVEL) / size)
    return ExposureMetrics(
        ev=math.log2(max(key, 1e-6) / _TYPICAL_KEY),
        blown=float(blown_mask.mean()),
        crushed=crushed,
        clipped=clipped,
    )

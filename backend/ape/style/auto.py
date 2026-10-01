# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic exposure and white balance, measured on the neutral rendering.

Two numbers every style needs about a photo before it can say anything about
it, and the whole of what the built-in *Neutro automatico* does (section 22):

* the **exposure anchor** -- the gain, in stops, that would put the scene's key
  on middle grey. The style vector stores exposure relative to it
  (``style/vector.py``), and the neutral profile applies it;
* the **automatic white balance** -- a robust grey world on the greyest
  pixels, confined to a plausible neighbourhood of the as-shot illuminant.

Both are measured on the *neutral display image*: the browsing proxy of a photo
in a project, the neutral render of a RAW in a training pair. It is the one
image both paths already have, and measuring the same thing in both is what
lets an offset learned on one be applied to the other. The scene values are
recovered through the inverse of the neutral sigmoid, which is monotonic, so
the key is a scene-referred quantity even though it is read off a JPEG.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..pipeline.colorspace import REC2020_TO_XYZ, XYZ_TO_RGB, OutputSpace
from ..pipeline.ops.tone import sigmoid_response
from ..pipeline.ops.white_balance import adaptation_matrix
from ..pipeline.params import ToneParams

__all__ = ["AutoResult", "KEY_TARGET_EV", "measure"]

#: Where a well-exposed scene's key sits, in stops from middle grey. A scene's
#: log-average luminance is below the 18% card, because most of a picture is
#: darker than its brightest parts. Measured: the camera's own metering puts
#: the key of the neutral rendering at -1.08 (median of the 24 fixtures, 10th
#: to 90th percentile -1.64..-0.66). The anchor therefore corrects a frame
#: towards what the camera does on average, not towards an abstract grey.
KEY_TARGET_EV = -1.1

#: The anchor never asks for more than this. A night frame is dark on purpose.
ANCHOR_LIMIT_EV = 3.0

#: The automatic white balance stays within this of the camera's own, in mired
#: and tint units: about +-750 K around daylight. The camera is usually close,
#: and a grey world that wants to go further is usually looking at a wall of
#: one colour: on the user's stack of logs (DSC05632) even the greyest pixels
#: are warm wood, and at +-40 it cooled the wood to grey.
WB_LIMIT_MIRED = 25.0
WB_LIMIT_TINT = 12.0

#: sRGB display -> Rec.2020 linear.
_SRGB_TO_REC2020 = (
    np.linalg.inv(REC2020_TO_XYZ) @ np.linalg.inv(XYZ_TO_RGB[OutputSpace.SRGB])
).astype(np.float32)
_Y_REC2020 = REC2020_TO_XYZ[1].astype(np.float32)


@dataclass(frozen=True, slots=True)
class AutoResult:
    #: Scene key, stops from middle grey (baseline exposure included).
    key_ev: float
    #: ``KEY_TARGET_EV - key_ev``, limited.
    exposure_anchor_ev: float
    #: Automatic white balance as a shift from as-shot (vector conventions:
    #: positive mired shift is warmer).
    wb_mired_shift: float
    wb_tint_shift: float


@lru_cache(maxsize=4)
def _inverse_sigmoid(tone_json: str) -> tuple[np.ndarray, np.ndarray]:
    ev = np.linspace(-14.0, 10.0, 2401)
    code = sigmoid_response(ev, ToneParams.model_validate_json(tone_json))
    # Strictly increasing where the curve is not flat; flat ends are clipped.
    keep = np.concatenate([[True], np.diff(code) > 1e-7])
    return code[keep], ev[keep]


def _linear(srgb: np.ndarray) -> np.ndarray:
    x = srgb.astype(np.float32)
    if srgb.dtype == np.uint8:
        x /= 255.0
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def _subsample(image: np.ndarray, limit: int = 60_000) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    step = max(1, len(pixels) // limit)
    return pixels[::step]


def _key(display_rec2020: np.ndarray, tone: ToneParams) -> float:
    luma = display_rec2020 @ _Y_REC2020
    code = np.power(np.clip(luma, 0.0, 1.0), 1.0 / 2.4)
    xs, ys = _inverse_sigmoid(tone.model_dump_json())
    ev = np.interp(code, xs, ys)
    # The clipped ends carry no information about the level, only that it is
    # beyond the curve: a trimmed mean, not a plain one.
    low, high = np.percentile(ev, [2.0, 98.0])
    inside = ev[(ev >= low) & (ev <= high)]
    return float(inside.mean()) if len(inside) else 0.0


def _grey_world(display_rec2020: np.ndarray) -> np.ndarray | None:
    """Robust average colour of the pixels that are probably grey.

    A plain grey world believes a forest is a green cast: on the user's woodland
    frames it asked for the full 40 mired every time. Averaging only the
    least chromatic quarter of the usable pixels -- bark, stone, overcast sky,
    the things that *are* nearly grey -- is the "grey pixel" variant, and it
    is what brought those frames back to within a few mired of the camera,
    which had them right.
    """
    luma = display_rec2020 @ _Y_REC2020
    peak = display_rec2020.max(axis=1)
    usable = (luma > 0.02) & (peak < 0.97)
    if usable.sum() < 500:
        return None
    rgb = display_rec2020[usable]
    chroma = rgb / np.maximum(rgb.sum(axis=1, keepdims=True), 1e-6)
    distance = np.linalg.norm(chroma - 1.0 / 3.0, axis=1)
    greyest = distance <= np.percentile(distance, 25.0)
    return rgb[greyest].mean(axis=0)


def _balance(mean: np.ndarray, as_shot_k: float, as_shot_tint: float) -> tuple[float, float]:
    """Gauss-Newton on (mired shift, tint shift) until the average is grey.

    Written out rather than handed to ``scipy.optimize``: the adaptation
    matrices are cached on rounded arguments, and a solver's own tiny
    finite-difference steps fall inside the rounding and see a flat function.
    """
    as_mired = 1e6 / as_shot_k

    def chromaticity(x: np.ndarray) -> np.ndarray:
        target_k = float(np.clip(1e6 / max(as_mired - x[0], 1.0), 1500.0, 25000.0))
        matrix = adaptation_matrix(
            round(target_k, 1),
            round(float(as_shot_tint + x[1]), 2),
            float(as_shot_k),
            float(as_shot_tint),
        )
        out = matrix @ mean
        return out[[0, 2]] / max(float(out.sum()), 1e-9) - 1.0 / 3.0

    x = np.zeros(2)
    steps = (2.0, 1.0)
    limits = np.array([WB_LIMIT_MIRED, WB_LIMIT_TINT])
    for _ in range(5):
        current = chromaticity(x)
        jacobian = np.empty((2, 2))
        for column, step in enumerate(steps):
            probe = x.copy()
            probe[column] += step
            jacobian[:, column] = (chromaticity(probe) - current) / step
        try:
            delta = np.linalg.solve(jacobian, -current)
        except np.linalg.LinAlgError:
            break
        x = np.clip(x + delta, -limits, limits)
        if np.all(np.abs(delta) < 0.1):
            break
    return float(x[0]), float(x[1])


def measure(
    neutral_srgb: np.ndarray,
    *,
    as_shot_temperature_k: float,
    as_shot_tint: float,
    tone: ToneParams | None = None,
    white_balance: bool = True,
) -> AutoResult:
    """Exposure anchor and automatic white balance of one photo.

    Args:
        neutral_srgb: the neutral rendering, ``(H, W, 3)`` sRGB, uint8 or
            float in [0, 1]. Any size.
        as_shot_temperature_k, as_shot_tint: the camera's white balance.
        tone: the sigmoid the image was rendered with; neutral by default.
        white_balance: skip the grey world when only the anchor is wanted.
    """
    tone = tone or ToneParams()
    display = _linear(_subsample(neutral_srgb)) @ _SRGB_TO_REC2020.T
    key = _key(display, tone)
    anchor = float(np.clip(KEY_TARGET_EV - key, -ANCHOR_LIMIT_EV, ANCHOR_LIMIT_EV))
    shift = tint = 0.0
    if white_balance:
        mean = _grey_world(display)
        if mean is not None:
            shift, tint = _balance(mean.astype(np.float64), as_shot_temperature_k, as_shot_tint)
    return AutoResult(key, anchor, shift, tint)

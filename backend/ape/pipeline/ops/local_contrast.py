# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Clarity and local shadows/highlights, through guided filters.

Clarity boosts the difference between each pixel and its neighbourhood. Done
with a Gaussian blur it produces the bright rims around dark subjects that make
the effect look cheap; done with a guided filter the base layer follows the
edges instead of crossing them, and the halo largely disappears.

The radius is relative to the image's long edge, so the same parameter picks out
the same structures on a 1024 px preview and on the full-resolution export --
which is the whole point of the resolution-independence requirement in section
6.2, and what test 1 of section 13 measures.

The adjustment is computed on luminance and applied to RGB as a ratio, so
clarity changes contrast without dragging colour with it.

**Local shadows and highlights** use the same machinery at a larger scale: a
base layer that follows the big regions of the picture (sky, wall, face), and a
gain that depends on the base layer's brightness only. A bright region is
darkened (or a dark one lifted) as a whole, and the texture inside it keeps its
contrast because every pixel of the region gets the same gain. They run before
clarity, so clarity sees the regions already balanced.
"""

from __future__ import annotations

import numpy as np

from ..colorspace import DISPLAY_GAMMA, luminance
from ..filters import guided_filter
from ..params import LocalContrastParams

__all__ = ["apply", "factors", "local_tone_gain", "scale", "tone_base"]

#: Edge threshold, in display-encoded units squared. Detail weaker than 0.05 is
#: treated as texture and boosted; anything stronger reads as an edge and is
#: kept in the base layer, where clarity leaves it alone.
_EPS = 0.05**2

#: Detail gain at full slider travel. Beyond this the effect stops looking like
#: local contrast and starts looking like an HDR filter from 2010.
_MAX_GAIN = 1.2


#: Base layer of the local tone controls: about 80 px on a 2048 proxy, so a
#: face or a window is a region and a brick is detail.
TONE_RADIUS = 0.04
#: Edges stronger than about 0.15 in display code bound a region; weaker ones
#: are texture inside it.
_TONE_EPS = 0.15**2
#: Full-slider gain, in stops, on the darkest shadow or the brightest region.
#: With the linear weights below, anything up to 2.2 keeps the mapping of the
#: base layer monotonic -- a sky can be darkened towards the buildings but
#: never below them -- and 1.5 leaves room.
_TONE_TRAVEL_EV = 1.5
#: Where the two weights start: shadows below 0.5 of display code (from full
#: weight at black), highlights above 0.35 (to full weight at white).
_SHADOW_END = 0.5
_HIGHLIGHT_START = 0.35


def tone_base(luma: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The base layer the local tone controls weigh their gain on."""
    return guided_filter(luma, luma, TONE_RADIUS, _TONE_EPS, shape)


def local_tone_gain(base: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    """Per-pixel gain in display code, from the base layer's brightness."""
    # In place on two buffers of its own: at full resolution each temporary of
    # the plain expressions is a frame's worth of float32. ``base`` is only read.
    ev = base / np.float32(_SHADOW_END)
    np.subtract(1.0, ev, out=ev)
    np.clip(ev, 0.0, 1.0, out=ev)
    ev *= np.float32(shadows)
    w_highlight = base - np.float32(_HIGHLIGHT_START)
    w_highlight /= np.float32(1.0 - _HIGHLIGHT_START)
    np.clip(w_highlight, 0.0, 1.0, out=w_highlight)
    w_highlight *= np.float32(highlights)
    ev += w_highlight
    del w_highlight
    ev *= np.float32(_TONE_TRAVEL_EV)
    # Display code is gamma-encoded: a gain of ``ev`` stops is 2^(ev / gamma).
    ev /= np.float32(DISPLAY_GAMMA)
    return np.exp2(ev, out=ev)


#: Rows scaled at a time: the RGB temporaries stay a band's worth instead of a
#: frame's (290 MB at 24 MP).
_BAND_ROWS = 256


def _bands(height: int) -> list[slice]:
    return [slice(top, top + _BAND_ROWS) for top in range(0, height, _BAND_ROWS)]


def scale(band: np.ndarray, gain: np.ndarray | None, ratio: np.ndarray | None) -> np.ndarray:
    """``clip(clip(band * gain) * ratio)``, the whole operation on a band of rows.

    ``gain`` and ``ratio`` are the matching rows of :func:`factors`. Returns a
    new array; ``band`` is never written.
    """
    out = band
    for factor in (gain, ratio):
        if factor is None:
            continue
        out = np.multiply(out, factor[..., None], out=None if out is band else out)
        np.clip(out, 0.0, 1.0, out=out)
    return out if out is not band else band.copy()


def _clarity_ratio(luma: np.ndarray, params: LocalContrastParams) -> np.ndarray:
    """Per-pixel ratio that boosts the detail of ``luma``. Consumes ``luma``."""
    base = guided_filter(luma, luma, params.radius, _EPS, luma.shape)
    # boosted = clip(luma + (luma - base) * gain), in the base's own buffer.
    np.subtract(luma, base, out=base)
    base *= np.float32(params.clarity * _MAX_GAIN)
    base += luma
    np.clip(base, 0.0, 1.0, out=base)
    # Ratio rather than difference: a difference would shift the colour of dark
    # pixels far more than that of bright ones.
    base /= np.maximum(luma, 1e-4, out=luma)
    return base


def factors(
    img: np.ndarray, params: LocalContrastParams
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The operation as two one-channel maps: the local tone gain, the clarity ratio.

    Both controls scale the three channels of a pixel by the same number, so
    the whole operation is ``clip(clip(img * gain) * ratio)`` (:func:`scale`),
    and the frame-sized arrays it needs are single-channel: a third of what an
    intermediate RGB frame would cost. ``None`` is a control at zero.
    """
    gain = ratio = None
    if abs(params.shadows) > 1e-6 or abs(params.highlights) > 1e-6:
        base = tone_base(luminance(img), img.shape)
        gain = local_tone_gain(base, params.shadows, params.highlights)
        del base
    if abs(params.clarity) >= 1e-6:
        if gain is None:
            luma = luminance(img)
        else:
            # Clarity works on the picture the local tone left: its luminance,
            # computed a band at a time instead of from a toned frame.
            luma = np.empty(img.shape[:2], dtype=np.float32)
            for rows in _bands(img.shape[0]):
                luma[rows] = luminance(scale(img[rows], gain[rows], None))
        ratio = _clarity_ratio(luma, params)
    return gain, ratio


def apply(img: np.ndarray, params: LocalContrastParams) -> np.ndarray:
    """Apply local shadows/highlights and clarity to a display-referred image.

    Args:
        img: ``(H, W, 3)`` float32 in [0, 1], Rec.2020, perceptually encoded.
            Never written.
        params: clarity amount in [-1, 1] and a long-edge-relative radius; the
            local shadows and highlights in [-1, 1].

    Returns:
        Same space and range; ``img`` itself when every control is at zero.
    """
    gain, ratio = factors(img, params)
    if gain is None and ratio is None:
        return img
    out = np.empty(img.shape, dtype=np.float32)
    for rows in _bands(img.shape[0]):
        out[rows] = scale(img[rows], _part(gain, rows), _part(ratio, rows))
    return out


def _part(factor: np.ndarray | None, rows: slice) -> np.ndarray | None:
    return None if factor is None else factor[rows]

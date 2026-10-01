# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local adjustments: what a mask does at each stage of section 6.2.

A mask carries four kinds of adjustment, and each is applied where its global
counterpart is, so a local change obeys the same physics as a global one:

``exposure``
    a gain of ``2 ** (ev * m)`` in linear light, just before the tone mapping --
    the "regolazioni locali su maschere parametriche, in lineare" of the stage
    list;
``tone_shaping``, ``color``, ``local_contrast``
    the global operation run a second time with the mask's settings, and blended
    in by the selection: ``out + m * (local(out) - out)``. A mask at zero is the
    global result to the bit.

The selections are computed once, at the ``masks`` stage, on the frame leaving
the exposure (``ops/masks.py``), and carried to the later stages in the render
context. During a banded full-resolution render the context also says which
rows the current band is (``ctx.rows``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .params import (
    ColorParams,
    ExposureParams,
    LocalContrastParams,
    MaskParams,
    ToneShapingParams,
)

__all__ = [
    "adjustment_key",
    "apply_exposure",
    "blend",
    "blend_local_contrast",
    "compute_selections",
    "selection_key",
]

_NEUTRAL: dict[str, Any] = {
    "exposure": ExposureParams(),
    "tone_shaping": ToneShapingParams(),
    "color": ColorParams(),
    "local_contrast": LocalContrastParams(),
}


def _active(masks: list[MaskParams], section: str) -> list[int]:
    """Indices of the masks whose ``section`` would change anything."""
    neutral = _NEUTRAL[section]
    return [i for i, mask in enumerate(masks) if getattr(mask, section) != neutral]


def selection_key(masks: list[MaskParams]) -> list:
    """What the selections depend on: everything but the adjustments and the name.

    The index is part of it: selections are stored by index, and reordering two
    masks must not hand one the other's selection from a cache.
    """
    return [
        [i, m.kind, m.definition, m.invert, m.opacity]
        for i, m in enumerate(masks)
        if any(getattr(m, section) != neutral for section, neutral in _NEUTRAL.items())
    ]


def adjustment_key(masks: list[MaskParams], section: str) -> list:
    return [[i, getattr(masks[i], section).model_dump()] for i in _active(masks, section)]


def compute_selections(
    img: np.ndarray, masks: list[MaskParams], tone: Any, load: Callable
) -> dict[int, np.ndarray]:
    """The selection of every mask that adjusts something, by index.

    A mask with every adjustment at zero is not evaluated: it cannot change a
    pixel, and a painted one would otherwise be read from disk for nothing.
    """
    from .ops.masks import evaluate

    wanted = {i for section in _NEUTRAL for i in _active(masks, section)}
    return {i: evaluate(img, masks[i], tone, load) for i in sorted(wanted)}


def _rows(selections: dict[int, np.ndarray], index: int, rows: slice | None) -> np.ndarray:
    try:
        selection = selections[index]
    except KeyError as exc:
        # Resuming the chain past the masks stage without having run it: a
        # caller bug, and one that would otherwise render the mask as absent.
        raise RuntimeError("selezioni delle maschere non calcolate per questo render") from exc
    band = selection if rows is None else selection[rows]
    return band.astype(np.float32)[..., None]


def apply_exposure(
    img: np.ndarray,
    masks: list[MaskParams],
    selections: dict[int, np.ndarray],
    rows: slice | None,
) -> np.ndarray:
    """The masks' exposure, as one gain per pixel. Never writes ``img``."""
    active = _active(masks, "exposure")
    if not active:
        return img
    ev = np.zeros(img.shape[:2] + (1,), dtype=np.float32)
    for i in active:
        ev += _rows(selections, i, rows) * np.float32(masks[i].exposure.ev)
    return img * np.exp2(ev)


#: Rows blended at a time: the selection is converted to float32 a slice at a
#: time instead of as a frame-sized temporary.
_BLEND_ROWS = 256


def blend(
    img: np.ndarray,
    masks: list[MaskParams],
    selections: dict[int, np.ndarray],
    rows: slice | None,
    section: str,
    operation: Callable[[np.ndarray, Any], np.ndarray],
) -> np.ndarray:
    """``img`` with each mask's ``section`` applied through its selection.

    Masks apply one after the other, each on the result of the previous one,
    as layers do. ``img`` is never written: it may be the previous stage's
    cached output. The blend happens inside the array the local operation
    returned, ``(local - out) * m + out`` a slice of rows at a time -- at full
    resolution the plain expression cost three frames of temporaries, 870 MB.
    Where ``m`` is 0 the result is ``out`` to the bit.
    """
    out = img
    for i in _active(masks, section):
        local = operation(out, getattr(masks[i], section))
        if local is out:
            continue
        if not local.flags.writeable or np.shares_memory(local, out):
            local = local.copy()
        for top in range(0, out.shape[0], _BLEND_ROWS):
            band = slice(top, top + _BLEND_ROWS)
            within = band
            if rows is not None:
                within = slice(rows.start + top, rows.start + top + _BLEND_ROWS)
            weight = _rows(selections, i, within)[: out.shape[0] - top]
            part = local[band]
            part -= out[band]
            part *= weight
            part += out[band]
        out = local
    return out


def blend_local_contrast(
    img: np.ndarray,
    masks: list[MaskParams],
    selections: dict[int, np.ndarray],
    *,
    owned: bool,
) -> np.ndarray:
    """:func:`blend` for the local contrast, without a frame per mask.

    The local contrast is two one-channel maps (``local_contrast.factors``),
    so the local result is made and blended a band at a time -- by
    construction the same bits as :func:`blend`, which at full resolution held
    the global result, the local one and the operation's intermediate frames
    at once (1.6 GB with two masks, against the 1.5 of section 26). ``owned``
    says ``img`` is this stage's own output and may be written; otherwise the
    result goes to a new frame and ``img`` is left alone.
    """
    from .ops import local_contrast

    out = img
    for i in _active(masks, "local_contrast"):
        gain, ratio = local_contrast.factors(out, masks[i].local_contrast)
        target = out if owned else np.empty(img.shape, dtype=np.float32)
        for top in range(0, out.shape[0], _BLEND_ROWS):
            band = slice(top, top + _BLEND_ROWS)
            part = local_contrast.scale(
                out[band],
                None if gain is None else gain[band],
                None if ratio is None else ratio[band],
            )
            part -= out[band]
            part *= _rows(selections, i, band)
            part += out[band]
            target[band] = part
        # Before the next mask's maps exist, not after.
        del gain, ratio
        out, owned = target, True
    return out

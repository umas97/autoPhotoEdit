# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-dimensional lookup tables for the display-referred tone operations.

Every operation that happens after the tone mapping -- the sigmoid itself, the
shadows/highlights shaping, the tone curve -- is a scalar function of a single
channel value in [0, 1]. Building each one as a LUT buys three things at once:

* **speed**: one table lookup over the image instead of a chain of transcendental
  functions, which is what keeps a full-resolution export inside its budget;
* **composability**: ``compose`` folds a chain of LUTs into a single table, so
  the renderer pays for one pass over the pixels no matter how many tone
  operations are enabled;
* **determinism**: the same parameters always produce the same table, and the
  same table always produces the same pixels (test 4 of docs/SPEC.md section 13).

Composition preserves the order of section 6.2 exactly: ``compose(a, b)`` is
"``a`` first, then ``b``".

**Two table sizes, on purpose.** Curves are *defined* on ``LUT_SIZE`` samples,
which is where composition and interpolation happen and where a few thousand
points are plenty for a smooth curve. Curves are *applied* through a table
resampled to ``APPLY_SIZE``, looked up by nearest neighbour with a single
gather. Interpolating at apply time would mean two gathers over the whole image
and would triple the cost of the pass; resampling the table instead moves that
work onto 65 thousand elements rather than 72 million. The residual error is the
input quantised to 16 bits before a curve whose output is quantised to at most
16 bits anyway.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "APPLY_SIZE",
    "LUT_SIZE",
    "apply_lut",
    "apply_range_lut",
    "compose",
    "identity_lut",
    "is_identity",
    "sample",
]

#: Resolution at which curves are defined and composed.
LUT_SIZE = 4096
#: Resolution at which curves are applied to pixels. One 16-bit code value.
APPLY_SIZE = 65536


def identity_lut(size: int = LUT_SIZE) -> np.ndarray:
    return np.linspace(0.0, 1.0, size, dtype=np.float32)


def is_identity(lut: np.ndarray, tolerance: float = 1e-6) -> bool:
    return bool(np.allclose(lut, identity_lut(lut.shape[0]), atol=tolerance))


def sample(lut: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolated lookup, for small arrays: composing tables, tests, plots."""
    size = lut.shape[0]
    return np.interp(
        np.clip(positions, 0.0, 1.0), np.linspace(0.0, 1.0, size), lut
    ).astype(np.float32, copy=False)


def _resampled(lut: np.ndarray) -> np.ndarray:
    if lut.shape[0] == APPLY_SIZE:
        return lut
    return sample(lut, identity_lut(APPLY_SIZE))


def apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Look ``values`` (clipped to [0, 1]) up in a uniformly sampled ``lut``.

    A table that is the identity short-circuits. That is not only a saving of
    one pass over the image when the tone curve is switched off -- which is the
    default -- it is what makes "a slider at zero changes nothing" literally
    true, instead of true to within the table's quantisation.
    """
    if is_identity(lut):
        return values
    table = _resampled(lut)
    scaled = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False) * (APPLY_SIZE - 1)
    return np.take(table, scaled.astype(np.int32))


def apply_range_lut(
    values: np.ndarray, lut: np.ndarray, low: float, high: float
) -> np.ndarray:
    """Like :func:`apply_lut`, for a table sampled uniformly over ``[low, high]``.

    Used for the scene-side tone table, whose axis is stops rather than [0, 1].
    """
    table = _resampled(lut)
    size = table.shape[0]
    scale = (size - 1) / (high - low)
    # In place after the first subtraction: two full-frame temporaries fewer.
    scaled = values - low
    scaled *= scale
    np.clip(scaled, 0.0, size - 1, out=scaled)
    return np.take(table, scaled.astype(np.int32))


def compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The LUT of ``second(first(x))``.

    Interpolated rather than nearest: composition happens on a few thousand
    elements, where accuracy is free, and errors here would accumulate through
    the chain instead of landing once on a pixel.
    """
    return sample(second, first)

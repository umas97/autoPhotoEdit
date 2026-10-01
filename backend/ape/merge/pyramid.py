# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Laplacian-pyramid fusion of a focus stack, a tile at a time (section 25.4).

At every level of the pyramid each pixel takes the coefficient of the frames
that are sharpest *there*, with a soft transition between winners: the weight
of a frame is its local energy at that level raised to :data:`POWER`,
normalised over the frames. A hard arg-max would switch frames on noise in
flat areas and draw a seam along every change of winner; the soft weight
averages frames that are equally sharp -- which lowers the noise -- and still
gives an edge to the frame that has it in focus.

The weights are normalised with the running maximum of an online softmax, so
the frames are folded in one at a time and a tile costs the same memory for
three frames as for twelve.

Tiles overlap by a margin wide enough for the coarsest level's filters, and
start on multiples of ``2 ** levels`` so that every tile's pyramid samples the
same grid as a pyramid of the whole frame would: the result does not depend on
where the tiles fall (``tests/test_merge_focus.py`` checks it).

Colour space: linear, scene-referred RGB in, the same out. The energies are
measured on luminance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import cv2
import numpy as np

__all__ = ["POWER", "fuse_stack", "levels_for"]

#: Exponent on the local energy. Energy is squared amplitude, so two frames
#: whose detail differs by a factor of 1.4 get weights 16 to 1: decisive where
#: one frame is clearly sharper, an average where they are alike.
POWER = 4.0

#: Energy is averaged over a Gaussian window of this sigma, in samples of each
#: level: wide enough that one noisy coefficient does not decide, narrow
#: enough that the transition between frames stays close to the focus edge.
_ENERGY_SIGMA = 1.5

#: Floor added to the energy before the logarithm, in linear units squared.
#: Below it -- a flat wall, a clear sky -- all frames weigh the same.
_ENERGY_FLOOR = 1e-10

#: Tile margins, in units of the coarsest level's sample spacing: the support
#: of the reduce/expand filters down to that level and of its energy window.
_MARGIN_UNITS = 6

#: Samples held in a tile, as a function of how many megabytes the accumulator
#: and one frame's pyramid may take: about 55 bytes per pixel.
_BYTES_PER_PIXEL = 55


def levels_for(long_edge: int) -> int:
    """Pyramid depth: the coarsest detail band near a hundredth of the frame.

    Defocus blur scales with the frame; 4 levels at a 1024 px preview, 6 at
    24 MP. Below that the coarse levels would still hold blur a frame has in
    focus; above it they hold only the scene's lighting.
    """
    return max(4, min(7, round(math.log2(max(long_edge, 1) / 100.0))))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2627 + rgb[..., 1] * 0.678 + rgb[..., 2] * 0.0593


def _laplacian(image: np.ndarray, levels: int) -> tuple[list[np.ndarray], np.ndarray]:
    bands = []
    current = image
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        bands.append(current - up)
        current = down
    return bands, current


def _reduce_mask(mask: np.ndarray, levels: int) -> list[np.ndarray]:
    """A pixel is valid at a level when everything its sample averages was valid."""
    out = []
    current = mask.astype(np.float32)
    for _ in range(levels):
        out.append(current >= 0.999)
        current = cv2.pyrDown(current)
    out.append(current >= 0.999)
    return out


class _Accumulator:
    """Online-softmax sums of the coefficients of one tile, per level."""

    def __init__(self, shapes: list[tuple[int, int]], top_shape: tuple[int, int]) -> None:
        self.num = [np.zeros((*s, 3), np.float32) for s in shapes]
        self.den = [np.zeros(s, np.float32) for s in shapes]
        self.peak = [np.full(s, -1e30, np.float32) for s in shapes]
        self.top = np.zeros((*top_shape, 3), np.float32)
        self.top_count = np.zeros(top_shape, np.float32)

    def add(self, bands: list[np.ndarray], residual: np.ndarray, valid: list[np.ndarray]) -> None:
        for level, band in enumerate(bands):
            energy = cv2.GaussianBlur(np.square(_luma(band)), (0, 0), _ENERGY_SIGMA)
            score = np.float32(POWER) * np.log(energy + np.float32(_ENERGY_FLOOR))
            score = np.where(valid[level], score, np.float32(-1e30)).astype(np.float32)
            peak = np.maximum(self.peak[level], score)
            keep = np.exp(self.peak[level] - peak)
            weight = np.exp(score - peak)
            self.num[level] *= keep[..., None]
            self.num[level] += weight[..., None] * band
            self.den[level] *= keep
            self.den[level] += weight
            self.peak[level] = peak
        # The residual is the lighting of the scene, the same in every frame
        # of a stack: an average, which also averages its noise.
        count = valid[-1].astype(np.float32)
        self.top += residual * count[..., None]
        self.top_count += count

    def collapse(self) -> np.ndarray:
        image = self.top / np.maximum(self.top_count, 1e-6)[..., None]
        for level in range(len(self.num) - 1, -1, -1):
            band = self.num[level] / np.maximum(self.den[level], 1e-30)[..., None]
            image = cv2.pyrUp(image, dstsize=(band.shape[1], band.shape[0])) + band
        return image


def _tile_core(levels: int, budget_mb: float) -> int:
    """Side of a tile's core: a multiple of the coarsest sample spacing."""
    unit = 2**levels
    side = int(math.sqrt(budget_mb * 1e6 / _BYTES_PER_PIXEL))
    return max(unit, (side - 2 * _MARGIN_UNITS * unit) // unit * unit)


def fuse_stack(
    frames: Sequence[np.ndarray],
    valid: Sequence[np.ndarray | None],
    out: np.ndarray,
    *,
    reference: int,
    levels: int | None = None,
    budget_mb: float = 180.0,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Fuse aligned frames into ``out``.

    Args:
        frames: the frames, all at the reference's geometry, linear RGB, any
            float dtype (memory maps are read a tile at a time). Pixels outside
            a frame's coverage may hold anything.
        valid: per frame, a boolean map of the pixels it covers, or ``None``
            for "all of them" (the reference).
        out: float32 ``(H, W, 3)`` array the result is written into.
        reference: index of the reference frame, which covers everything; its
            pixels stand in for a frame's uncovered ones.
        levels: pyramid depth; :func:`levels_for` the long edge by default.
        budget_mb: memory a tile may use.

    Returns:
        ``out``.
    """
    height, width = out.shape[:2]
    levels = levels if levels is not None else levels_for(max(height, width))
    unit = 2**levels
    margin = _MARGIN_UNITS * unit
    core = _tile_core(levels, budget_mb)
    tiles = [(t, left) for t in range(0, height, core) for left in range(0, width, core)]
    for done, (top, left) in enumerate(tiles):
        t0, l0 = max(0, top - margin), max(0, left - margin)
        t1, l1 = min(height, top + core + margin), min(width, left + core + margin)
        ref_tile = np.asarray(frames[reference][t0:t1, l0:l1], dtype=np.float32)
        accumulator: _Accumulator | None = None
        for index, frame in enumerate(frames):
            if index == reference:
                tile = ref_tile
            else:
                tile = np.asarray(frame[t0:t1, l0:l1], dtype=np.float32)
            mask = valid[index]
            if mask is None:
                mask_tile = np.ones(tile.shape[:2], dtype=bool)
            else:
                mask_tile = np.asarray(mask[t0:t1, l0:l1], dtype=bool)
                tile = np.where(mask_tile[..., None], tile, ref_tile)
            bands, residual = _laplacian(tile, levels)
            if accumulator is None:
                accumulator = _Accumulator([b.shape[:2] for b in bands], residual.shape[:2])
            accumulator.add(bands, residual, _reduce_mask(mask_tile, levels))
            del bands, residual
        assert accumulator is not None
        result = accumulator.collapse()
        # Ringing past a strong edge can dip below zero where the scene is
        # not: nothing darker than black comes out where the reference had none.
        np.maximum(result, np.minimum(ref_tile, 0.0), out=result)
        rows = slice(top - t0, top - t0 + min(core, height - top))
        cols = slice(left - l0, left - l0 + min(core, width - left))
        out[top : top + core, left : left + core] = result[rows, cols]
        if progress is not None:
            progress((done + 1) / len(tiles))
    return out

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A panorama's plan executed at full resolution, one band of rows at a time.

Section 25.5.2: "il warping e il multi-band blending finali sono eseguiti a
piena risoluzione per bande orizzontali, tenendo in RAM solo la banda corrente
più i margini di sovrapposizione". For each band the frames that reach it are
resampled -- only over the band, from only the part of the frame that lands
there -- and blended with Laplacian pyramids weighted by their seam masks, the
multi-band blending of Burt and Adelson that OpenCV's stitcher uses. The core
rows of the band go to the sink (the intermediate on disk) and the band is
dropped.

Bands overlap by margins wide enough for the pyramid's filters and start on
multiples of ``2 ** levels``, so their pyramids sample the grid a pyramid of
the whole panorama would: the result does not depend on the band height
(``tests/test_merge_panorama.py``). Each frame's piece is aligned the same way
along the row.

Colour space: linear, scene-referred RGB in and out; the exposure gains of the
plan are applied as the frames are read.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .pano_estimate import Plan
from .pano_geometry import Camera, backward_map, warp_roi
from .pano_photometry import falloff

__all__ = ["Layout", "compose", "layout", "ram_estimate_mb"]

#: Rows written per band; the bands' margins are added on top.
BAND_ROWS = 256

#: Margin in units of the coarsest level's spacing: the support of the
#: reduce/expand filters of the pyramid down to that level.
_MARGIN_UNITS = 4


@dataclass(frozen=True)
class Layout:
    """Where every frame lands in an output of a given resolution."""

    scale: float
    cameras: list[Camera]
    #: Per frame ``(x, y, w, h)`` relative to the output's top-left corner.
    rects: list[tuple[int, int, int, int]]
    #: The output's top-left corner in warper coordinates, and its size.
    origin: tuple[int, int]
    size: tuple[int, int]


def layout(plan: Plan, frame_size: tuple[int, int], output_scale: float) -> Layout:
    """The plan for frames of ``frame_size`` and an output ``output_scale`` times full size."""
    factor = frame_size[0] / plan.size[0]
    cameras = [c.scaled(factor) for c in plan.cameras]
    scale = plan.scale * factor * output_scale
    raw = [warp_roi(plan.warper, scale, c, frame_size) for c in cameras]
    x0, y0 = min(r[0] for r in raw), min(r[1] for r in raw)
    x1 = max(r[0] + r[2] for r in raw)
    y1 = max(r[1] + r[3] for r in raw)
    rects = [(x - x0, y - y0, w, h) for x, y, w, h in raw]
    return Layout(scale=scale, cameras=cameras, rects=rects, origin=(x0, y0),
                  size=(x1 - x0, y1 - y0))


def _margin(levels: int) -> int:
    return _MARGIN_UNITS * 2**levels


def ram_estimate_mb(size: tuple[int, int], frame_size: tuple[int, int], levels: int) -> float:
    """Peak memory of the composition, in MB: band accumulators, one frame's piece."""
    band = BAND_ROWS + 2 * _margin(levels)
    accumulators = size[0] * band * (12 + 4) * 1.34
    piece = min(size[0], frame_size[0] * 2) * band * (12 * 3 + 4) * 1.34
    source = frame_size[0] * band * 2 * 12
    return (accumulators + piece + source) / 1e6


def _pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    out = [image]
    for _ in range(levels):
        out.append(cv2.pyrDown(out[-1]))
    return out


def _laplacian(gaussian: list[np.ndarray]) -> list[np.ndarray]:
    bands = []
    for level in range(len(gaussian) - 1):
        finer = gaussian[level]
        up = cv2.pyrUp(gaussian[level + 1], dstsize=(finer.shape[1], finer.shape[0]))
        bands.append(finer - up)
    bands.append(gaussian[-1])
    return bands


class _Band:
    """Weighted sums of the frames' pyramids over one band, per level."""

    def __init__(self, height: int, width: int, levels: int) -> None:
        shapes = [(height, width)]
        for _ in range(levels):
            h, w = shapes[-1]
            shapes.append(((h + 1) // 2, (w + 1) // 2))
        self.num = [np.zeros((*s, 3), np.float32) for s in shapes]
        self.den = [np.zeros(s, np.float32) for s in shapes]

    def add(self, piece: np.ndarray, weight: np.ndarray, column: int) -> None:
        levels = len(self.num) - 1
        bands = _laplacian(_pyramid(piece, levels))
        weights = _pyramid(weight, levels)
        for level, (band, w) in enumerate(zip(bands, weights, strict=True)):
            left = column >> level
            cols = slice(left, left + band.shape[1])
            self.num[level][: band.shape[0], cols] += band * w[..., None]
            self.den[level][: band.shape[0], cols] += w

    def collapse(self) -> np.ndarray:
        image = self.num[-1] / np.maximum(self.den[-1], 1e-8)[..., None]
        for level in range(len(self.num) - 2, -1, -1):
            band = self.num[level] / np.maximum(self.den[level], 1e-8)[..., None]
            image = cv2.pyrUp(image, dstsize=(band.shape[1], band.shape[0])) + band
        # Outside every frame: empty, not the blur of the neighbours' edges.
        image[self.den[0] < 1e-3] = 0.0
        return image


def _seam_weight(plan: Plan, index: int, out: Layout, u0: int, v0: int, size: tuple[int, int]):
    """The frame's seam mask over a piece of the output, 0..1."""
    seam = plan.seams[index].astype(np.float32) / 255.0
    seam = cv2.dilate(seam, np.ones((3, 3), np.uint8))
    # Output pixel -> estimation-size warper coordinate -> seam-mask pixel.
    to_est = plan.scale / out.scale
    a = to_est * plan.seam_factor
    corner_x = round(plan.corners[index][0] * plan.seam_factor)
    corner_y = round(plan.corners[index][1] * plan.seam_factor)
    matrix = np.array([
        [a, 0.0, a * (u0 + out.origin[0]) - corner_x],
        [0.0, a, a * (v0 + out.origin[1]) - corner_y],
    ])
    return cv2.warpAffine(seam, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _piece(source, plan: Plan, out: Layout, index: int, rows: slice, cols: slice):
    """Frame ``index`` resampled over output ``rows`` x ``cols``, with its weight."""
    height_src, width_src = source.shape[:2]
    v, u = np.mgrid[rows, cols]
    map_x, map_y = backward_map(
        plan.warper, out.scale, out.cameras[index],
        u.astype(np.float64) + out.origin[0], v.astype(np.float64) + out.origin[1],
    )
    inside = (map_x >= 0) & (map_x <= width_src - 1) & (map_y >= 0) & (map_y <= height_src - 1)
    if not inside.any():
        return None
    xs, ys = map_x[inside], map_y[inside]
    x0, x1 = max(0, int(xs.min()) - 3), min(width_src, int(math.ceil(xs.max())) + 4)
    y0, y1 = max(0, int(ys.min()) - 3), min(height_src, int(math.ceil(ys.max())) + 4)
    region = np.asarray(source[y0:y1, x0:x1], dtype=np.float32) * np.float32(plan.gains[index])
    # Outside the frame the border is replicated: the pyramid near the frame's
    # edge then sees a continuation of the picture, not a black wall.
    pixels = cv2.remap(region, map_x - x0, map_y - y0, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
    if any(plan.vignetting):
        # The lens's falloff, undone where each output pixel was in its frame.
        cx, cy = (width_src - 1) / 2.0, (height_src - 1) / 2.0
        r2 = ((map_x - cx) ** 2 + (map_y - cy) ** 2) / np.float32(cx * cx + cy * cy)
        pixels /= falloff(r2, plan.vignetting)[..., None]
    size = (cols.stop - cols.start, rows.stop - rows.start)
    weight = _seam_weight(plan, index, out, cols.start, rows.start, size) * inside
    return pixels, weight.astype(np.float32)


def compose(
    sources: Sequence[np.ndarray],
    plan: Plan,
    out: Layout,
    sink: Callable[[int, np.ndarray], None],
    *,
    levels: int = 5,
    rows: int = BAND_ROWS,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Write the panorama's rows to ``sink(top, rows)``, top to bottom."""
    width, height = out.size
    unit = 2**levels
    margin = _margin(levels)
    rows = max(unit, rows // unit * unit)
    for top in range(0, height, rows):
        t0, t1 = max(0, top - margin), min(height, top + rows + margin)
        band = _Band(t1 - t0, width, levels)
        for index, (x, y, w, h) in enumerate(out.rects):
            if y >= t1 or y + h <= t0:
                continue
            c0 = max(0, x - margin) // unit * unit
            c1 = min(width, x + w + margin)
            found = _piece(sources[index], plan, out, index, slice(t0, t1), slice(c0, c1))
            if found is not None:
                band.add(*found, column=c0)
        result = band.collapse()
        sink(top, result[top - t0 : top - t0 + min(rows, height - top)])
        if progress is not None:
            progress(min(1.0, (top + rows) / height))

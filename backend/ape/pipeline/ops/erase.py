# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The magic eraser at render time: composing a saved fill (docs/SPEC_rimozione.md 4.3).

A fill invented at two resolutions is two different fills, so it is invented
once, on the proxy, by a worker (``retouch/``), and saved in two layers:

**structure**, the large shapes and colours of the fill, low-passed at a
fixed fraction of the long edge; resampled onto the area here, whatever the
size of the render.

**offsets**, for every pixel of the area the place its texture was taken
from. The fine detail -- grain, micro-texture -- is the *real* photo's at
that place, read at the resolution being rendered: the ratio of each pixel to
its own low-pass, the same low-pass the structure went through. At 1024 px
that ratio is nearly 1 and the structure carries everything; at 24 MP it
carries the grain the proxy never had. Both are the same fill at the scale
they share, which is what test 1 of section 13 measures.

The area is the painted raster, grown by ``expand`` and faded over
``feather`` of its radius, computed at the raster's own resolution and
resampled: the same outline at every size.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ...retouch_store import Patch
from ..retouch_params import EraseItem

__all__ = ["STRUCTURE_SIGMA", "AreaAlpha", "area_alpha", "erase_into"]

#: The low-pass of the structure, in pixels of the patch. The fill's own fine
#: texture is an average of many copied patches, softer than a photograph;
#: below this scale the render uses the real photo's instead.
STRUCTURE_SIGMA = 1.5

#: The narrowest fade, in pixels of the raster, when ``feather`` is 0.
_MIN_EDGE_PX = 1.5

#: Linear-light floor of the detail ratio, and its limits: a ratio is
#: meaningless in the black, and a hot pixel must not print a star.
_EPS = 2e-3
_RATIO_LIMITS = (0.33, 3.0)

#: Rows of the area composed at a time: the gathered detail is a band's worth.
_BAND_ROWS = 256


class AreaAlpha:
    """The weight of the removal over the frame, at the raster's resolution.

    Only the box where it is above 0 is kept: the server caches a few of
    these, and a raster-sized float32 each would be 11 MB for a speck.
    """

    def __init__(
        self, crop: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, int]
    ) -> None:
        #: The weight inside ``bbox``, float32 in [0, 1].
        self.crop = crop
        #: ``(y0, y1, x0, x1)`` in raster pixels: where the weight is above 0.
        self.bbox = bbox
        self._shape = shape

    @property
    def shape(self) -> tuple[int, int]:
        """The raster's ``(Hr, Wr)``."""
        return self._shape

    def window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The weight over a box of the raster, 0 outside the area."""
        out = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=np.float32)
        by0, by1, bx0, bx1 = self.bbox
        iy0, iy1, ix0, ix1 = max(y0, by0), min(y1, by1), max(x0, bx0), min(x1, bx1)
        if iy1 > iy0 and ix1 > ix0:
            out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = self.crop[
                iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0
            ]
        return out

    def full(self) -> np.ndarray:
        h, w = self._shape
        return self.window(0, h, 0, w)

    def normalised_bbox(self) -> tuple[float, float, float, float]:
        y0, y1, x0, x1 = self.bbox
        h, w = self.shape
        return x0 / w, y0 / h, x1 / w, y1 / h


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def area_alpha(raster: np.ndarray, expand: float, feather: float) -> AreaAlpha:
    """The removal's weight at the raster's resolution, grown and faded.

    ``raster`` is the painted area, ``(Hr, Wr)`` uint8, at the frame's aspect.
    """
    h, w = raster.shape
    long = float(max(h, w))
    painted = raster >= 128
    if not painted.any():
        return AreaAlpha(np.zeros((0, 0), dtype=np.float32), (0, 0, 0, 0), (h, w))
    # Distance to the painted area, in raster pixels.
    distance = cv2.distanceTransform((~painted).astype(np.uint8), cv2.DIST_L2, 5)
    grow = expand * long
    # The radius of a disc with the area's surface: a person and a speck of
    # dust fade in proportion to their size, not by the same few pixels.
    radius = math.sqrt(float(painted.sum()) / math.pi)
    width = max(feather * radius, _MIN_EDGE_PX)
    alpha = (1.0 - _smoothstep((distance - np.float32(grow)) / np.float32(width))).astype(
        np.float32
    )
    # The brush's own soft edge counts too, where it reaches further.
    np.maximum(alpha, raster.astype(np.float32) * np.float32(1.0 / 255.0), out=alpha)
    ys, xs = np.nonzero(alpha > 0.0)
    bbox = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    crop = np.ascontiguousarray(alpha[bbox[0] : bbox[1], bbox[2] : bbox[3]])
    crop.setflags(write=False)
    return AreaAlpha(crop, bbox, (h, w))


def _pixel_box(
    bbox: tuple[float, float, float, float], h: int, w: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, int(math.floor(y0 * h))),
        min(h, int(math.ceil(y1 * h))),
        max(0, int(math.floor(x0 * w))),
        min(w, int(math.ceil(x1 * w))),
    )


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    grows = width >= image.shape[1] and height >= image.shape[0]
    interpolation = cv2.INTER_CUBIC if grows else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


def erase_into(frame: np.ndarray, item: EraseItem, patch: Patch, area: AreaAlpha) -> None:
    """Compose one saved fill into ``frame``, in place.

    Args:
        frame: ``(H, W, 3)`` float32, linear scene-referred Rec.2020, white at
            1.0 -- after the lens and the removals before this one. Written
            only where the area's weight is above 0.
        item: the removal, for its opacity.
        patch: its fill (``retouch_store``), made on the proxy.
        area: :func:`area_alpha` of its raster.
    """
    h, w = frame.shape[:2]
    y0, y1, x0, x1 = _pixel_box(patch.bbox, h, w)
    bh, bw = y1 - y0, x1 - x0
    if bh <= 0 or bw <= 0:
        return
    ph, pw = patch.structure.shape[:2]

    # The area's weight over the patch's region, at the render's resolution.
    ah, aw = area.shape
    ry0, ry1, rx0, rx1 = _pixel_box(patch.bbox, ah, aw)
    alpha = _resize(area.window(ry0, ry1, rx0, rx1), bw, bh)
    alpha *= np.float32(item.opacity)
    if not (alpha > 0.0).any():
        return

    structure = _resize(np.ascontiguousarray(patch.structure), bw, bh)
    # Render pixels per patch pixel. The detail below the structure's
    # low-pass is the photo's own, at this resolution.
    scale = bw / pw
    region = frame[y0:y1, x0:x1]
    ratio = None
    sigma = STRUCTURE_SIGMA * scale
    # Below half a pixel the low-pass is the identity and there is no detail
    # the structure does not already hold.
    if sigma >= 0.5:
        low = cv2.GaussianBlur(region, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        # In place: the low-pass becomes the detail ratio.
        np.add(low, np.float32(_EPS), out=low)
        np.divide(np.maximum(region, 0.0) + np.float32(_EPS), low, out=low)
        np.clip(low, *_RATIO_LIMITS, out=low)
        ratio = low
        rows_of = np.minimum((np.arange(bh) * ph) // bh, ph - 1)
        cols_of = np.minimum((np.arange(bw) * pw) // bw, pw - 1)
        # The offsets are in patch pixels; in render pixels they are scaled.
        dy_all = np.rint(patch.offsets[..., 0].astype(np.float32) * np.float32(bh / ph))
        dx_all = np.rint(patch.offsets[..., 1].astype(np.float32) * np.float32(scale))
        cols = np.arange(bw)

    for top in range(0, bh, _BAND_ROWS):
        band = slice(top, min(bh, top + _BAND_ROWS))
        weight = alpha[band]
        touched = weight > 0.0
        if not touched.any():
            continue
        fill = structure[band]
        if ratio is not None:
            prow = rows_of[band]
            dy = dy_all[prow][:, cols_of].astype(np.int64)
            dx = dx_all[prow][:, cols_of].astype(np.int64)
            src_y = np.clip(np.arange(band.start, band.stop)[:, None] + dy, 0, bh - 1)
            src_x = np.clip(cols[None, :] + dx, 0, bw - 1)
            fill = fill * ratio[src_y, src_x]
        part = region[band]
        blended = part + weight[..., None] * (fill - part)
        np.copyto(part, blended, where=touched[..., None])

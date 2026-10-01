# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Masks: the 0..1 selection of each ``MaskParams`` (docs/SPEC.md section 6.3).

Evaluated once per render, on the frame as it leaves the exposure stage --
lens-corrected, upright, uncropped, linear -- which is the frame the definitions
are written in (``mask_defs.py``). What the selections are *for* is
``pipeline/local.py``; this module only says how much of each pixel a mask
covers.

Resolution independence (section 6.2, test 1) holds by construction for the
shapes, which are analytic in normalised coordinates, and for rasters, which
are resampled to the frame. A parametric selection follows the pixels, so it
is softened by a blur whose radius is a fraction of the long edge: without it,
a luminance range would cut along the noise, differently at every size.

Everything is computed a band of rows at a time into one frame-sized array per
mask. At 24 MP a mask is 96 MB of float32 while it is built and half that once
stored as float16 -- a precision of 1/2048, far below what a selection
resolves -- and the temporaries stay a band's worth.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from ..colorspace import DISPLAY_GAMMA, luminance
from ..mask_defs import (
    HueRange,
    LinearDef,
    ParametricDef,
    RadialDef,
    RasterDef,
    SegmentDef,
    ValueRange,
    parse,
)
from ..params import MaskParams, ToneParams
from .tone import display_lightness

__all__ = ["RasterLoader", "SELECTION_DTYPE", "evaluate"]

SELECTION_DTYPE = np.float16

#: Returns the 8-bit raster a definition names (``masks_store.load``).
RasterLoader = Callable[[str], np.ndarray]

#: Rows per band while building a selection: about a megapixel of float32.
_BAND_ROWS = 256

#: Softening of a parametric selection, as a fraction of the long edge: 3 px on
#: a 2048 px proxy, 9 at 24 MP. Enough to stop a range from following sensor
#: noise, small enough that a selection still hugs a skyline.
_PARAMETRIC_SIGMA = 0.0015

#: The narrowest edge a hard-edged shape gets, in pixels: a feather of 0 means
#: "as sharp as the image", not an aliased staircase.
_MIN_EDGE_PX = 1.5


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _Frame:
    """Pixel-centre coordinates of a frame, normalised, one band at a time."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self.long = float(max(height, width))
        self.x = ((np.arange(width, dtype=np.float32) + 0.5) / width)[None, :]
        self._y = (np.arange(height, dtype=np.float32) + 0.5) / height

    def y(self, rows: slice) -> np.ndarray:
        return self._y[rows][:, None]


def _linear(shape: LinearDef, frame: _Frame, rows: slice) -> np.ndarray:
    # In units of the long edge, so the ramp is perpendicular to the drawn line
    # on a frame that is not square.
    sx, sy = frame.width / frame.long, frame.height / frame.long
    ax, ay = (shape.x1 - shape.x0) * sx, (shape.y1 - shape.y0) * sy
    norm = ax * ax + ay * ay
    t = ((frame.x - shape.x0) * np.float32(sx * ax / norm)) + (
        (frame.y(rows) - shape.y0) * np.float32(sy * ay / norm)
    )
    return (1.0 - _smoothstep(t)).astype(np.float32)


def _radial(shape: RadialDef, frame: _Frame, rows: slice) -> np.ndarray:
    sx, sy = frame.width / frame.long, frame.height / frame.long
    dx = (frame.x - shape.cx) * np.float32(sx)
    dy = (frame.y(rows) - shape.cy) * np.float32(sy)
    # Counter-clockwise on screen, where y points down: the major axis runs
    # along (cos, -sin).
    theta = math.radians(shape.angle)
    c, s = np.float32(math.cos(theta)), np.float32(math.sin(theta))
    u = (dx * c - dy * s) / np.float32(shape.rx)
    v = (dx * s + dy * c) / np.float32(shape.ry)
    radius = np.sqrt(u * u + v * v)
    # In units of the normalised radius, which stretches with the axis: the
    # geometric mean keeps the edge near its pixel width on both axes.
    edge = _MIN_EDGE_PX / (math.sqrt(shape.rx * shape.ry) * frame.long)
    feather = max(shape.feather, edge)
    return (1.0 - _smoothstep((radius - (1.0 - feather)) / feather)).astype(np.float32)


def _range(values: np.ndarray, limits: ValueRange) -> np.ndarray:
    feather = max(limits.feather, 1e-3)
    rise = _smoothstep((values - (limits.low - feather)) / feather)
    fall = 1.0 - _smoothstep((values - limits.high) / feather)
    return rise * fall


def _hue_window(hue: np.ndarray, limits: HueRange) -> np.ndarray:
    distance = np.abs(hue - np.float32(limits.center)) % 360.0
    distance = np.minimum(distance, 360.0 - distance)
    half = limits.width / 2.0
    feather = max(limits.feather, 1e-3)
    return 1.0 - _smoothstep((distance - half) / feather)


def _parametric(shape: ParametricDef, band: np.ndarray, tone: ToneParams) -> np.ndarray:
    out = np.ones(band.shape[:2], dtype=np.float32)
    if shape.luminance is not None:
        out *= _range(display_lightness(luminance(band), tone), shape.luminance)
    if shape.hue is not None or shape.saturation is not None:
        # Hue and saturation as the HSL bands read them (``ops/color.py``):
        # HSV of the perceptually encoded working space. Both are ratios, so
        # the exposure of the frame does not move them.
        encoded = np.power(np.maximum(band, 0.0), np.float32(1.0 / DISPLAY_GAMMA))
        hsv = cv2.cvtColor(encoded.astype(np.float32, copy=False), cv2.COLOR_RGB2HSV)
        if shape.saturation is not None:
            out *= _range(hsv[..., 1], shape.saturation)
        if shape.hue is not None:
            # A grey has no hue: fade the hue test out below a little chroma,
            # or every neutral would count as red.
            chroma = np.clip(hsv[..., 1] / np.float32(0.08), 0.0, 1.0)
            out *= _hue_window(hsv[..., 0], shape.hue) * chroma
    return out


def _raster(name: str, frame: _Frame, load: RasterLoader) -> np.ndarray:
    raster = load(name)
    resized = cv2.resize(raster, (frame.width, frame.height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) * np.float32(1.0 / 255.0)


def _component(
    kind: str,
    definition: dict,
    img: np.ndarray,
    frame: _Frame,
    tone: ToneParams,
    load: RasterLoader,
) -> np.ndarray:
    """One selection, frame-sized float32, before invert and opacity."""
    shape = parse(kind, definition)
    if isinstance(shape, SegmentDef):
        if shape.raster is None:
            raise ValueError(
                "la maschera del soggetto non è ancora stata calcolata: "
                "avvia il rilevamento dal pannello Maschere"
            )
        return _raster(shape.raster, frame, load)
    if isinstance(shape, RasterDef):
        return _raster(shape.raster, frame, load)

    out = np.empty((frame.height, frame.width), dtype=np.float32)
    for top in range(0, frame.height, _BAND_ROWS):
        rows = slice(top, min(frame.height, top + _BAND_ROWS))
        if isinstance(shape, LinearDef):
            out[rows] = _linear(shape, frame, rows)
        elif isinstance(shape, RadialDef):
            out[rows] = _radial(shape, frame, rows)
        else:
            out[rows] = _parametric(shape, img[rows], tone)
    if isinstance(shape, ParametricDef):
        sigma = _PARAMETRIC_SIGMA * frame.long
        cv2.GaussianBlur(out, (0, 0), sigma, dst=out, borderType=cv2.BORDER_REFLECT)
    return out


def evaluate(
    img: np.ndarray, mask: MaskParams, tone: ToneParams, load: RasterLoader
) -> np.ndarray:
    """The selection of one mask on one frame.

    Args:
        img: ``(H, W, 3)`` float32, linear scene-referred Rec.2020 -- the frame
            the parametric ranges read. Never written.
        mask: the mask; its ``combine`` steps are applied in order.
        tone: the photo's sigmoid, which says how bright a pixel will look.
        load: where rasters come from.

    Returns:
        ``(H, W)`` :data:`SELECTION_DTYPE` in [0, 1], ``invert`` and
        ``opacity`` included.
    """
    frame = _Frame(img.shape[0], img.shape[1])
    selection = _component(mask.kind, mask.definition, img, frame, tone, load)
    for step in parse(mask.kind, mask.definition).combine:
        other = _component(step.kind, step.definition, img, frame, tone, load)
        if step.invert:
            np.subtract(1.0, other, out=other)
        if step.op == "add":
            np.maximum(selection, other, out=selection)
        elif step.op == "intersect":
            selection *= other
        else:  # subtract
            np.subtract(1.0, other, out=other)
            selection *= other
        del other
    if mask.invert:
        np.subtract(1.0, selection, out=selection)
    if mask.opacity < 1.0:
        selection *= np.float32(mask.opacity)
    np.clip(selection, 0.0, 1.0, out=selection)
    return selection.astype(SELECTION_DTYPE)

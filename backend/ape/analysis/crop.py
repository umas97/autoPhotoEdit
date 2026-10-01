# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The compositional crop, as a *proposal* (section 6.4).

Nothing here is ever applied. The analysis stores the best candidate as a
``CropProposal`` in state ``pending``; the interface draws it over the photo
with "Applica crop proposto", and only that click writes a crop into the
``EditParams``. Two rejections in a row in a project and the program stops
proposing there (``analysis/service``).

The score of a candidate combines, as section 6.4 lists them:

* **coverage** -- the share of the frame's saliency the crop keeps;
* **thirds** -- how close the saliency's centre of mass falls to a power point
  of the crop;
* **cuts** -- saliency along the crop's *new* edges, the ones inside the frame:
  an edge through a subject is the classic bad crop;
* **area** -- what is thrown away, because every pixel is one the photographer
  chose to take.

The uncropped frame is scored the same way and a candidate is proposed only if
it beats it by a clear margin: a proposal that is not obviously better is noise
the user has to reject.

Coordinates are normalised to the image given -- the straightened frame, which
is what ``CropRect`` is relative to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .saliency import saliency_map

__all__ = ["ASPECTS", "CropSuggestion", "propose_crop", "score_rect"]

#: Aspect ratios tried, oriented like the frame: "4:5" on a landscape frame is
#: 5:4. ``original`` is a tighter crop that keeps the camera's own ratio.
ASPECTS: tuple[str, ...] = ("original", "4:5", "16:9", "1:1")

#: Fractions of the largest crop of each ratio that are tried.
_SCALES = (1.0, 0.92, 0.84, 0.76, 0.68)

#: Positions tried along each free axis.
_STEPS = 9

#: Weights of the four terms. Coverage and area are what a crop trades against
#: each other; thirds is the reason to crop at all; a cut through the subject
#: is the one mistake a proposal must never make, hence the heaviest penalty.
_W_COVERAGE, _W_THIRDS, _W_CUT, _W_AREA = 1.0, 0.6, 0.9, 0.4

#: Mean saliency along an edge below which it runs through background.
_CUT_FLOOR = 0.3

#: How much better than the uncropped frame a candidate has to score.
_MARGIN = 0.06

#: Never propose keeping less than this much of the frame.
_MIN_AREA = 0.45

#: Width of the strip along a new edge whose saliency counts as "cut", as a
#: fraction of the crop's short side.
_EDGE_BAND = 0.04


@dataclass(frozen=True, slots=True)
class CropSuggestion:
    x: float
    y: float
    width: float
    height: float
    aspect: str
    score: float
    #: Score of the uncropped frame, for the "why" in the interface.
    baseline: float

    def rect(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def _ratio(aspect: str, frame_ratio: float) -> float:
    if aspect == "original":
        return frame_ratio
    a, b = (float(v) for v in aspect.split(":"))
    ratio = a / b
    # Orient like the frame: a portrait frame gets a portrait crop.
    if (ratio > 1) != (frame_ratio > 1):
        ratio = 1 / ratio
    return ratio


class _Integral:
    """Constant-time sums over rectangles of the saliency map."""

    def __init__(self, values: np.ndarray) -> None:
        self.table = np.pad(values.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        weights = values.astype(np.float64) ** 2
        ys, xs = np.mgrid[0 : values.shape[0], 0 : values.shape[1]]
        self.w = np.pad(weights, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.wx = np.pad(weights * (xs + 0.5), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.wy = np.pad(weights * (ys + 0.5), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    @staticmethod
    def _sum(table: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
        return float(table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0])

    def total(self, x0: int, y0: int, x1: int, y1: int) -> float:
        return self._sum(self.table, x0, y0, x1, y1)

    def centre(self, x0: int, y0: int, x1: int, y1: int) -> tuple[float, float] | None:
        mass = self._sum(self.w, x0, y0, x1, y1)
        if mass <= 1e-9:
            return None
        return self._sum(self.wx, x0, y0, x1, y1) / mass, self._sum(self.wy, x0, y0, x1, y1) / mass


def _thirds(centre: tuple[float, float] | None, x0: int, y0: int, x1: int, y1: int) -> float:
    """1 when the centre of mass sits on a power point of the rectangle, 0 far off."""
    if centre is None:
        return 0.0
    u = (centre[0] - x0) / max(1, x1 - x0)
    v = (centre[1] - y0) / max(1, y1 - y0)
    distance = min(math.hypot(u - a, v - b) for a in (1 / 3, 2 / 3) for b in (1 / 3, 2 / 3))
    return math.exp(-((distance / 0.12) ** 2))


def score_rect(
    integral: _Integral, shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int
) -> float:
    height, width = shape
    total = integral.total(0, 0, width, height)
    if total <= 1e-9:
        return 0.0
    inside = integral.total(x0, y0, x1, y1)
    coverage = inside / total
    area = (x1 - x0) * (y1 - y0) / (width * height)

    band = max(1, round(_EDGE_BAND * min(x1 - x0, y1 - y0)))
    cut = 0.0
    edges = []
    if x0 > 0:
        edges.append((x0, y0, x0 + band, y1))
    if x1 < width:
        edges.append((x1 - band, y0, x1, y1))
    if y0 > 0:
        edges.append((x0, y0, x1, y0 + band))
    if y1 < height:
        edges.append((x0, y1 - band, x1, y1))
    for ex0, ey0, ex1, ey1 in edges:
        mean_edge = integral.total(ex0, ey0, ex1, ey1) / max(1, (ex1 - ex0) * (ey1 - ey0))
        # In absolute terms: the map is normalised to 0..1 over the frame, and
        # background sits at 0.1-0.3 in it while subjects reach 0.6 and more.
        # An edge running through background costs nothing.
        cut = max(cut, min(1.0, max(0.0, mean_edge - _CUT_FLOOR) / (1.0 - _CUT_FLOOR - 0.3)))
    thirds = _thirds(integral.centre(x0, y0, x1, y1), x0, y0, x1, y1)
    return (
        _W_COVERAGE * coverage + _W_THIRDS * thirds - _W_CUT * cut - _W_AREA * (1.0 - area)
    )


def propose_crop(
    image: np.ndarray, aspects: tuple[str, ...] = ASPECTS
) -> CropSuggestion | None:
    """The best crop of ``image``, or ``None`` when the frame is best as it is.

    Args:
        image: the straightened frame, display-referred RGB (uint8 or 0..1).
        aspects: the ratios to try, from :data:`ASPECTS`.
    """
    salience = saliency_map(image)
    height, width = salience.shape
    integral = _Integral(salience)
    baseline = score_rect(integral, (height, width), 0, 0, width, height)
    frame_ratio = width / height

    best: tuple[float, int, int, int, int, str] | None = None
    for aspect in aspects:
        ratio = _ratio(aspect, frame_ratio)
        if ratio >= frame_ratio:
            full_w, full_h = float(width), width / ratio
        else:
            full_w, full_h = height * ratio, float(height)
        for scale in _SCALES:
            cw, ch = round(full_w * scale), round(full_h * scale)
            if cw * ch < _MIN_AREA * width * height or cw < 8 or ch < 8:
                continue
            if aspect == "original" and scale == 1.0:
                continue  # that is the uncropped frame
            for fx in np.linspace(0, 1, _STEPS if width - cw > 1 else 1):
                for fy in np.linspace(0, 1, _STEPS if height - ch > 1 else 1):
                    x0, y0 = round(fx * (width - cw)), round(fy * (height - ch))
                    score = score_rect(integral, (height, width), x0, y0, x0 + cw, y0 + ch)
                    if best is None or score > best[0]:
                        best = (score, x0, y0, cw, ch, aspect)
    if best is None or best[0] < baseline + _MARGIN:
        return None
    score, x0, y0, cw, ch, aspect = best
    return CropSuggestion(
        x=x0 / width,
        y=y0 / height,
        width=min(1.0 - x0 / width, cw / width),
        height=min(1.0 - y0 / height, ch / height),
        aspect=aspect,
        score=round(score, 4),
        baseline=round(baseline, 4),
    )

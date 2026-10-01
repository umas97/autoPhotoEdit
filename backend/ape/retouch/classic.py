# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The magic eraser's classic engine: filling an area from the photo itself.

Exemplar-based image completion, the method of Wexler, Shechtman and Irani
(2007) as Barnes et al. made it practical with PatchMatch (2009): the hole is
filled so that every 7x7 patch of the result looks like some patch of the
known photo around it. Coarse to fine over a pyramid, so the large structure
is settled where the hole is small and the texture where it is sharp; at each
level, alternately, find for every patch its nearest neighbour among the known
ones (PatchMatch: random guesses, propagated between neighbours, refined by a
shrinking random search), then rebuild each hole pixel as the similarity
weighted vote of the patches covering it. OpenCV's Telea inpainting (which on
its own smears anything larger than a few pixels) only seeds the coarsest
level.

Written in NumPy, no new dependency: every step is vectorised over
all the patches of a level at once (``nnf.py``).

The result has two layers (``retouch_store.Patch``): the filled region,
low-passed, and for every hole pixel where its patch came from -- the render
copies the fine detail from there at its own resolution (``ops/erase.py``).
The work happens at a resolution chosen so that the hole has at most
``HOLE_BUDGET`` pixels: a speck is filled at proxy resolution, a person at a
coarser one, and the render's detail transfer makes up the difference.

Deterministic for a given seed: one ``numpy.random.Generator``, seeded by the
seed and ``CLASSIC_VERSION``, and no threads of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from ..pipeline.ops.erase import STRUCTURE_SIGMA, AreaAlpha
from ..retouch_store import Patch
from .nnf import PATCH, Level, onion_peel, patchmatch, vote

__all__ = ["CLASSIC_VERSION", "HOLE_BUDGET", "fill_region", "fill_classic", "hole_region"]

#: Bumped whenever the fill of a given area changes: part of every patch's key,
#: so an old patch is recognised as stale instead of shown.
CLASSIC_VERSION = 1

#: Hole pixels at the finest level. Measured on the proxy: 30 000 fills an
#: area of 5% of the frame in about a second and keeps 20% under six.
HOLE_BUDGET = 30_000

#: The margin around the hole the sources come from, as a fraction of the
#: square root of its surface (about half the diameter of a round area, a few
#: widths of a cable), and at least this many proxy pixels: enough
#: surroundings to find the texture of a small speck, and not the whole photo
#: for a person.
MARGIN = 0.6
MIN_MARGIN_PX = 32

#: The pyramid stops when the hole fits in this many patches.
_COARSEST_PATCHES = 3

#: EM iterations at the coarsest level and at the finest; the levels between
#: interpolate. The coarse levels settle the structure and are cheap.
_EM_COARSE, _EM_FINE = 10, 3
_PM_ITERATIONS = 3

#: How many of the best matches the onion peel chooses among when the seed is
#: not 0: "Altra variante" then changes the structure, not just the grain.
_VARIANT_SPREAD = 3

#: Cube root of linear light: a perceptual encoding for comparing and voting,
#: close to L*, without the logarithm's noise in the blacks.
_EPS = 1e-4


def hole_region(
    frame_shape: tuple[int, int], area: AreaAlpha
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The hole at the frame's resolution and the region around it the fill works in.

    Returns ``(hole, (y0, y1, x0, x1))``: the hole as a boolean mask of the
    region, the region in frame pixels.
    """
    h, w = frame_shape
    alpha = cv2.resize(area.full(), (w, h), interpolation=cv2.INTER_LINEAR)
    hole = alpha > 0.0
    # One pixel more: the render may put a sliver of weight where the
    # nearest-neighbour map of the patch lands just outside.
    hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(hole)
    if ys.size == 0:
        raise ValueError("l'area da rimuovere è vuota")
    margin = max(MIN_MARGIN_PX, int(math.ceil(MARGIN * math.sqrt(float(ys.size)))))
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin)
    return hole[y0:y1, x0:x1], (y0, y1, x0, x1)


def fill_classic(
    frame: np.ndarray,
    area: AreaAlpha,
    seed: int,
    progress: Callable[[float], None] | None = None,
) -> Patch:
    """Fill ``area`` of ``frame`` from its surroundings.

    Args:
        frame: ``(H, W, 3)`` float32, linear Rec.2020, white at 1.0: the proxy
            after the lens and the removals before this one. Never written.
        area: the removal's weight (``ops/erase.area_alpha``); where it is
            above 0 is the hole.
        seed: the item's seed. The same seed, the same fill.
        progress: called with 0..1 as the levels complete.

    Returns:
        The patch: region, structure (linear, low-passed), offsets.
    """
    hole, (y0, y1, x0, x1) = hole_region(frame.shape[:2], area)
    region = frame[y0:y1, x0:x1]
    structure, offsets = fill_region(region, hole, seed, progress)
    h, w = frame.shape[:2]
    return Patch(bbox=(x0 / w, y0 / h, x1 / w, y1 / h), structure=structure, offsets=offsets)


def fill_region(
    region: np.ndarray,
    hole: np.ndarray,
    seed: int,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The completion of ``region`` where ``hole`` is set, at the working resolution.

    Returns ``(structure, offsets)``: the region, filled and low-passed, float32
    linear; and ``(dy, dx)`` int16 from each hole pixel to its source, both at
    the working resolution (the region reduced so the hole has at most
    ``HOLE_BUDGET`` pixels).
    """
    rng = np.random.default_rng([int(seed), CLASSIC_VERSION])
    factor = max(1.0, math.sqrt(float(hole.sum()) / HOLE_BUDGET))
    rh, rw = hole.shape
    wh, ww = max(PATCH * 2, int(round(rh / factor))), max(PATCH * 2, int(round(rw / factor)))
    work = region
    if (wh, ww) != (rh, rw):
        work = cv2.resize(region, (ww, wh), interpolation=cv2.INTER_AREA)
    work_hole = hole if (wh, ww) == (rh, rw) else (
        cv2.resize(hole.astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA) > 0.0
    )
    encoded = np.cbrt(np.maximum(work, 0.0) + np.float32(_EPS)).astype(np.float32)
    ys, xs = np.nonzero(work_hole)
    # The known margin around the hole at the working resolution: the median
    # of the four sides, so that a hole against the frame's edge, with no
    # margin on that side, still has the others to draw from.
    margin = int(np.median([ys.min(), xs.min(), wh - 1 - ys.max(), ww - 1 - xs.max()]))
    filled, source = _complete(encoded, work_hole, max(margin, 1), rng, seed, progress)
    linear = np.maximum(filled.astype(np.float32) ** 3 - np.float32(_EPS), 0.0)
    linear[~work_hole] = work[~work_hole]
    structure = cv2.GaussianBlur(linear, (0, 0), STRUCTURE_SIGMA, borderType=cv2.BORDER_REFLECT)
    ys, xs = np.nonzero(work_hole)
    offsets = np.zeros((wh, ww, 2), dtype=np.int16)
    offsets[ys, xs, 0] = source[ys, xs, 0] - ys
    offsets[ys, xs, 1] = source[ys, xs, 1] - xs
    return structure.astype(np.float32), offsets


# --------------------------------------------------------------------------- #
# The completion
# --------------------------------------------------------------------------- #


def _pyramid(
    image: np.ndarray, hole: np.ndarray, margin: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Levels from fine to coarse: image halved, hole kept wherever any of it falls.

    Down until the hole fits in a few patches, but never so far that the
    known margin around it gets narrower than two patches: below that no
    whole patch is known, and the fill would copy the very object it removes.
    """
    levels = [(image, hole)]
    while True:
        img, h = levels[-1]
        ys, xs = np.nonzero(h)
        extent = max(int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1) if ys.size else 0
        margin //= 2
        if extent <= _COARSEST_PATCHES * PATCH or margin < 2 * PATCH:
            break
        size = (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2))
        smaller = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        smaller_hole = cv2.resize(h.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.0
        levels.append((smaller, smaller_hole))
    return levels


def _match_boundary(level: Level, estimate: np.ndarray) -> np.ndarray:
    """Take away the fill's mismatch with the photo along the hole's edge.

    The fill copies patches from anywhere around the hole, and a sheet of ice
    lit on one side and shaded on the other gives it patches brighter or
    darker than the edge they end up next to: a seam. Poisson blending's
    remedy: the difference between the photo and what the fill *would* put on
    a ring of known pixels around the hole (the vote, extended there) is
    carried smoothly into the hole -- a membrane (``_pull_push``) -- and added.
    """
    kernel = np.ones((PATCH, PATCH), np.uint8)
    grown = cv2.dilate(level.hole.astype(np.uint8), kernel).astype(bool)
    ring = grown & ~level.hole
    if not ring.any():
        return estimate
    virtual = vote(level, estimate, where=grown)
    difference = np.zeros_like(estimate)
    difference[ring] = level.image[ring] - virtual[ring]
    membrane = _pull_push(difference, ring.astype(np.float32))
    corrected = estimate.copy()
    corrected[level.hole] += membrane[level.hole]
    return corrected


def _pull_push(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """A smooth interpolation of ``values`` from where ``weight`` is 1 to everywhere.

    Gortler et al.'s pull-push: averaged down a pyramid where there is data,
    pushed back up where there is none. Every result is a convex combination
    of the data -- no overshoot, which a gradient-extrapolating inpainting has
    (Telea's gave -0.76 in the middle of a boundary going from 0 to 0.1).
    """
    levels = [(values.astype(np.float32), np.clip(weight, 0.0, 1.0).astype(np.float32))]
    while min(levels[-1][1].shape) > 1:
        val, wt = levels[-1]
        size = (max(1, wt.shape[1] // 2), max(1, wt.shape[0] // 2))
        total_w = cv2.resize(wt, size, interpolation=cv2.INTER_AREA)
        total_v = cv2.resize(val * wt[..., None], size, interpolation=cv2.INTER_AREA)
        present = total_w[..., None] > 0
        coarse = np.where(present, total_v / np.maximum(total_w, 1e-12)[..., None], 0.0)
        # Four children with data make a parent with full data.
        full = np.minimum(1.0, total_w * 4.0).astype(np.float32)
        levels.append((coarse.astype(np.float32), full))
    filled = levels[-1][0]
    for val, wt in reversed(levels[:-1]):
        up = cv2.resize(filled, (wt.shape[1], wt.shape[0]), interpolation=cv2.INTER_LINEAR)
        filled = wt[..., None] * val + (1.0 - wt[..., None]) * up
    return filled


def _complete(
    image: np.ndarray,
    hole: np.ndarray,
    margin: int,
    rng: np.random.Generator,
    seed: int,
    progress: Callable[[float], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill ``image`` (encoded) where ``hole``; returns it and the source of each pixel."""
    pyramid = _pyramid(image, hole, margin)
    count = len(pyramid)
    level = previous = None
    estimate = None
    for step, (img, h) in enumerate(reversed(pyramid)):
        level = Level(img, h)
        fraction = step / max(1, count - 1)
        iterations = int(round(_EM_COARSE + (_EM_FINE - _EM_COARSE) * fraction))
        if previous is None:
            estimate, peeled = onion_peel(level, rng, _VARIANT_SPREAD if seed else 1)
            # The peel's own choices start the field; the targets it did not
            # fill (the known band around the hole) start at random.
            level.nnf = level.random_sources(len(level.targets), rng)
            chosen = peeled[level.targets[:, 0], level.targets[:, 1]]
            level.nnf[chosen[:, 0] >= 0] = chosen[chosen[:, 0] >= 0]
            for _ in range(iterations):
                patchmatch(level, estimate, rng, _PM_ITERATIONS)
                estimate = vote(level, estimate)
        else:
            # The coarse matches, doubled, are the fine ones' first guess;
            # where a coarse target did not exist, a random one.
            coarse = level.targets // 2
            coarse[:, 0] = np.minimum(coarse[:, 0], previous.h - 1)
            coarse[:, 1] = np.minimum(coarse[:, 1], previous.w - 1)
            rows = previous.index[coarse[:, 0], coarse[:, 1]]
            guess = level.random_sources(len(level.targets), rng)
            known = rows >= 0
            guess[known] = previous.nnf[rows[known]] * 2 + (level.targets[known] % 2)
            guess[:, 0] = np.clip(guess[:, 0], 0, level.h - 1)
            guess[:, 1] = np.clip(guess[:, 1], 0, level.w - 1)
            bad = ~level.valid[guess[:, 0], guess[:, 1]]
            guess[bad] = level.random_sources(int(bad.sum()), rng)
            level.nnf = guess
            up = cv2.resize(estimate, (level.w, level.h), interpolation=cv2.INTER_LINEAR)
            estimate = img.copy()
            estimate[h] = up[h]
            estimate = vote(level, estimate)
            for _ in range(iterations):
                patchmatch(level, estimate, rng, _PM_ITERATIONS)
                estimate = vote(level, estimate)
        previous = level
        if progress is not None:
            progress((step + 1) / count)
    assert level is not None and estimate is not None
    estimate = _match_boundary(level, estimate)
    source = np.zeros((level.h, level.w, 2), dtype=np.int64)
    source[level.targets[:, 0], level.targets[:, 1]] = level.nnf
    return estimate, source

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nearest-neighbour fields between patches, and the vote that turns one into pixels.

The machinery of ``classic.py``, one pyramid level at a time. Targets are the
patches touching the hole; sources the patches wholly in the known photo. Two
ways of matching them:

``exhaustive``
    every target against every source, as one matrix product. Exact, and
    affordable at the coarsest level, where there are a few hundred targets
    and a couple of thousand sources. It also fills that level "onion peel"
    first: ring after ring from the boundary inwards, each pixel copied from
    the best match of its patch's known part -- the structure then comes from
    the photo rather than from a diffusion that would pull a dark shore into
    a pale sheet of ice.

``patchmatch``
    Barnes et al. (2009), vectorised over every target at once: each tries its
    neighbours' matches shifted by one step, then random guesses in a
    shrinking window. On a descriptor, the 3x3 box mean at nine positions: the
    7x7 patch at half resolution, 27 contiguous numbers to compare.

Matrix products are in float64 and reduce along the patch, which OpenBLAS
does not split between threads: the same inputs give the same bits whatever
the thread count, which the fill's determinism rests on.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["PATCH", "Level", "exhaustive", "onion_peel", "patchmatch", "vote"]

#: Patch side. Seven pixels carry a texture's grain and a little of its
#: structure; five follows noise, nine blurs the vote.
PATCH = 7
_R = PATCH // 2
_OFFSETS = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]


class Level:
    """One level of the pyramid: where targets and sources are, and the field."""

    def __init__(self, image: np.ndarray, hole: np.ndarray) -> None:
        self.image = image
        self.hole = hole
        self.h, self.w = hole.shape
        kernel = np.ones((PATCH, PATCH), np.uint8)
        known = (~hole).astype(np.uint8)
        valid = cv2.erode(known, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if valid.sum() < 16:
            # Almost no whole patch is known: accept sources whose patch
            # overlaps the hole or the border, as long as their centre is known.
            valid = ~hole
        if not valid.any():
            raise ValueError("intorno troppo piccolo per riempire l'area: allarga la zona")
        self.valid = valid
        self.sources = np.argwhere(valid)
        grown = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)
        self.targets = np.argwhere(grown)
        self.index = np.full((self.h, self.w), -1, dtype=np.int64)
        self.index[self.targets[:, 0], self.targets[:, 1]] = np.arange(len(self.targets))
        self.nnf = np.zeros_like(self.targets)
        self.distance = np.zeros(len(self.targets), dtype=np.float32)

    def random_sources(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return self.sources[rng.integers(0, len(self.sources), size=count)]


# --------------------------------------------------------------------------- #
# Exhaustive matching, for the coarsest level
# --------------------------------------------------------------------------- #


def _patches(image: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``(n, PATCH * PATCH * C)`` float64: the patches around ``centres``, zero-padded."""
    padded = np.pad(image, ((_R, _R), (_R, _R)) + ((0, 0),) * (image.ndim - 2))
    ys, xs = centres[:, 0], centres[:, 1]
    columns = [padded[ys + _R + dy, xs + _R + dx] for dy in range(-_R, _R + 1)
               for dx in range(-_R, _R + 1)]
    stacked = np.stack(columns, axis=1).astype(np.float64)
    return stacked.reshape(len(centres), -1)


def _masked_distances(values: np.ndarray, mask: np.ndarray, sources: np.ndarray,
                      sources_sq: np.ndarray) -> np.ndarray:
    """Mean squared difference over the known part of each target patch, to every source."""
    weighted = values * mask
    d = (weighted * values).sum(axis=1)[:, None] - 2.0 * weighted @ sources.T + mask @ sources_sq.T
    return d / np.maximum(mask.sum(axis=1), 1.0)[:, None]


def onion_peel(
    level: Level, rng: np.random.Generator, spread: int
) -> tuple[np.ndarray, np.ndarray]:
    """The coarsest estimate: the hole filled ring by ring from its edge inwards.

    Each pixel of a ring takes the centre of the source patch that best
    matches its patch's known part. ``spread`` above 1 picks at random among
    that many best matches: the seed's say in the structure of the fill.

    Returns the estimate and, for each hole pixel, the source it was copied
    from (``-1`` elsewhere).
    """
    image = level.image.copy()
    source = np.full((level.h, level.w, 2), -1, dtype=np.int64)
    filled = ~level.hole
    sources = _patches(level.image, level.sources)
    sources_sq = sources * sources
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    channels = image.shape[2]
    while not filled.all():
        ring = (cv2.dilate(filled.astype(np.uint8), cross) > 0) & ~filled
        targets = np.argwhere(ring)
        if targets.size == 0:
            break
        values = _patches(image, targets)
        mask = np.repeat(_patches(filled.astype(np.float32), targets), channels, axis=1)
        d = _masked_distances(values, mask, sources, sources_sq)
        choice = np.argmin(d, axis=1)
        if spread > 1 and d.shape[1] > spread:
            best = np.argpartition(d, spread, axis=1)[:, :spread]
            pick = rng.integers(0, spread, size=len(targets))
            choice = best[np.arange(len(targets)), pick]
        chosen = level.sources[choice]
        image[targets[:, 0], targets[:, 1]] = level.image[chosen[:, 0], chosen[:, 1]]
        source[targets[:, 0], targets[:, 1]] = chosen
        filled[targets[:, 0], targets[:, 1]] = True
    return image, source


def exhaustive(level: Level, estimate: np.ndarray) -> None:
    """The exact nearest source of every target, on the whole patches of ``estimate``."""
    sources = _patches(estimate, level.sources)
    targets = _patches(estimate, level.targets)
    d = (targets * targets).sum(axis=1)[:, None] - 2.0 * targets @ sources.T
    d += (sources * sources).sum(axis=1)[None, :]
    choice = np.argmin(d, axis=1)
    level.nnf = level.sources[choice]
    level.distance = np.maximum(d[np.arange(len(choice)), choice], 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# PatchMatch, for the finer levels
# --------------------------------------------------------------------------- #


def descriptors(image: np.ndarray) -> np.ndarray:
    """``(h, w, 27)``: the 3x3 box means at nine positions around each pixel."""
    boxed = cv2.blur(image, (3, 3), borderType=cv2.BORDER_REFLECT)
    pad = 2
    padded = cv2.copyMakeBorder(boxed, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = image.shape[:2]
    out = np.empty((h, w, 27), dtype=np.float32)
    for k, (dy, dx) in enumerate(_OFFSETS):
        out[..., 3 * k : 3 * k + 3] = padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
    return out


def _distance(desc_t: np.ndarray, desc: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    diff = desc[candidates[:, 0], candidates[:, 1]] - desc_t
    return np.einsum("ij,ij->i", diff, diff)


def _try(level: Level, desc_t: np.ndarray, desc: np.ndarray, rows: np.ndarray,
         candidates: np.ndarray) -> None:
    """Replace the match of ``rows`` by ``candidates`` where they are valid and closer."""
    inside = (
        (candidates[:, 0] >= 0) & (candidates[:, 0] < level.h)
        & (candidates[:, 1] >= 0) & (candidates[:, 1] < level.w)
    )
    rows, candidates = rows[inside], candidates[inside]
    ok = level.valid[candidates[:, 0], candidates[:, 1]]
    rows, candidates = rows[ok], candidates[ok]
    if rows.size == 0:
        return
    d = _distance(desc_t[rows], desc, candidates)
    better = d < level.distance[rows]
    level.nnf[rows[better]] = candidates[better]
    level.distance[rows[better]] = d[better]


def patchmatch(level: Level, estimate: np.ndarray, rng: np.random.Generator,
               iterations: int) -> None:
    """Improve ``level.nnf`` for the patches of ``estimate``."""
    desc = descriptors(estimate)
    targets = level.targets
    desc_t = desc[targets[:, 0], targets[:, 1]]
    level.distance = _distance(desc_t, desc, level.nnf).astype(np.float32)
    everyone = np.arange(len(targets))
    for iteration in range(iterations):
        steps = [(0, 1), (1, 0)] if iteration % 2 == 0 else [(0, -1), (-1, 0)]
        for dy, dx in steps:
            # A few hops per direction: each target tries its neighbour's
            # match shifted by the step between them, and a good match
            # travels along a coherent region.
            for _hop in range(4):
                ny, nx = targets[:, 0] - dy, targets[:, 1] - dx
                inside = (ny >= 0) & (ny < level.h) & (nx >= 0) & (nx < level.w)
                neighbour = np.full(len(targets), -1, dtype=np.int64)
                neighbour[inside] = level.index[ny[inside], nx[inside]]
                rows = np.flatnonzero(neighbour >= 0)
                candidates = level.nnf[neighbour[rows]] + np.array([dy, dx])
                _try(level, desc_t, desc, rows, candidates)
        radius = max(level.h, level.w)
        while radius >= 1:
            jump = rng.integers(-radius, radius + 1, size=(len(targets), 2))
            _try(level, desc_t, desc, everyone, level.nnf + jump)
            radius //= 2


# --------------------------------------------------------------------------- #
# The vote
# --------------------------------------------------------------------------- #


def vote(level: Level, estimate: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
    """Every hole pixel as the similarity-weighted mean of the patches covering it.

    ``where`` votes other pixels too -- a ring of known ones around the hole,
    to see what the fill would have put there -- and leaves them unforced.
    """
    region = level.hole if where is None else where
    targets, nnf = level.targets, level.nnf
    finite = np.isfinite(level.distance)
    scale = float(np.percentile(level.distance[finite], 75)) if finite.any() else 1.0
    weight = np.exp(-level.distance / (2.0 * max(scale, 1e-12))).astype(np.float64)
    weight[~finite] = 0.0
    size = level.h * level.w
    total = np.zeros((size, 3), dtype=np.float64)
    wsum = np.zeros(size, dtype=np.float64)
    for oy in range(-_R, _R + 1):
        for ox in range(-_R, _R + 1):
            py, px = targets[:, 0] + oy, targets[:, 1] + ox
            sy, sx = nnf[:, 0] + oy, nnf[:, 1] + ox
            keep = (
                (py >= 0) & (py < level.h) & (px >= 0) & (px < level.w)
                & (sy >= 0) & (sy < level.h) & (sx >= 0) & (sx < level.w)
            )
            keep &= region[np.clip(py, 0, level.h - 1), np.clip(px, 0, level.w - 1)]
            lin = py[keep] * level.w + px[keep]
            w = weight[keep]
            values = estimate[sy[keep], sx[keep]]
            wsum += np.bincount(lin, weights=w, minlength=size)
            for c in range(3):
                total[:, c] += np.bincount(lin, weights=w * values[:, c], minlength=size)
    out = estimate.copy()
    flat = out.reshape(-1, 3)
    inside = region.reshape(-1) & (wsum > 0)
    flat[inside] = (total[inside] / wsum[inside, None]).astype(np.float32)
    if where is None:
        # A known pixel is the photo, always.
        out[~level.hole] = level.image[~level.hole]
    return out

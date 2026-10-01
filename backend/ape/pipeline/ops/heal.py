# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spot healing: a disc of the photo covered with a disc from nearby (docs/SPEC_rimozione.md 4.2).

The texture is the source's, copied by a whole-pixel shift so it keeps every
bit of its grain. The colour and brightness are the destination's: the ratio
between the two, measured on a ring just outside the disc, is carried into
the disc as a *harmonic* function -- the membrane of Poisson image editing
(Pérez et al. 2003), which is what "seamless cloning" solves for. On a disc the
harmonic extension of a boundary function has a closed form: its Fourier
series with each order ``n`` scaled by ``(r / R) ** n``. No solver, no grid,
and nothing that depends on the pixel count -- so the preview at 1024 px and
the export at 24 MP agree by construction (test 1 of section 13).

The boundary is read low-pass: the ring is cut into angular sectors and each
sector gives one mean, robust to the odd outlier. That is the "equivalent a
bassa frequenza" of the prompt, and what stops a hair crossing the ring from
printing a streak across the spot. Ratios, not differences, because the frame
is linear light: a source half as bright as the destination has half its
grain as well, and a ratio keeps the two in proportion.

``find_source`` is the automatic choice of the source. It runs once, on the
proxy, when the spot is made; the result is saved in the parameters and the
render never searches.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from ..retouch_params import HealItem

__all__ = ["RING", "find_source", "heal_into"]

#: The ring outside the disc where the destination's colour is read, as a
#: fraction of the radius. Wide enough to average a few hundred pixels on a
#: spot of 20 px, narrow enough to stay on the same surface as the spot.
RING = 0.3

#: Orders of the boundary's Fourier series. Six follow a gradient across the
#: spot and the curve of a cheek; more would follow the noise of the ring.
_ORDERS = 6

#: The narrowest fade, in pixels, when the feather is 0: the edge of the spot
#: is then as sharp as the image, not an aliased staircase.
_MIN_EDGE_PX = 1.5

#: Linear-light floor of the ratios: about 13 stops below white, under the
#: noise floor of any sensor, so a black pixel does not make an infinite ratio.
_EPS = 1e-4


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box(
    frame_h: int, frame_w: int, cx: float, cy: float, reach: float
) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(cx - reach)))
    x1 = min(frame_w, int(math.ceil(cx + reach)))
    y0 = max(0, int(math.floor(cy - reach)))
    y1 = min(frame_h, int(math.ceil(cy + reach)))
    return y0, y1, x0, x1


def _sector_means(values: np.ndarray, angle: np.ndarray, sectors: int) -> np.ndarray:
    """Robust mean of ``values`` (n, 3) in each of ``sectors`` angular sectors."""
    median = np.median(values, axis=0)
    spread = np.median(np.abs(values - median), axis=0) * 1.4826 + 1e-6
    # A hair or a speck in the ring is a few pixels far from the rest:
    # clipped to three spreads, it moves its sector's mean a little instead of a lot.
    clipped = np.clip(values, median - 3.0 * spread, median + 3.0 * spread)
    index = np.minimum((angle / (2.0 * math.pi) * sectors).astype(np.int64), sectors - 1)
    count = np.bincount(index, minlength=sectors).astype(np.float64)
    means = np.empty((sectors, 3), dtype=np.float64)
    for channel in range(3):
        sums = np.bincount(index, weights=clipped[:, channel], minlength=sectors)
        means[:, channel] = np.where(count > 0, sums / np.maximum(count, 1.0), np.nan)
    # Sectors outside the frame, or empty on a tiny spot: their neighbours'
    # value, going round the circle, or the ring's median if nothing is left.
    empty = ~np.isfinite(means[:, 0])
    if empty.all():
        return np.broadcast_to(median, (sectors, 3)).copy()
    if empty.any():
        filled = np.flatnonzero(~empty)
        for k in np.flatnonzero(empty):
            gaps = np.abs(filled - k)
            nearest = filled[np.argmin(np.minimum(gaps, sectors - gaps))]
            means[k] = means[nearest]
    return means


def _membrane(boundary: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The harmonic function on the unit disc with ``boundary`` on its circle.

    ``boundary`` is (sectors, 3), sampled at the sectors' centres; ``z`` the
    complex position of each pixel, ``|z| <= 1`` inside. Returns ``z.shape + (3,)``.
    """
    sectors = boundary.shape[0]
    orders = min(_ORDERS, (sectors - 1) // 2)
    theta = (np.arange(sectors) + 0.5) * (2.0 * math.pi / sectors)
    out = np.empty(z.shape + (3,), dtype=np.float32)
    for channel in range(3):
        g = boundary[:, channel]
        # c_n = a_n - i b_n, so that u = Re(sum c_n z^n): Horner in z.
        coeffs = [complex(g.mean())]
        for n in range(1, orders + 1):
            a = 2.0 / sectors * float(np.sum(g * np.cos(n * theta)))
            b = 2.0 / sectors * float(np.sum(g * np.sin(n * theta)))
            coeffs.append(complex(a, -b))
        acc = np.full(z.shape, coeffs[-1], dtype=np.complex64)
        for c in reversed(coeffs[:-1]):
            acc = acc * z + np.complex64(c)
        out[..., channel] = acc.real
    return out


def heal_into(frame: np.ndarray, item: HealItem) -> None:
    """Heal one spot of ``frame``, in place.

    Args:
        frame: ``(H, W, 3)`` float32, linear scene-referred Rec.2020, white at
            1.0 -- the frame after the lens correction and the removals before
            this one. Written only inside the spot's disc.
        item: the spot. Coordinates in the frame, radius in long edges.

    Every pixel beyond the disc keeps its bits (test 2 of the prompt).
    """
    h, w = frame.shape[:2]
    long = float(max(h, w))
    r = item.radius * long
    cx, cy = item.cx * w, item.cy * h
    # A whole-pixel shift: the copied grain stays grain. The half pixel of
    # rounding is below anything the eye resolves at the scale of a spot.
    dx = int(round((item.sx - item.cx) * w))
    dy = int(round((item.sy - item.cy) * h))
    y0, y1, x0, x1 = _box(h, w, cx, cy, r * (1.0 + RING))
    if y1 <= y0 or x1 <= x0:
        return

    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5 - np.float32(cy))[:, None]
    xs = (np.arange(x0, x1, dtype=np.float32) + 0.5 - np.float32(cx))[None, :]
    rho = np.sqrt(xs * xs + ys * ys) / np.float32(r)
    feather = max(item.feather, _MIN_EDGE_PX / max(r, 1e-6))
    alpha = (1.0 - _smoothstep((rho - (1.0 - feather)) / feather)) * np.float32(item.opacity)
    inside = alpha > 0.0
    if not inside.any():
        return

    rows = np.clip(np.arange(y0, y1) + dy, 0, h - 1)
    cols = np.clip(np.arange(x0, x1) + dx, 0, w - 1)
    # Copies: the source may overlap the box being written.
    dst = frame[y0:y1, x0:x1].copy()
    src = frame[rows[:, None], cols[None, :]]

    ring = (rho >= 1.0) & (rho <= 1.0 + RING)
    if ring.sum() >= 4:
        ratio = np.log(np.maximum(dst[ring], 0.0) + _EPS)
        ratio -= np.log(np.maximum(src[ring], 0.0) + _EPS)
        ring_y = np.broadcast_to(ys, rho.shape)[ring]
        angle = np.arctan2(ring_y, np.broadcast_to(xs, rho.shape)[ring])
        angle = np.mod(angle, 2.0 * math.pi)
        # About one sector per 4 px of circumference: each keeps a dozen
        # pixels of ring on the smallest spots, and 32 is plenty on a large one.
        sectors = int(np.clip(round(2.0 * math.pi * r / 4.0), 4, 32))
        boundary = _sector_means(ratio.astype(np.float64), angle, sectors)
        z = (xs + 1j * ys).astype(np.complex64) / np.complex64(r)
        # Outside the unit circle alpha is 0; clamping keeps the series tame there.
        z = np.where(rho > 1.0, z / np.maximum(rho, 1.0), z)
        gain = np.exp(_membrane(boundary, z))
    else:
        gain = np.ones_like(dst)

    healed = src * gain
    weight = alpha[..., None]
    blended = dst + weight * (healed - dst)
    np.copyto(frame[y0:y1, x0:x1], blended, where=inside[..., None])


# -- the automatic source -----------------------------------------------------

#: The spot is looked at with this radius in pixels, whatever its real size:
#: enough to tell skin from an eyebrow, and a search that costs the same on a
#: speck and on a cheek.
_SEARCH_RADIUS_PX = 10.0

#: How far a source may be, in radii. Lightroom's stays close too: the further
#: away, the likelier the light and the surface have changed.
_SEARCH_RADII = 7.0

#: The ring compared around destination and candidate, outside the disc, as a
#: fraction of the radius. Wider than ``RING``: it says what surface the spot
#: sits on, not just its colour at the edge.
_MATCH_RING = 0.8

#: Weight of the inside terms (spread and level of the candidate's disc against
#: the spot's ring) against the ring's squared log difference, and of the
#: distance, per radius squared. The ring's term on a smooth surface is around
#: 1e-3 (log units squared); a source seven radii away pays ten times that, so
#: the nearest clean place wins and only a clearly better one further away
#: beats it. Measured on the user's files: at 0.002 the search always took the
#: nearest place allowed, whatever was there; at 2e-5 it wandered onto the
#: crease of a smile (DSC05633). At 2e-4, with the level term, the five specks
#: of dust of DSC06396 and the two marks of DSC05633 heal clean.
_TEXTURE_WEIGHT = 1.0
_DISTANCE_WEIGHT = 2e-4


def _log_luma_rgb(img: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(img, 0.0) + np.float32(_EPS)).astype(np.float32)


def find_source(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    avoid: Iterable[tuple[float, float]] = (),
) -> tuple[float, float]:
    """The most similar nearby disc to cover a spot with, destination excluded.

    Args:
        frame: ``(H, W, 3)`` float32, linear Rec.2020 -- the proxy after the
            lens and the removals before this one. Never written.
        cx, cy, radius: the spot, in frame coordinates and long edges.
        avoid: sources already tried ("Scegli un'altra sorgente"): nothing
            within a radius of one of them is proposed again.

    Returns:
        ``(sx, sy)`` in frame coordinates.

    The score of a candidate is how alike the rings around it and around the
    spot are (same surface, same light), plus how alike its inside is to the
    spot's surroundings (no pore, edge or speck of its own), plus a little for
    distance. The destination and its ring are excluded, and so is any
    candidate whose disc would leave the frame.
    """
    h, w = frame.shape[:2]
    long = float(max(h, w))
    r_px = radius * long
    scale = min(1.0, _SEARCH_RADIUS_PX / max(r_px, 1e-6))
    r = r_px * scale
    outer = r * (1.0 + _MATCH_RING)
    reach = _SEARCH_RADII * r_px + r_px * (1.0 + _MATCH_RING)

    y0, y1, x0, x1 = _box(h, w, cx * w, cy * h, reach)
    window = frame[y0:y1, x0:x1]
    size = (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale))))
    small = cv2.resize(window, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else window
    encoded = _log_luma_rgb(small)
    sh, sw = encoded.shape[:2]
    # The spot's position in the reduced window.
    px = (cx * w - x0) * (sw / (x1 - x0))
    py = (cy * h - y0) * (sh / (y1 - y0))

    half = int(math.ceil(outer))
    side = 2 * half + 1
    if sh < side + 2 or sw < side + 2:
        return _fallback(cx, cy, radius, h, w)
    ty = int(np.clip(round(py - 0.5), half, sh - 1 - half))
    tx = int(np.clip(round(px - 0.5), half, sw - 1 - half))
    template = encoded[ty - half : ty + half + 1, tx - half : tx + half + 1]
    offs = np.arange(-half, half + 1, dtype=np.float32)
    rho = np.sqrt(offs[None, :] ** 2 + offs[:, None] ** 2) / np.float32(r)
    ring = ((rho >= 1.0) & (rho <= 1.0 + _MATCH_RING)).astype(np.float32)
    ring_n = float(ring.sum())
    if ring_n < 4:
        return _fallback(cx, cy, radius, h, w)

    cost = np.zeros((sh - side + 1, sw - side + 1), dtype=np.float32)
    for channel in range(3):
        cost += cv2.matchTemplate(
            np.ascontiguousarray(encoded[..., channel]),
            np.ascontiguousarray(template[..., channel]),
            cv2.TM_SQDIFF,
            mask=ring,
        )
    cost /= np.float32(ring_n * 3.0)

    # Texture inside a candidate against the texture around the spot, on the
    # log luminance: the standard deviation over a box the size of the disc.
    luma = encoded.mean(axis=2)
    box = max(1, int(round(2.0 * r * 0.8)) | 1)
    mean = cv2.blur(luma, (box, box), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(luma * luma, (box, box), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    ring_luma = template.mean(axis=2)[ring > 0]
    around = float(np.std(ring_luma))
    inner = (slice(half, half + cost.shape[0]), slice(half, half + cost.shape[1]))
    texture = (std[inner] - around) ** 2
    # And its brightness: a crease or a mole inside a candidate is darker than
    # the skin around the spot even when its spread is not larger.
    level = (mean[inner] - float(np.mean(ring_luma))) ** 2
    cost += np.float32(_TEXTURE_WEIGHT) * (texture + level)

    # Candidate centres in the reduced window, and back in frame pixels.
    cy_s = np.arange(cost.shape[0], dtype=np.float32)[:, None] + half + 0.5
    cx_s = np.arange(cost.shape[1], dtype=np.float32)[None, :] + half + 0.5
    fx = x0 + cx_s * ((x1 - x0) / sw)
    fy = y0 + cy_s * ((y1 - y0) / sh)
    dist = np.sqrt((fx - cx * w) ** 2 + (fy - cy * h) ** 2) / np.float32(r_px)
    cost += np.float32(_DISTANCE_WEIGHT) * dist * dist

    # Never over the spot or its ring, never out of the frame.
    edge = r_px * (1.0 + RING)
    banned = dist < 2.0 * (1.0 + RING)
    banned |= (fx < edge) | (fx > w - edge) | (fy < edge) | (fy > h - edge)
    for ax, ay in avoid:
        banned |= np.sqrt((fx - ax * w) ** 2 + (fy - ay * h) ** 2) < r_px
    cost = np.where(banned, np.float32(np.inf), cost)
    if not np.isfinite(cost).any():
        return _fallback(cx, cy, radius, h, w)
    best = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return float(fx[0, best[1]] / w), float(fy[best[0], 0] / h)


def _fallback(cx: float, cy: float, radius: float, h: int, w: int) -> tuple[float, float]:
    """Two and a half radii towards the middle of the frame: nothing to compare with."""
    long = float(max(h, w))
    step = 2.5 * (1.0 + RING) * radius * long
    direction = -1.0 if cx > 0.5 else 1.0
    return float(np.clip(cx + direction * step / w, 0.0, 1.0)), cy

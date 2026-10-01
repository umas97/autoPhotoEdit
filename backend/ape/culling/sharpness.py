# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focus and motion, measured on the camera's embedded preview (section 7.3).

**Focus is measured where the photograph is sharpest.** A portrait at f/1.8 is
mostly blur, and it is a good photograph. So the frame is cut into tiles and
only the best few count: a photo is sharp if *some* region of it is.

**The measure is a ratio, not an energy.** The variance of a Laplacian says as
much about the scene's contrast as about its focus -- a sharp picture of fog
scores lower than a blurred picture of a brick wall. What is computed instead is
how much detail survives an extra blur (after Crété-Roffet et al., "The blur
effect", SPIE 2007): re-blurring a sharp edge destroys most of its local
variation, re-blurring a soft one changes almost nothing. The result is a number
in [0, 1] that does not care how contrasty the scene is, which is the
"normalised for content" of section 7.3. The raw Laplacian energy is kept too,
because *within* a burst the scene is the same and it is the finer instrument.

**Motion blur is anisotropic, defocus is not.** The same ratio is computed in
four directions. A frame that is soft in one direction and crisp across it was
moved during the exposure; a frame that is soft in every direction was not in
focus. The distinction matters to the user, who fixes the two differently.

**Noise is taken out, not merely thresholded.** Grain is "detail" to any
sharpness measure, so its expected share is subtracted from both sides of the
ratio, with the noise level estimated on the frame itself (section 7.3's
"normalised for ISO").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["ANALYSIS_EDGE", "SharpnessMetrics", "analysis_gray", "sharpness_metrics"]

#: Long edge the sharpness is measured at. The A7 III's embedded preview is
#: 1616 px; measuring at a fixed size makes a sidecar JPEG (6000 px) and an
#: embedded preview give the same number for the same photograph.
ANALYSIS_EDGE = 1600

#: Taps of the re-blur. Nine is the value of the original metric, and at this
#: resolution it spans about as far as a clearly missed focus spreads an edge.
_TAPS = 9

#: Tiles along the long and the short side of the frame. Forty-eight tiles make
#: a region about a sixth of the frame across: small enough for a face in a
#: half-length portrait to fill one, large enough to hold real structure.
_TILES_LONG, _TILES_SHORT = 8, 6

#: How many tiles make "the sharpest region". Three is about six per cent of the
#: frame: a subject, not a speck of texture that happens to be crisp.
_TOP_TILES = 3

#: Candidate tiles for the four-direction measure. The diagonal pass is done on
#: these crops only, which is what keeps the whole analysis inside the 80 ms of
#: section 12.
_CANDIDATES = 6

#: The diagonal measure samples at a pitch of sqrt(2) pixels, so a blur of a
#: given width looks narrower along a diagonal than along an axis and the raw
#: ratio comes out systematically higher. Fitted on the 24 A7 III previews of
#: ``tests/fixtures`` blurred by sigma 0, 1.5 and 3 px: raising the diagonal
#: ratio to this power brings it onto the axial one within 0.02 at all three.
_DIAGONAL_EXPONENT = 1.25

#: Where is there something to be sharp? A tile is judged only if it has
#: structure at a quarter of the analysis size -- edges, texture, objects --
#: because that is what a missed focus *keeps*: blur removes the fine detail
#: of an edge, not the edge. Asking for fine detail instead would make every
#: badly blurred frame "not assessable", and so kept.
#:
#: The floor is the mean absolute difference between neighbouring pixels at
#: that scale. Measured: tiles of the sharp A7 III previews of
#: ``tests/fixtures`` sit above 0.011 in three cases out of four, and the same
#: previews blurred by sigma 3.5 px keep a median of 0.015; a synthetic sky of
#: soft white clouds peaks at 0.0097 and a clear sky at 0.0015. Below the floor
#: a frame is not judged at all, which is the right answer for a picture of
#: clouds: nothing in it was ever meant to be crisp.
_COARSE_SCALE = 4
_STRUCTURE_FLOOR = 0.010

#: A frame is judged only when at least this many tiles have structure. A
#: single contrasty cloud in an empty sky is not enough to call the frame
#: out of focus; a defocused scene has structure all over.
_MIN_STRUCTURED = 3

#: Patches of the noise estimate, rows by columns: about fifty pixels square
#: at the analysis size.
_NOISE_PATCHES = (24, 32)

#: Mean absolute difference of two neighbouring pixels of pure noise, per unit
#: of standard deviation: ``E|x - y| = 2 sigma / sqrt(pi)``. Both the variation
#: and what the re-blur removes contain this much noise; taking it out of both
#: is what keeps grain from passing for detail.
_NOISE_DIFFERENCE = 2.0 / math.sqrt(math.pi)


def analysis_gray(bgr: np.ndarray) -> np.ndarray:
    """8-bit BGR preview to float32 luma in [0, 1] at :data:`ANALYSIS_EDGE`.

    Luma in the preview's own gamma, not linear light: sharpness is about the
    edges a person sees, and the camera's encoding is close to perceptual.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray *= np.float32(1.0 / 255.0)
    height, width = gray.shape
    scale = ANALYSIS_EDGE / max(height, width)
    # Within three per cent of the target the resampling would blur more than
    # the size difference matters; the embedded 1616 px preview stays as is.
    if scale < 0.97:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


@dataclass(frozen=True)
class SharpnessMetrics:
    """Measured on the sharpest region. Ratios in [0, 1], higher is sharper."""

    #: Mean of the four directional ratios. ``None`` when nothing in the frame
    #: has enough structure to judge (a clear sky, a black frame).
    acuity: float | None
    acuity_min: float | None
    acuity_max: float | None
    #: ``(max - min) / max`` of the directional ratios: 0 is isotropic.
    anisotropy: float | None
    #: Mean squared Laplacian on the region, x 1e4. Comparable only between
    #: frames of the same scene, which is what bursts are.
    laplacian: float
    #: Estimated noise standard deviation, in 8-bit grey levels.
    noise: float
    structured_tiles: int
    #: The best tile, as ``[x, y, w, h]`` fractions of the frame: where the
    #: comparison view zooms to when the camera did not record a focus point.
    region: tuple[float, float, float, float] | None

    @property
    def assessable(self) -> bool:
        return self.acuity is not None


def _tile_means(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of each tile. An area resize to one pixel per tile is exactly that,
    and several times faster than the equivalent reshape in NumPy."""
    height, width = values.shape
    th, tw = height // rows, width // cols
    cropped = values[: th * rows, : tw * cols]
    return cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float64)


def _noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's estimator ("Fast noise variance estimation", CVIU 1996).

    The operator cancels any locally planar signal, so what remains in a flat
    patch is noise. It is averaged over small patches -- about fifty pixels --
    and the fifth percentile taken, which finds the flat ones without having to
    look for them. Large tiles do not work: a frame filled with a stack of logs
    has no flat tile a sixth of the frame wide, and its texture then passes for
    noise (measured: 1.8 grey levels on such a frame at ISO 100, against 1.1
    with small patches and 0.2 on an ordinary scene).
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
    np.abs(response, out=response)
    patches = _tile_means(response, _NOISE_PATCHES[0], _NOISE_PATCHES[1])
    return float(np.percentile(patches, 5)) * math.sqrt(math.pi / 2.0) / 6.0


def _axial(gray: np.ndarray, axis: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-tile variation and variation lost to a re-blur, along one axis."""
    size = (_TAPS, 1) if axis == 1 else (1, _TAPS)
    blurred = cv2.blur(gray, size, borderType=cv2.BORDER_REFLECT)
    if axis == 1:
        original = cv2.absdiff(gray[:, 1:], gray[:, :-1])
        reblurred = cv2.absdiff(blurred[:, 1:], blurred[:, :-1])
    else:
        original = cv2.absdiff(gray[1:], gray[:-1])
        reblurred = cv2.absdiff(blurred[1:], blurred[:-1])
    # subtract saturates at zero for unsigned types only; for float it is the
    # max(0, .) of the metric that has to be spelled out.
    lost = cv2.max(cv2.subtract(original, reblurred), 0.0)
    return _tile_means(original, rows, cols), _tile_means(lost, rows, cols)


def _diagonal(crop: np.ndarray, anti: bool) -> tuple[float, float]:
    """The same measure along a diagonal, on one small crop."""
    height, width = crop.shape
    r = _TAPS // 2
    total = np.zeros((height - 2 * r, width - 2 * r), dtype=np.float32)
    for k in range(-r, r + 1):
        dx = -k if anti else k
        total += crop[r + k : height - r + k, r + dx : width - r + dx]
    blurred = total / np.float32(_TAPS)
    centre = crop[r : height - r, r : width - r]
    if anti:
        original = np.abs(centre[1:, :-1] - centre[:-1, 1:])
        reblurred = np.abs(blurred[1:, :-1] - blurred[:-1, 1:])
    else:
        original = np.abs(centre[1:, 1:] - centre[:-1, :-1])
        reblurred = np.abs(blurred[1:, 1:] - blurred[:-1, :-1])
    return float(original.mean()), float(np.maximum(original - reblurred, 0.0).mean())


def _coarse_diagonal(coarse: np.ndarray, rows: int, cols: int, anti: bool) -> np.ndarray:
    if anti:
        diff = cv2.absdiff(coarse[1:, :-1], coarse[:-1, 1:])
    else:
        diff = cv2.absdiff(coarse[1:, 1:], coarse[:-1, :-1])
    return _tile_means(diff, rows, cols)


def sharpness_metrics(gray: np.ndarray) -> SharpnessMetrics:
    """Focus and motion measures of one frame.

    Two regions are measured, because the two questions need different ones.
    **Focus** is read on the sharpest tiles: a photo is in focus if something
    in it is. **Motion** is read on the tiles with structure in every
    direction: camera shake smears the whole frame along one direction, and
    the tiles that look sharpest in a shaken frame are precisely the ones whose
    edges run along the shake -- which is why they look sharp, and why they
    cannot show it.

    Args:
        gray: float32 luma in [0, 1], as returned by :func:`analysis_gray`.

    Returns:
        A :class:`SharpnessMetrics`. Never raises on a flat or black frame:
        it reports it as not assessable instead.
    """
    height, width = gray.shape
    rows, cols = (_TILES_SHORT, _TILES_LONG) if width >= height else (_TILES_LONG, _TILES_SHORT)
    th, tw = height // rows, width // cols

    noise = _noise_sigma(gray)
    horizontal = _axial(gray, 1, rows, cols)
    vertical = _axial(gray, 0, rows, cols)
    energy = (horizontal[0] + vertical[0]) / 2.0
    grain = _NOISE_DIFFERENCE * noise

    coarse = cv2.resize(
        gray,
        (max(cols, width // _COARSE_SCALE), max(rows, height // _COARSE_SCALE)),
        interpolation=cv2.INTER_AREA,
    )
    # Structure along each of the four directions, at the scale where blur has
    # not erased it: 0, 45, 90 and 135 degrees.
    presence = np.stack(
        [
            _tile_means(cv2.absdiff(coarse[:, 1:], coarse[:, :-1]), rows, cols),
            _coarse_diagonal(coarse, rows, cols, anti=False),
            _tile_means(cv2.absdiff(coarse[1:], coarse[:-1]), rows, cols),
            _coarse_diagonal(coarse, rows, cols, anti=True),
        ]
    ).reshape(4, -1)
    floor = _STRUCTURE_FLOOR
    structured = (presence[0] + presence[2]) / 2.0 > floor

    def laplacian_energy(index: int) -> float:
        r, c = divmod(index, cols)
        tile = gray[r * th : (r + 1) * th, c * tw : (c + 1) * tw]
        response = cv2.Laplacian(tile, cv2.CV_32F, ksize=3)
        return float(cv2.mean(cv2.multiply(response, response))[0])

    count = int(structured.sum())
    if count < _MIN_STRUCTURED:
        return SharpnessMetrics(
            None, None, None, None, laplacian_energy(int(np.argmax(energy))) * 1e4,
            noise * 255.0, count, None,
        )

    def ratio(variation, lost):
        # No fine variation at all, net of grain, is the limit of blur: the
        # ratio is 0 there, not undefined.
        return np.maximum(lost - grain, 0.0) / np.maximum(variation - grain, 1e-6)

    margin = _TAPS // 2 + 1
    cache: dict[int, list[float]] = {}

    def directional(index: int) -> list[float]:
        """The ratio along each direction that has structure, diagonals corrected."""
        if index in cache:
            return cache[index]
        r, c = divmod(index, cols)
        crop = gray[
            max(0, r * th - margin) : min(height, (r + 1) * th + margin),
            max(0, c * tw - margin) : min(width, (c + 1) * tw + margin),
        ]
        measured = [
            (horizontal[0].flat[index], horizontal[1].flat[index]),
            _diagonal(crop, anti=False),
            (vertical[0].flat[index], vertical[1].flat[index]),
            _diagonal(crop, anti=True),
        ]
        values = []
        for axis, (variation, lost) in enumerate(measured):
            # A direction along which the tile has no structure says nothing
            # about focus: a picket fence has no vertical detail to lose.
            # Leaving it out is what stops such a tile reading as "moved".
            if presence[axis, index] < floor * 0.5:
                continue
            value = float(ratio(variation, lost))
            if axis % 2 == 1:
                value = value**_DIAGONAL_EXPONENT
            values.append(value)
        cache[index] = values
        return values

    axial = np.where(
        structured, (ratio(*horizontal).ravel() + ratio(*vertical).ravel()) / 2.0, -1.0
    )
    focus_tiles = [
        (i, directional(i)) for i in np.argsort(axial)[-_CANDIDATES:] if axial[i] >= 0.0
    ]
    focus_tiles = [(i, v) for i, v in focus_tiles if v]
    if not focus_tiles:
        return SharpnessMetrics(
            None, None, None, None, laplacian_energy(int(np.argmax(energy))) * 1e4,
            noise * 255.0, count, None,
        )
    focus_tiles.sort(key=lambda item: float(np.mean(item[1])))
    best = focus_tiles[-_TOP_TILES:]
    acuity = float(np.mean([np.mean(v) for _, v in best]))
    high = float(np.mean([max(v) for _, v in best]))

    # Motion: the tiles whose weakest direction still has the most structure.
    isotropic = np.where(structured, presence.min(axis=0), -1.0)
    motion_tiles = [
        directional(i) for i in np.argsort(isotropic)[-_CANDIDATES:] if isotropic[i] > 0.0
    ]
    motion_tiles = [v for v in motion_tiles if len(v) == 4]
    if motion_tiles:
        low = float(np.median([min(v) for v in motion_tiles]))
        anisotropy = float(np.median([(max(v) - min(v)) / max(max(v), 1e-6) for v in motion_tiles]))
    else:
        low = float(np.mean([min(v) for _, v in best]))
        anisotropy = (high - low) / max(high, 1e-6)
    laplacian = float(np.mean([laplacian_energy(int(i)) for i, _ in best]) * 1e4)

    r, c = divmod(int(best[-1][0]), cols)
    region = (c * tw / width, r * th / height, tw / width, th / height)
    return SharpnessMetrics(
        acuity=acuity,
        acuity_min=low,
        acuity_max=high,
        anisotropy=anisotropy,
        laplacian=laplacian,
        noise=noise * 255.0,
        structured_tiles=count,
        region=tuple(round(v, 4) for v in region),  # type: ignore[arg-type]
    )

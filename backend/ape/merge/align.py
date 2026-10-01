# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registration of the frames of a merge onto the reference (sections 25.3, 25.4).

Everything is estimated on reduced frames and applied at full resolution: a
transform found at a quarter of the size is the same transform, rescaled, and
finding it there costs a sixteenth.

Two stages, as sections 25.3 and 25.4 ask:

1. **ORB + RANSAC** gives a robust first guess that survives large shifts,
   which ECC alone would not (its basin of convergence is a few pixels);
2. **ECC** refines it on the pixels themselves, to a fraction of a pixel, which
   feature points alone rarely reach on soft or low-contrast frames.

The model depends on the merge: a homography for HDR frames (a hand-held
bracketing tilts as well as shifts), a similarity for a focus stack (focus
breathing is a change of scale about the centre, and the extra freedom of a
homography only lets a blurred frame drift).

The images compared are brightness-matched first -- an exposure bracketing is
the same scene at different levels -- and gamma-encoded, because feature
detectors and ECC both work on perceptual contrast, which linear light hides
in the shadows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "ALIGN_EDGE",
    "Alignment",
    "align",
    "for_matching",
    "residual_shift",
    "scale_transform",
]

#: Long edge at which transforms are estimated. A 6000 px frame aligned at 1500
#: px and rescaled lands within a quarter of a full-resolution pixel of where
#: the full-size estimate does, measured on the fixtures shifted by known
#: amounts (``tests/test_merge_hdr.py``), for a sixteenth of the work.
ALIGN_EDGE = 1500

#: ORB features asked for. Two thousand is where the RANSAC inlier count on the
#: fixtures stops growing; more only costs matching time.
_ORB_FEATURES = 2000

#: Lowe's ratio test on the two nearest descriptors.
_RATIO = 0.75

#: Fewest RANSAC inliers that make an estimate worth refining. Below this the
#: frames do not share enough structure, and the merge says so.
MIN_INLIERS = 24

#: ECC stops after this many iterations or this change in the correlation.
_ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)


@dataclass(frozen=True)
class Alignment:
    #: 3x3 transform taking reference coordinates to frame coordinates, in the
    #: pixels of the images it was estimated on.
    matrix: np.ndarray
    #: RMS distance, in those pixels, between matched feature points after the
    #: transform: the scatter of ORB's localisation, not the misalignment
    #: (:func:`residual_shift` measures that).
    point_rms_px: float
    inliers: int
    #: Whether ECC converged; when it did not, ``matrix`` is the RANSAC guess.
    refined: bool

    def to_full(self, scale: float) -> np.ndarray:
        """The transform for frames ``scale`` times larger."""
        return scale_transform(self.matrix, scale)


def scale_transform(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Conjugate a transform by a uniform scaling of both images."""
    s = np.diag([scale, scale, 1.0])
    return s @ matrix @ np.linalg.inv(s)


def for_matching(linear: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """An 8-bit gamma-encoded grey image of a linear frame, brightness-matched.

    ``gain`` divides the frame first, so the frames of a bracketing compare at
    the reference's exposure. Values past white clip, which is exactly what the
    frame recorded there.
    """
    grey = linear[..., 0] * 0.2627 + linear[..., 1] * 0.678 + linear[..., 2] * 0.0593
    grey = np.clip(grey / max(gain, 1e-6), 0.0, 1.0) ** (1.0 / 2.2)
    return (grey * 255.0 + 0.5).astype(np.uint8)


def _features(image: np.ndarray):
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    # CLAHE evens out the contrast so the features are spread over the frame
    # instead of piling up on its brightest edge.
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return orb.detectAndCompute(equalised, None)


def _ransac(reference: np.ndarray, frame: np.ndarray, model: str):
    keys_ref, desc_ref = _features(reference)
    keys_frm, desc_frm = _features(frame)
    if desc_ref is None or desc_frm is None or len(keys_ref) < 8 or len(keys_frm) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_ref, desc_frm, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < _RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([keys_ref[m.queryIdx].pt for m in good])
    dst = np.float32([keys_frm[m.trainIdx].pt for m in good])
    if model == "homography":
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=4000)
    else:
        affine, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000
        )
        matrix = None if affine is None else np.vstack([affine, [0.0, 0.0, 1.0]])
    if matrix is None or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < MIN_INLIERS:
        return None
    return matrix, src[inliers], dst[inliers]


def _residual(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))


#: Tile size, at the alignment scale, of the residual measurement. Large enough
#: for phase correlation to lock on to texture, small enough to see a corner
#: that a homography could not follow.
_RESIDUAL_TILE = 128

#: Least phase-correlation response for a tile's shift to count. Below it the
#: peak is noise: on the user's hand-held brackets at ISO 2500-6400
#: (``tests/fixtures/fase11``) the tiles under 0.3 reported shifts of 3 to 15
#: pixels where the 1:1 crops show none.
_RESIDUAL_RESPONSE = 0.3

#: Which share of the tiles the answer speaks for. A misalignment moves most of
#: the frame; a subject that moved -- a curtain, a hanging basket -- moves a few
#: tiles, and that is the deghosting's business (section 25.3.5), not the
#: alignment's. With the 90th percentile five tiles out of fifty decided the
#: answer, and the user's brackets, sharp at 1:1, read as 4-5 px misaligned.
_RESIDUAL_PERCENTILE = 75


def residual_shift(reference: np.ndarray, warped: np.ndarray) -> float:
    """How far the aligned frame still is from the reference, in pixels.

    Phase correlation on a grid of tiles, after the transform: each textured
    tile reports the shift left between the two, and the answer is the 75th
    percentile of their lengths. This is the "residual displacement" of
    section 25.3.2 -- what a viewer would see as a double edge -- rather than
    the scatter of the feature points, which says how well ORB located them.
    """
    height, width = reference.shape[:2]
    window = cv2.createHanningWindow((_RESIDUAL_TILE, _RESIDUAL_TILE), cv2.CV_32F)
    shifts = []
    for top in range(0, height - _RESIDUAL_TILE + 1, _RESIDUAL_TILE):
        for left in range(0, width - _RESIDUAL_TILE + 1, _RESIDUAL_TILE):
            a = reference[top : top + _RESIDUAL_TILE, left : left + _RESIDUAL_TILE]
            b = warped[top : top + _RESIDUAL_TILE, left : left + _RESIDUAL_TILE]
            if float(a.std()) < 6.0 or float(b.std()) < 6.0:
                continue  # sky, a wall: nothing to measure a shift on
            (dx, dy), response = cv2.phaseCorrelate(
                a.astype(np.float32), b.astype(np.float32), window
            )
            if response >= _RESIDUAL_RESPONSE:
                shifts.append(float(np.hypot(dx, dy)))
    if len(shifts) < 4:
        return 0.0
    return float(np.percentile(shifts, _RESIDUAL_PERCENTILE))


def align(
    reference: np.ndarray,
    frame: np.ndarray,
    *,
    model: str = "homography",
    mask: np.ndarray | None = None,
) -> Alignment | None:
    """Find the transform that maps the reference onto ``frame``.

    Args:
        reference, frame: 8-bit grey images of the same size, from
            :func:`for_matching`.
        model: ``"homography"`` or ``"similarity"``.
        mask: optional 8-bit mask of the reference pixels ECC may use -- for a
            bracketing, the ones neither frame clipped.

    Returns:
        An :class:`Alignment`, or ``None`` when the frames do not share enough
        structure to be registered at all.
    """
    found = _ransac(reference, frame, model)
    if found is None:
        return None
    guess, src, dst = found

    motion = cv2.MOTION_HOMOGRAPHY if model == "homography" else cv2.MOTION_AFFINE
    warp = guess.astype(np.float32) if model == "homography" else guess[:2].astype(np.float32)
    refined = False
    try:
        # ECC's warp maps template (reference) coordinates to input (frame)
        # coordinates, the same direction as ``guess``.
        _, warp = cv2.findTransformECC(
            reference.astype(np.float32), frame.astype(np.float32), warp, motion,
            _ECC_CRITERIA, mask, 5,
        )
        refined = True
    except cv2.error:
        pass
    matrix = warp.astype(np.float64)
    if matrix.shape[0] == 2:
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if refined and _residual(matrix, src, dst) > 2.0 * _residual(guess, src, dst) + 1.0:
        # ECC wandered off -- a repeated texture, a moving subject -- and the
        # feature points disagree with it. Trust the points.
        matrix, refined = guess, False
    return Alignment(
        matrix=matrix,
        point_rms_px=round(_residual(matrix, src, dst), 3),
        inliers=int(len(src)),
        refined=refined,
    )

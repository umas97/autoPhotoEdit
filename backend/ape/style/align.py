# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the edited reference sits inside the neutral rendering (section 8.2.2).

A reference is rarely the whole frame: the user straightened it, cropped it,
often turned a landscape frame into a portrait one. Comparing colours pixel by
pixel needs the two images registered first, and the registration has to
survive the very thing the comparison is about -- the reference looks
*different*, darker or brighter, warmer, more saturated.

**Method.** SIFT keypoints on both images after a local histogram equalisation
(CLAHE), which takes most of the tonal difference out of the descriptors;
Lowe's ratio test; a similarity transform (rotation, uniform scale,
translation) fitted with RANSAC. A similarity rather than a homography because
that is what an editor's crop and straighten are, and four fewer degrees of
freedom make the fit robust on the frames with few keypoints (a dark stage, a
cloudy sky).

Measured on the user's 65 Lightroom edits: every one registered, the rotation
recovered agrees with ``crs:CropAngle`` within 0.02 degrees, and the median
residual of the inliers is 0.2-0.6 px at 1024 px. The lens profiles differ
(Adobe's against lensfun), which is what the residual mostly is.

Output: the reference resampled into the *neutral* frame, with a mask of where
it is defined. The comparison then happens at the parameters' own geometry, so
a radius learned here means the same thing when it is applied to a new photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Alignment", "AlignmentError", "align", "warp_reference"]

#: Long edge at which keypoints are found. SIFT is scale-invariant, so this is
#: about speed and about having enough detail on a heavily cropped reference:
#: at 1024, 0.1 s per pair and never fewer than 77 inliers on the user's set.
ALIGN_EDGE = 1024

#: Lowe's ratio. 0.75 is the value of the original paper and it holds here.
_RATIO = 0.75

#: RANSAC inlier distance, in pixels at ALIGN_EDGE. Twice the worst median
#: residual seen, so lens-profile differences in the corners do not exclude
#: the corners.
_RANSAC_PX = 2.0

#: Below this many inliers the transform is a guess, not a measurement.
MIN_INLIERS = 25


class AlignmentError(ValueError):
    """The reference could not be registered on the RAW: not the same shot,
    or too different to tell."""


@dataclass(frozen=True, slots=True)
class Alignment:
    #: 2x3 similarity mapping reference pixels to neutral pixels, both at the
    #: sizes given below.
    matrix: np.ndarray
    reference_size: tuple[int, int]  # (width, height)
    neutral_size: tuple[int, int]
    inliers: int
    residual_px: float

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def scale(self) -> float:
        return float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def coverage(self) -> float:
        """Fraction of the neutral frame the reference covers."""
        rw, rh = self.reference_size
        nw, nh = self.neutral_size
        return float(min(1.0, (self.scale**2) * rw * rh / (nw * nh)))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _resize(image: np.ndarray, edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = edge / max(height, width)
    if abs(scale - 1.0) < 1e-3:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def _equalised(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)


def align(neutral: np.ndarray, reference: np.ndarray) -> Alignment:
    """Register ``reference`` on ``neutral``.

    Args:
        neutral: the neutral rendering of the RAW, upright, sRGB.
        reference: the user's edited version, upright, sRGB. Any size.

    Returns:
        The transform from reference to neutral pixels at ``ALIGN_EDGE``.

    Raises:
        AlignmentError: too few consistent matches.
    """
    a = _resize(_to_uint8(reference), ALIGN_EDGE)
    b = _resize(_to_uint8(neutral), ALIGN_EDGE)
    sift = cv2.SIFT_create(nfeatures=4000)
    keys_a, desc_a = sift.detectAndCompute(_equalised(a), None)
    keys_b, desc_b = sift.detectAndCompute(_equalised(b), None)
    if desc_a is None or desc_b is None or len(keys_a) < MIN_INLIERS or len(keys_b) < MIN_INLIERS:
        raise AlignmentError("troppo pochi dettagli per allineare il riferimento al RAW")

    matches = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < _RATIO * m[1].distance]
    if len(good) < MIN_INLIERS:
        raise AlignmentError("il riferimento non sembra la stessa foto del RAW")
    src = np.float32([keys_a[m.queryIdx].pt for m in good])
    dst = np.float32([keys_b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_PX,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if matrix is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        raise AlignmentError("il riferimento non sembra la stessa foto del RAW")
    inlier = mask.ravel().astype(bool)
    projected = cv2.transform(src[inlier][None], matrix)[0]
    residual = float(np.median(np.linalg.norm(projected - dst[inlier], axis=1)))
    return Alignment(
        matrix=matrix.astype(np.float64),
        reference_size=(a.shape[1], a.shape[0]),
        neutral_size=(b.shape[1], b.shape[0]),
        inliers=int(inlier.sum()),
        residual_px=residual,
    )


def warp_reference(
    alignment: Alignment,
    reference: np.ndarray,
    neutral_shape: tuple[int, int],
    *,
    erode_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """The reference resampled into the neutral frame, and where it is defined.

    Args:
        alignment: from :func:`align`.
        reference: the reference at any size (it is rescaled to match).
        neutral_shape: ``(height, width)`` of the neutral image to compare with.
        erode_px: margin removed from the mask, so that resampling at the
            border -- half reference, half nothing -- never enters a loss.

    Returns:
        ``(image, mask)``: float32 sRGB in [0, 1] and a boolean mask, both of
        ``neutral_shape``.
    """
    height, width = neutral_shape
    rw, rh = alignment.reference_size
    nw, nh = alignment.neutral_size
    # Compose: reference(actual) -> reference(ALIGN_EDGE) -> neutral(ALIGN_EDGE)
    # -> neutral(target size).
    to_ref = np.diag([rw / reference.shape[1], rh / reference.shape[0], 1.0])
    to_target = np.diag([width / nw, height / nh, 1.0])
    full = to_target @ np.vstack([alignment.matrix, [0, 0, 1]]) @ to_ref
    source = reference.astype(np.float32) / (255.0 if reference.dtype == np.uint8 else 1.0)
    # Pre-filter a reference much larger than the target: warpAffine samples,
    # it does not average, and aliasing would add a noise floor to the loss.
    factor = float(np.hypot(full[0, 0], full[1, 0]))
    if factor < 0.5:
        blur = 0.5 / factor
        source = cv2.GaussianBlur(source, (0, 0), sigmaX=blur * 0.6)
    image = cv2.warpAffine(
        source, full[:2], (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    coverage = cv2.warpAffine(
        np.ones(reference.shape[:2], np.float32),
        full[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    mask = coverage > 0.999
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return np.clip(image, 0.0, 1.0), mask

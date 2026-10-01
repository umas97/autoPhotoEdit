# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Everything a panorama decides, decided on reduced frames (section 25.5.1).

Homographies, the order and rotation of the cameras, the projection, the
exposure compensation, the seams and the crop are all found here, on frames at
a quarter of their side: a sixteenth of the pixels, where a mistake costs
seconds instead of minutes. ``pano_compose.py`` then only executes the plan at
full resolution.

OpenCV's stitching classes do the estimation, called one by one rather than
through ``cv2.Stitcher``, so that every step can fail with its own reason and
none decides anything this module does not see. Two of their defaults are
overridden:

* **Match confidence.** ``BestOf2NearestMatcher`` sets to zero the confidence
  of a pair above 3, meant to drop duplicate images -- which also drops the
  pairs of a panorama shot with generous overlap, the best ones it has. The
  confidence is recomputed here without that cap.
* **Exposure.** The gains are solved in linear light, together with the lens's
  vignetting, on the overlaps, with the reference frame fixed: in a RAW's
  linear space an exposure difference is exactly a gain, and the reference's
  exposure is the one the user metered (``pano_photometry.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..analysis.borders import inner_rectangle
from .align import for_matching
from .errors import MergeFailure
from .pano_geometry import Camera, warp_roi, warper_name
from .pano_photometry import NO_VIGNETTING, falloff, radius2, solve

__all__ = ["Plan", "estimate"]

#: ORB features per frame at the estimation size. Panoramas overlap by 15-60%,
#: so only part of the features can match; three thousand leave a few hundred
#: in a 20% overlap on the fixtures.
_FEATURES = 3000

#: Least confidence (inliers over matches, OpenCV's measure) for two frames to
#: be linked. One is OpenCV's own threshold for "these two overlap".
_LINK_CONFIDENCE = 1.0

#: Long edge of the frames the seams are cut on. Graph cuts grow faster than
#: the pixel count; OpenCV's default of 0.1 MP per frame is about this.
_SEAM_EDGE = 400

#: Proposal of section 25.5.3. A horizontal sweep narrower than this (degrees)
#: is "pochi scatti a focale lunga", for the plane; a vertical extent wider
#: than that needs the sphere.
_PLANE_BELOW_DEG, _SPHERE_ABOVE_DEG = 40.0, 70.0


@dataclass
class Plan:
    projection: str
    #: Warper scale at the estimation size.
    scale: float
    #: Per frame, at the estimation size ``size``.
    cameras: list[Camera]
    size: tuple[int, int]
    corners: list[tuple[int, int]]
    sizes: list[tuple[int, int]]
    gains: list[float]
    #: Seam masks, uint8, at ``seam_factor`` times the estimation scale, over
    #: each frame's warped rectangle.
    seams: list[np.ndarray]
    seam_factor: float
    #: The panorama's rectangle in warper coordinates at the estimation size.
    roi: tuple[int, int, int, int]
    #: Largest rectangle with no empty pixel, relative to ``roi``.
    crop: tuple[int, int, int, int] | None
    span_deg: tuple[float, float]
    pairs: list[dict] = field(default_factory=list)
    #: The sweep tilts rather than turns (:func:`_vertical`).
    vertical: bool = False
    #: The lens's falloff left in the frames (``pano_photometry``), undone as
    #: each frame is read.
    vignetting: tuple[float, float, float, float] = NO_VIGNETTING

    @property
    def warper(self) -> str:
        """The name of the OpenCV warper, and of ``backward_map``'s formula."""
        return warper_name(self.projection, self.vertical)


def _features(frames: Sequence[np.ndarray]):
    orb = cv2.ORB_create(nfeatures=_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    found = []
    for index, frame in enumerate(frames):
        features = cv2.detail.computeImageFeatures2(orb, clahe.apply(for_matching(frame)))
        features.img_idx = index
        found.append(features)
    return found


#: Seed of OpenCV's random generator for the matching's RANSAC.
_SEED = 25_05


def _matches(features) -> list:
    # The matcher runs RANSAC per pair on worker threads, each with its own
    # random state: the inliers -- and the whole panorama -- changed between
    # two runs of the same recipe. One thread and a fixed seed make a lost
    # intermediate come back identical (section 25.1); matching takes 0.1 s.
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        cv2.setRNGSeed(_SEED)
        matcher = cv2.detail_BestOf2NearestMatcher(False, 0.3)
        pairs = matcher.apply2(features)
        matcher.collectGarbage()
    finally:
        cv2.setNumThreads(threads)
    for pair in pairs:
        if pair.src_img_idx != pair.dst_img_idx and pair.num_inliers > 0:
            pair.confidence = pair.num_inliers / (8.0 + 0.3 * len(pair.matches))
    return pairs


def _cameras(features, pairs, names: Sequence[str]) -> list:
    component = cv2.detail.leaveBiggestComponent(features, pairs, _LINK_CONFIDENCE)
    keep = [int(i) for i in np.ravel(component)]
    if len(keep) < len(names):
        lost = ", ".join(names[i] for i in range(len(names)) if i not in keep)
        raise MergeFailure(
            f"{lost} non si sovrappone abbastanza agli altri scatti: "
            "servono almeno il 15% di area in comune e dettagli riconoscibili"
        )
    ok, cameras = cv2.detail_HomographyBasedEstimator().apply(features, pairs, None)
    if not ok:
        raise MergeFailure("la geometria della panoramica non si ricostruisce dagli scatti")
    for camera in cameras:
        camera.R = camera.R.astype(np.float32)
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(_LINK_CONFIDENCE)
    ok, cameras = adjuster.apply(features, pairs, cameras)
    if not ok:
        raise MergeFailure(
            "gli scatti non stanno insieme in una panoramica: parallasse o soggetti in "
            "movimento lungo le giunzioni"
        )
    return list(cameras)


def _vertical(cameras) -> bool:
    """Whether the sweep tilts the camera rather than turning it.

    A turn moves the cameras' X axes around the vertical and leaves their Y
    axes where they are; a tilt does the opposite. The spread of a set of unit
    axes is what the largest eigenvalue of their moment leaves out of their
    count: zero when they are all parallel.
    """
    def spread(column: int) -> float:
        axes = np.array([np.asarray(c.R, np.float64)[:, column] for c in cameras])
        return len(axes) - float(np.linalg.eigvalsh(axes.T @ axes)[-1])

    return spread(1) > spread(0)


def _straighten(cameras, vertical: bool) -> None:
    """Level the horizon of the sweep.

    Horizontal sweeps get OpenCV's horizontal correction. Not its vertical
    one: that turns every camera by 90 degrees, and on two real frames with
    more tilt than pan it laid the panorama on its side. A vertical sweep is
    levelled here instead: the world's X axis is the one the camera tilted
    around -- the mean of the cameras' X axes -- and Z the mean view, made
    square to it; the lying cylinder (``pano_geometry.warper_name``) then draws
    it upright. OpenCV's horizontal correction on such a sweep looks for the
    normal of a plane of X axes that are all the same, and picks a vertical at
    random: on the user's 07336-07354 the panorama came out spherical, 360 by
    180 degrees, mostly black.
    """
    if not vertical:
        levelled = cv2.detail.waveCorrect([c.R for c in cameras], cv2.detail.WAVE_CORRECT_HORIZ)
        for camera, rotation in zip(cameras, levelled, strict=True):
            camera.R = rotation
        return
    rotations = [np.asarray(c.R, np.float64) for c in cameras]
    x_axis = sum(r[:, 0] for r in rotations)
    x_axis /= np.linalg.norm(x_axis)
    view = sum(r[:, 2] for r in rotations)
    z_axis = view - (view @ x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    world = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    for camera, rotation in zip(cameras, rotations, strict=True):
        camera.R = (world @ rotation).astype(np.float32)


def _span(cameras: list[Camera], scale: float, size: tuple[int, int],
          vertical: bool) -> tuple[float, float]:
    """Horizontal and vertical extent of the panorama, in degrees."""
    warper = "transverseMercator" if vertical else "spherical"
    rois = [warp_roi(warper, scale, c, size) for c in cameras]
    left = min(r[0] for r in rois)
    right = max(r[0] + r[2] for r in rois)
    top = min(r[1] for r in rois)
    bottom = max(r[1] + r[3] for r in rois)
    return math.degrees((right - left) / scale), math.degrees((bottom - top) / scale)


def _propose(span: tuple[float, float], vertical_sweep: bool) -> str:
    horizontal, vertical = span
    if max(horizontal, vertical) < _PLANE_BELOW_DEG:
        return "plane"
    if vertical > _SPHERE_ABOVE_DEG and not vertical_sweep:
        return "spherical"
    return "cylindrical"


def _warp_all(frames, cameras, projection, scale):
    """The frames on the panorama's surface, their masks, corners, and the
    squared radius each warped pixel had in its own frame."""
    warper = cv2.PyRotationWarper(projection, float(scale))
    warped, masks, radii, corners = [], [], [], []
    for frame, camera in zip(frames, cameras, strict=True):
        k, r = camera.K.astype(np.float32), camera.R.astype(np.float32)
        corner, image = warper.warp(frame, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        ones = np.full(frame.shape[:2], 255, np.uint8)
        _, mask = warper.warp(ones, k, r, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        _, radius = warper.warp(radius2(frame.shape[1], frame.shape[0]), k, r,
                                cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
        warped.append(image)
        masks.append(mask)
        radii.append(radius)
        corners.append((int(corner[0]), int(corner[1])))
    return warped, masks, radii, corners


def _seams(warped, masks, corners) -> tuple[list[np.ndarray], float]:
    factor = min(1.0, _SEAM_EDGE / max(max(m.shape) for m in masks))
    small_images, small_masks, small_corners = [], [], []
    for image, mask, (x, y) in zip(warped, masks, corners, strict=True):
        size = (max(1, round(mask.shape[1] * factor)), max(1, round(mask.shape[0] * factor)))
        grey = np.clip(np.nan_to_num(image), 0.0, 1.0) ** (1.0 / 2.2)
        small_images.append(cv2.resize(grey.astype(np.float32), size, interpolation=cv2.INTER_AREA))
        small_masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
        small_corners.append((round(x * factor), round(y * factor)))
    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    found = finder.find(small_images, small_corners, small_masks)
    return [np.asarray(m.get() if hasattr(m, "get") else m, dtype=np.uint8) for m in found], factor


def _crop(masks, corners, roi) -> tuple[int, int, int, int] | None:
    """Largest rectangle inside the union of the warped frames, relative to ``roi``."""
    x0, y0, width, height = roi
    union = np.zeros((height, width), bool)
    for mask, (x, y) in zip(masks, corners, strict=True):
        region = union[y - y0 : y - y0 + mask.shape[0], x - x0 : x - x0 + mask.shape[1]]
        region |= mask[: region.shape[0], : region.shape[1]] > 0
    return inner_rectangle(union)


def estimate(
    frames: Sequence[np.ndarray], names: Sequence[str], reference: int,
    *, projection: str | None = None,
) -> Plan:
    """Plan a panorama from reduced, lens-corrected, linear frames.

    Raises:
        MergeFailure: the frames do not overlap enough, or do not fit together.
    """
    features = _features(frames)
    pairs = _matches(features)
    detail_cameras = _cameras(features, pairs, names)
    vertical = _vertical(detail_cameras)
    _straighten(detail_cameras, vertical)
    cameras = [Camera(K=c.K().astype(np.float64), R=c.R.astype(np.float64)) for c in detail_cameras]
    scale = float(np.median([c.focal for c in detail_cameras]))
    size = (frames[0].shape[1], frames[0].shape[0])
    span = _span(cameras, scale, size, vertical)
    chosen = projection or _propose(span, vertical)
    warped, masks, radii, corners = _warp_all(frames, cameras, warper_name(chosen, vertical),
                                              scale)
    x0 = min(x for x, _ in corners)
    y0 = min(y for _, y in corners)
    x1 = max(x + m.shape[1] for (x, _), m in zip(corners, masks, strict=True))
    y1 = max(y + m.shape[0] for (_, y), m in zip(corners, masks, strict=True))
    roi = (x0, y0, x1 - x0, y1 - y0)
    photometry = solve(warped, masks, radii, corners, reference)
    # The seams are cut on the frames as they will be blended: a seam that
    # follows a vignetting step would be looking for the wrong edge.
    for index, (gain, radius) in enumerate(zip(photometry.gains, radii, strict=True)):
        correction = np.float32(gain) / falloff(radius, photometry.vignetting)
        warped[index] = warped[index] * correction[..., None]
    seams, seam_factor = _seams(warped, masks, corners)
    count = len(frames)
    linked = [
        {"a": names[p.src_img_idx], "b": names[p.dst_img_idx],
         "confidence": round(float(p.confidence), 2), "inliers": int(p.num_inliers)}
        for k, p in enumerate(pairs)
        if divmod(k, count)[0] < divmod(k, count)[1] and p.confidence >= _LINK_CONFIDENCE
    ]
    return Plan(
        projection=chosen, scale=scale, cameras=cameras, size=size, corners=corners,
        sizes=[(m.shape[1], m.shape[0]) for m in masks], gains=photometry.gains, seams=seams,
        seam_factor=seam_factor, roi=roi, crop=_crop(masks, corners, roi),
        span_deg=(round(span[0], 1), round(span[1], 1)), pairs=linked, vertical=vertical,
        vignetting=photometry.vignetting,
    )

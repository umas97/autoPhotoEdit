# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The projections of a panorama, as maps from the output back to a frame.

The estimation (``pano_estimate.py``) runs on OpenCV's stitching classes; the
composition (``pano_compose.py``) needs, for one band of the output at a time,
where each output pixel comes from in each frame. OpenCV's warpers build that
map for a whole frame at once -- 200 MB of coordinates for a 24 MP frame --
so the backward mapping is written out here, with OpenCV's own conventions
(``modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp``), and
evaluated only on the pixels of the band. ``tests/test_merge_panorama.py``
checks it against ``cv2.PyRotationWarper``.

Coordinates: the output is in the warper's coordinates -- ``u = scale * angle``
for the cylinder, ``scale * x / z`` for the plane -- and the frame in pixels.

A **vertical** panorama -- a sweep that tilts instead of turning, like the
user's 07336-07354 -- has its cylinder lying along the sweep, around the
world's X axis: OpenCV's ``transverseMercator``, whose rows are the angle
around X and whose columns are the conformal distance from the sweep's plane.
It is what "cilindrica" means for such a sweep (:func:`warper_name`); the
sphere would put its pole inside the panorama.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["PROJECTIONS", "Camera", "backward_map", "warp_roi", "warper_name"]

#: Section 25.5.3, in the names cv2.PyRotationWarper takes.
PROJECTIONS = ("cylindrical", "spherical", "plane")


def warper_name(projection: str, vertical: bool) -> str:
    """The OpenCV warper that draws ``projection`` for a sweep of this direction.

    For a vertical sweep both curved projections are the lying cylinder: the
    upright ones have their axis, or their pole, across the sweep.
    """
    if vertical and projection != "plane":
        return "transverseMercator"
    return projection


@dataclass(frozen=True)
class Camera:
    """A frame's intrinsics and rotation, in the pixels of some frame size."""

    K: np.ndarray
    R: np.ndarray

    def scaled(self, factor: float) -> Camera:
        """The same camera for a frame ``factor`` times larger."""
        k = self.K.astype(np.float64).copy()
        k[:2] *= factor
        return Camera(K=k, R=self.R)


def warp_roi(
    projection: str, scale: float, camera: Camera, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """``(x, y, width, height)`` of a frame of ``size`` once warped, as OpenCV finds it."""
    warper = cv2.PyRotationWarper(projection, float(scale))
    x, y, w, h = warper.warpRoi(size, camera.K.astype(np.float32), camera.R.astype(np.float32))
    return int(x), int(y), int(w), int(h)


def backward_map(
    projection: str, scale: float, camera: Camera, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frame pixel ``(x, y)`` that output point ``(u, v)`` samples.

    ``u``, ``v`` are float arrays of the same shape in warper coordinates.
    Points behind the camera map to ``-1`` (outside every frame).
    """
    k_rinv = camera.K.astype(np.float64) @ camera.R.astype(np.float64).T
    u = u.astype(np.float64) / scale
    v = v.astype(np.float64) / scale
    if projection == "cylindrical":
        x_, y_, z_ = np.sin(u), v, np.cos(u)
    elif projection == "spherical":
        sinv = np.sin(np.pi - v)
        x_, y_, z_ = sinv * np.sin(u), np.cos(np.pi - v), sinv * np.cos(u)
    elif projection == "plane":
        x_, y_, z_ = u, v, np.ones_like(u)
    elif projection == "transverseMercator":
        lat = np.arcsin(np.sin(v) / np.cosh(u))
        lon = np.arctan2(np.sinh(u), np.cos(v))
        x_, y_, z_ = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    else:
        raise ValueError(f"proiezione sconosciuta: {projection}")
    x = k_rinv[0, 0] * x_ + k_rinv[0, 1] * y_ + k_rinv[0, 2] * z_
    y = k_rinv[1, 0] * x_ + k_rinv[1, 1] * y_ + k_rinv[1, 2] * z_
    z = k_rinv[2, 0] * x_ + k_rinv[2, 1] * y_ + k_rinv[2, 2] * z_
    front = z > 0
    safe = np.where(front, z, 1.0)
    map_x = np.where(front, x / safe, -1.0).astype(np.float32)
    map_y = np.where(front, y / safe, -1.0).astype(np.float32)
    return map_x, map_y

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Our straighten-and-crop, expressed as points of the uncropped frame.

Both sidecar formats need the crop as positions in the picture *before* the
rotation, which is not how ``GeometryParams`` stores it: ours is normalised to
the straightened frame (``pipeline/geometry.py``), so that a crop stays put
while the angle is adjusted. This module walks the corners of the final frame
back through the same affine map the renderer uses, and hands them out in the
upright frame or in the sensor's own orientation.

Lightroom's convention was checked against the user's 65 edits rather than
assumed (phase 8): registering each delivered JPEG on its RAW gives
the true crop, and ``crs:CropLeft/Top/Right/Bottom`` are the corners of the
rotated crop *in sensor orientation* -- the top-left corner of the rectangle
as the sensor sees it, and the opposite one -- normalised to the sensor frame;
``crs:CropAngle`` has the sign of our ``rotation_deg``. Agreement: within
0.002 of the frame on every checked pair, which is the difference between
Adobe's and lensfun's distortion profiles.
"""

from __future__ import annotations

import math

from ..pipeline.geometry import straightened_size
from ..pipeline.params import CropRect, GeometryParams

__all__ = ["frame_corners", "sensor_corners", "to_sensor"]


def frame_corners(
    geometry: GeometryParams, width: float, height: float
) -> list[tuple[float, float]]:
    """Corners of the output frame, normalised to the upright uncropped picture.

    Order: top-left, top-right, bottom-right, bottom-left, *of the output*.
    With a rotation they are not axis-aligned in the source.
    """
    sw, sh = straightened_size(int(round(width)), int(round(height)), geometry.rotation_deg)
    crop = geometry.crop or CropRect()
    theta = math.radians(geometry.rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    corners = []
    for a, b in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        # Same map as ``geometry.apply``, on continuous coordinates centred on
        # the frame: output -> straightened frame -> inverse rotation -> source.
        qx = (crop.x + a * crop.width) * sw - sw / 2
        qy = (crop.y + b * crop.height) * sh - sh / 2
        x = c * qx - s * qy + width / 2
        y = s * qx + c * qy + height / 2
        corners.append((x / width, y / height))
    return corners


def to_sensor(u: float, v: float, orientation: int | None) -> tuple[float, float]:
    """An upright normalised point, in the sensor's orientation.

    EXIF orientation 6 is displayed after a quarter turn clockwise, 8 after a
    quarter turn anticlockwise, 3 upside down. Sony bodies write only these and
    1; the mirrored ones (2, 4, 5, 7) never come out of a camera.
    """
    if orientation == 6:
        return v, 1.0 - u
    if orientation == 8:
        return 1.0 - v, u
    if orientation == 3:
        return 1.0 - u, 1.0 - v
    return u, v


def sensor_corners(
    geometry: GeometryParams, width: float, height: float, orientation: int | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The crop's top-left and bottom-right corners as the sensor sees them.

    ``width`` and ``height`` are the *upright* picture's. Returns normalised
    ``((left, top), (right, bottom))`` in the sensor frame -- Lightroom's
    ``CropLeft/Top`` and ``CropRight/Bottom``.
    """
    upright = frame_corners(geometry, width, height)
    points = [to_sensor(u, v, orientation) for u, v in upright]
    swapped = orientation in (6, 8)
    sensor_w, sensor_h = (height, width) if swapped else (width, height)
    theta = math.radians(geometry.rotation_deg)
    c, s = math.cos(theta), math.sin(theta)

    def along(point: tuple[float, float]) -> float:
        # Undo the rotation in pixel units and rank by x + y: the top-left
        # corner of the rectangle, in its own axes, has the smallest sum.
        x = (point[0] - 0.5) * sensor_w
        y = (point[1] - 0.5) * sensor_h
        return (c * x + s * y) + (-s * x + c * y)

    first = min(range(4), key=lambda index: along(points[index]))
    return points[first], points[(first + 2) % 4]

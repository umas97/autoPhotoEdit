# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Panning shots of a real frame, for the panorama tests.

The world is a real A7 III frame on a plane in front of the camera, seen by a
pinhole wide enough to hold it all. The shots are what a narrower camera sees
turning about its centre -- yaw, a little pitch and roll, as a hand-held pan
has -- which is exactly the model a panorama assumes: a homography per frame,
``H = K_world R K_shot^-1``. The world plane seen through the plane
projection is the source frame again, which gives the tests a truth.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["rotation", "shoot"]


def rotation(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def shoot(
    world: np.ndarray, world_focal: float, size: tuple[int, int], focal: float,
    turns: list[tuple[float, float, float]], *, gains: list[float] | None = None,
) -> list[np.ndarray]:
    """One frame of ``size`` (width, height) per ``(yaw, pitch, roll)`` in degrees."""
    height_w, width_w = world.shape[:2]
    k_world = np.array([[world_focal, 0, width_w / 2], [0, world_focal, height_w / 2], [0, 0, 1]])
    width, height = size
    k_shot = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    frames = []
    for index, turn in enumerate(turns):
        homography = k_world @ rotation(*turn) @ np.linalg.inv(k_shot)
        frame = cv2.warpPerspective(
            world, homography, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        if gains is not None:
            frame = frame * np.float32(gains[index])
        frames.append(np.ascontiguousarray(frame, dtype=np.float32))
    return frames

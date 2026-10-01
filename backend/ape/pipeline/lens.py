# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lens correction: vignetting, distortion and lateral chromatic aberration.

The first stage after the decode (docs/SPEC.md section 6.2), because all three are
properties of the optics and describe the frame as the sensor recorded it:
vignetting is a gain that belongs in linear light, and distortion is defined on
the uncropped, unrotated frame -- the straightening detector looks for straight
lines, and it can only find them once the lens has stopped bending them.

The profiles come from lensfun, through ``lensdb``. The corrections are applied
here rather than by lensfun's own pixel loops for two reasons:

* **Memory.** lensfun's coordinate buffer for a 24 MP frame with TCA is
  (H, W, 3, 2) float32 -- 576 MB, a third of a worker's budget. It is computed
  here in bands of rows and consumed by ``cv2.remap`` band by band.
* **Resolution independence** (section 6.2, test 1). Vignetting is sampled once
  on a small grid of fixed size and stretched to the frame, so the proxy and the
  export apply the same gain field; distortion is computed on the frame's own
  coordinates, which lensfun normalises to the frame's size.

``scale=0`` asks lensfun for the smallest zoom that leaves no empty border after
correcting barrel distortion, the same thing every raw developer does. It
depends only on the geometry, so it is the same at every resolution.

Colour space in and out: linear, scene-referred, any RGB (the corrections are
per-pixel gains and a resampling, blind to the primaries), float32.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

import cv2
import numpy as np

from ..lensdb import LensIdentity, LensProfile, database, lensfun_available, resolve

__all__ = ["apply", "profile_for"]

_log = logging.getLogger(__name__)

#: Long edge of the grid the vignetting gain is sampled on. The gain is a
#: polynomial in the radius with three terms; 96 samples across the frame put
#: the interpolation error below 0.1% of the gain, far under what an eye sees
#: in a gradient spanning the whole frame.
_VIGNETTING_GRID = 96

#: Rows resampled at once. 256 rows of a 6000 px frame with per-channel TCA
#: carry 37 MB of coordinates, instead of the 576 MB of the whole frame.
_BAND_ROWS = 256

#: Spacing of the rows on which lensfun computes coordinates exactly.
_ROW_STEP = 16


def profile_for(identity: LensIdentity | None) -> LensProfile | None:
    if identity is None or not lensfun_available():
        return None
    return resolve(identity)


@lru_cache(maxsize=32)
def _lens_object(maker: str, model: str, stamp_key: int) -> object | None:
    for lens in database().lenses:
        if lens.maker == maker and lens.model == model:
            return lens
    return None


def _lens_for(profile: LensProfile) -> object | None:
    # ``id(database())`` changes when the database is reloaded after an update.
    return _lens_object(profile.maker, profile.model, id(database()))


def _modifier(lens, crop: float, width: int, height: int, identity: LensIdentity, flags: int):
    import lensfunpy

    modifier = lensfunpy.Modifier(lens, crop, width, height)
    modifier.initialize(
        float(identity.focal_length or lens.min_focal),
        float(identity.aperture or lens.min_aperture or 8.0),
        float(identity.distance),
        scale=0.0,
        targeom=lens.type,
        pixel_format=np.float32,
        flags=flags,
    )
    return modifier


def _vignetting_grid(
    lens, profile: LensProfile, identity: LensIdentity, shape
) -> np.ndarray | None:
    """The gain on a small grid spanning the frame, or ``None`` if there is none.

    Returned small: it is sampled per band at the corrected coordinates, which
    both applies it where lensfun defines it -- on the distorted frame -- and
    avoids a full-frame copy of the image just to multiply it.
    """
    import lensfunpy

    height, width = shape
    scale = _VIGNETTING_GRID / max(height, width)
    small_w = max(8, round(width * scale))
    small_h = max(8, round(height * scale))
    modifier = _modifier(
        lens, profile.crop_factor, small_w, small_h, identity, lensfunpy.ModifyFlags.VIGNETTING
    )
    grid = np.ones((small_h, small_w, 3), dtype=np.float32)
    if not modifier.apply_color_modification(grid):
        return None
    return np.ascontiguousarray(grid[..., 1])


def _sampled_rows(modifier, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Corrected source coordinates on every ``_ROW_STEP``-th row, exactly.

    The distortion field is a low-order polynomial of the radius, so between two
    sampled rows 16 px apart it is linear to within a thousandth of a pixel --
    and sampling it costs 12 ms on a 24 MP frame instead of the 200 ms of
    asking lensfun for every pixel.
    """
    count = max(2, math.ceil((height - 1) / _ROW_STEP) + 1)
    ys = np.linspace(0.0, height - 1.0, count)
    rows = np.stack(
        [modifier.apply_subpixel_geometry_distortion(0, float(y), width, 1)[0] for y in ys]
    )
    return ys, rows  # rows: (count, W, 3, 2)


def apply(img: np.ndarray, identity: LensIdentity | None, enabled: bool = True) -> np.ndarray:
    """Correct the optics of one frame, or return it untouched.

    Args:
        img: ``(H, W, 3)`` float32, linear scene-referred, upright.
        identity: camera, lens and shooting settings from the EXIF, plus the
            user's manual association if any. ``None`` is "unknown lens".
        enabled: ``EditParams.geometry.lens_correction``.

    Returns:
        A new ``(H, W, 3)`` float32 array, same size, same colour space; or
        ``img`` itself when there is nothing to correct. Never modifies ``img``.
    """
    if not enabled:
        return img
    profile = profile_for(identity)
    if profile is None:
        return img
    lens = _lens_for(profile)
    if lens is None:
        return img
    import lensfunpy

    height, width = img.shape[:2]
    gain = _vignetting_grid(lens, profile, identity, (height, width))
    has_geometry = bool(lens.calib_distortion) or bool(lens.calib_tca)
    if not has_geometry:
        if gain is None:
            return img
        full = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)
        return img * full[..., None]

    flags = (
        lensfunpy.ModifyFlags.DISTORTION
        | lensfunpy.ModifyFlags.TCA
        | lensfunpy.ModifyFlags.SCALE
    )
    modifier = _modifier(lens, profile.crop_factor, width, height, identity, flags)
    ys, rows = _sampled_rows(modifier, width, height)
    per_channel = bool(lens.calib_tca)
    channels = (0, 1, 2) if per_channel else (1,)
    # One small two-channel "image" per colour channel holding the
    # *displacement* -- corrected coordinate minus pixel position -- rather than
    # the coordinate. cv2.remap quantises its interpolation weights to 1/32:
    # applied to coordinates that advance 16 px between two sampled rows that is
    # half a pixel of error, applied to a displacement that changes by a few
    # tenths of a pixel it is a few thousandths.
    identity_rows = np.empty(rows.shape[:2] + (2,), dtype=np.float32)
    identity_rows[..., 0] = np.arange(width, dtype=np.float32)[None, :]
    identity_rows[..., 1] = ys.astype(np.float32)[:, None]
    fields = [np.ascontiguousarray(rows[:, :, c, :] - identity_rows) for c in channels]
    gains: list[np.ndarray] = []
    if gain is not None:
        # The gain belongs to the *source* position of each output pixel, so it
        # is read there -- on the sampled rows only, then interpolated with the
        # displacement. Reading it per pixel costs as much as the resampling.
        to_gain = np.array(
            [(gain.shape[1] - 1) / max(1, width - 1), (gain.shape[0] - 1) / max(1, height - 1)],
            dtype=np.float32,
        )
        for c in channels:
            where = np.ascontiguousarray(rows[:, :, c, :] * to_gain)
            gains.append(
                cv2.remap(gain, where, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            )

    out = np.empty_like(img)
    spacing = float(ys[1] - ys[0])
    column = np.arange(width, dtype=np.float32)
    identity = np.empty((_BAND_ROWS, width, 2), dtype=np.float32)
    identity[..., 0] = column[None, :]
    for top in range(0, height, _BAND_ROWS):
        rows_here = min(_BAND_ROWS, height - top)
        # Interpolating the sampled field is itself a remap: every output pixel
        # reads the field at its own column and at its fractional sample row.
        field_x = np.broadcast_to(column, (rows_here, width))
        field_y = np.broadcast_to(
            (np.arange(top, top + rows_here, dtype=np.float32) / spacing)[:, None],
            (rows_here, width),
        )
        field_x, field_y = np.ascontiguousarray(field_x), np.ascontiguousarray(field_y)
        here = identity[:rows_here]
        here[..., 1] = np.arange(top, top + rows_here, dtype=np.float32)[:, None]
        for index, channel in enumerate(channels):
            coords = cv2.remap(
                fields[index], field_x, field_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            coords += here
            source = img[..., channel] if per_channel else img
            corrected = cv2.remap(
                source, coords, None, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            if gains:
                factor = cv2.remap(
                    gains[index], field_x, field_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                corrected *= factor if per_channel else factor[..., None]
            if per_channel:
                out[top : top + rows_here, :, channel] = corrected
            else:
                out[top : top + rows_here] = corrected
    return out

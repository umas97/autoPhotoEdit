# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The engineered scene features of section 8.3.

A fixed-length vector per photo, the half of the scene description that needs
no model: what the EXIF says about the light, and what the neutral proxy says
about the tones and colours. Style learning (phase 6) regresses parameters on
it, the clustering of section 9.2 groups on it when the CLIP model is not
there, and the confidence of section 9.1 uses its ranges to tell
interpolation from extrapolation.

The order of :data:`FEATURE_NAMES` *is* the format. It is versioned: a vector of
another version is recomputed, never reinterpreted.

Input: the browsing proxy -- neutral development, sRGB, upright,
lens-corrected -- plus EXIF and the camera's as-shot white balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCALES",
    "SCENE_FEATURES_VERSION",
    "SceneInputs",
    "decode_features",
    "encode_features",
    "scene_features",
]

#: 2: ``tint`` changed sign with the white balance (``raw/whitepoint.py``).
#: Version 1 still decodes, flipped: re-analysing every photo for a sign would
#: cost minutes of CPU for nothing.
SCENE_FEATURES_VERSION = 2

#: Eight hue sectors, the HSL bands of ``params.HSL_BANDS`` in spirit: a
#: histogram of where the colour of the frame is.
_HUE_BINS = 8

FEATURE_NAMES: tuple[str, ...] = (
    "ev100",  # scene brightness from the exposure triangle
    "log_iso",
    "aperture",
    "log_focal",
    "lum_mean",
    "lum_median",
    "lum_p01",
    "lum_p05",
    "lum_p50",
    "lum_p95",
    "lum_p99",
    "saturation_mean",
    "cct_mired",  # as-shot white balance, in mired so that distances are perceptual
    "tint",
    "clipped_high",
    "clipped_low",
    "lum_entropy",
    *(f"hue_{i}" for i in range(_HUE_BINS)),
    "faces_count",
    "faces_area",
    "faces_known",  # 0 while no face detector is installed: the two above mean nothing
    "outdoor",  # 0..1, from the light level and its colour
)

#: The smallest difference in each feature that means anything, in its own
#: units. Clustering divides by the project's spread, and a feature that barely
#: varies across a project -- one light, one lens -- would otherwise have its
#: noise blown up into distance: 0.01 of jitter over an interquartile range of
#: 0.01 is a full unit. These floors are what "the same" means per feature.
FEATURE_SCALES: dict[str, float] = {
    "ev100": 1.0,  # a stop
    "log_iso": 0.5,  # half a stop of gain
    "aperture": 1.0,
    "log_focal": 0.3,  # about 23% of focal length
    "saturation_mean": 0.05,
    "cct_mired": 15.0,  # about 500 K around daylight: a visible shift
    "tint": 5.0,
    "clipped_high": 0.02,
    "clipped_low": 0.02,
    "lum_entropy": 0.3,
    "faces_count": 1.0,
    "faces_area": 0.05,
    "faces_known": 1.0,
    "outdoor": 0.2,
}
for _name in FEATURE_NAMES:
    if _name.startswith("lum_") and _name != "lum_entropy":
        FEATURE_SCALES[_name] = 0.05  # 5% of the tonal range
    elif _name.startswith("hue_"):
        FEATURE_SCALES[_name] = 0.05
assert set(FEATURE_SCALES) == set(FEATURE_NAMES)

_MAGIC = b"APSF"
_TINT = FEATURE_NAMES.index("tint")


@dataclass(slots=True)
class SceneInputs:
    iso: float | None = None
    aperture: float | None = None
    shutter: float | None = None
    focal_length: float | None = None
    as_shot_temperature_k: float | None = None
    as_shot_tint: float | None = None


def _ev100(inputs: SceneInputs) -> float:
    """``log2(N^2 / t) - log2(ISO / 100)``: 15 is sunny snow, 5 a lit room."""
    if not (inputs.aperture and inputs.shutter and inputs.iso):
        return math.nan
    return math.log2(inputs.aperture**2 / inputs.shutter) - math.log2(inputs.iso / 100.0)


def _outdoor(ev100: float, cct: float | None) -> float:
    """A soft guess at "outdoors", which section 8.3 lists as an indicator.

    Daylight is bright and near 5500 K; interiors are dim and warm. Each cue
    alone fails (an overcast dusk, a window-lit room), which is why it is a
    probability and why the regression, not this function, decides what it
    means for the edit.
    """
    bright = 0.5 if math.isnan(ev100) else 1.0 / (1.0 + math.exp(-(ev100 - 9.5) / 1.2))
    if cct is None:
        return bright
    daylight = 1.0 / (1.0 + math.exp(-(cct - 4300.0) / 400.0))
    return 0.65 * bright + 0.35 * daylight


def scene_features(image: np.ndarray, inputs: SceneInputs) -> np.ndarray:
    """The vector of :data:`FEATURE_NAMES` for one photo.

    Args:
        image: the proxy, ``(H, W, 3)`` uint8 sRGB. Any size; it is reduced.
        inputs: EXIF and as-shot white balance.

    Returns:
        float32 of length ``len(FEATURE_NAMES)``. Unknown values are NaN, and
        the consumers decide how to fill them -- a zero would be a lie.
    """
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = 512 / max(height, width)
    if scale < 1:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] / 255.0  # OpenCV scales L* 0..100 to 0..255
    chroma_a = lab[..., 1] - 128.0
    chroma_b = lab[..., 2] - 128.0
    chroma = np.hypot(chroma_a, chroma_b)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    percentiles = np.percentile(lightness, [1, 5, 50, 95, 99])
    histogram, _ = np.histogram(lightness, bins=64, range=(0.0, 1.0))
    p = histogram / max(1, histogram.sum())
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    hue = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV hue is 0..180
    sector = (np.floor(((hue + 360.0 / _HUE_BINS / 2) % 360.0) / (360.0 / _HUE_BINS))).astype(int)
    weights = chroma.ravel()
    hue_hist = np.bincount(sector.ravel(), weights=weights, minlength=_HUE_BINS)[:_HUE_BINS]
    hue_hist = hue_hist / max(1e-9, hue_hist.sum())

    peak = image.max(axis=2)
    trough = image.min(axis=2)
    ev = _ev100(inputs)
    cct = inputs.as_shot_temperature_k
    values = [
        ev,
        math.log2(inputs.iso) if inputs.iso else math.nan,
        inputs.aperture if inputs.aperture else math.nan,
        math.log2(inputs.focal_length) if inputs.focal_length else math.nan,
        float(lightness.mean()),
        float(np.median(lightness)),
        *(float(v) for v in percentiles),
        float(hsv[..., 1].mean() / 255.0),
        1e6 / cct if cct else math.nan,
        inputs.as_shot_tint if inputs.as_shot_tint is not None else math.nan,
        float((peak >= 250).mean()),
        float((trough <= 4).mean()),
        entropy,
        *(float(v) for v in hue_hist),
        0.0,
        0.0,
        0.0,
        _outdoor(ev, cct),
    ]
    assert len(values) == len(FEATURE_NAMES)
    return np.asarray(values, dtype=np.float32)


def encode_features(vector: np.ndarray) -> bytes:
    """``Photo.scene_features``: a four-byte tag, the version, the floats."""
    return _MAGIC + bytes([SCENE_FEATURES_VERSION]) + vector.astype("<f4").tobytes()


def decode_features(blob: bytes | None) -> np.ndarray | None:
    """The vector, or ``None`` if absent or of another version (recompute it)."""
    if not blob or blob[:4] != _MAGIC or blob[4] not in (1, SCENE_FEATURES_VERSION):
        return None
    vector = np.frombuffer(blob[5:], dtype="<f4")
    if len(vector) != len(FEATURE_NAMES):
        return None
    vector = vector.copy()
    if blob[4] == 1:
        vector[_TINT] = np.float32(0.0) - vector[_TINT]
    return vector

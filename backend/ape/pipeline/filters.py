# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial filters shared by the pipeline operations.

Every filter here takes its radius as a **fraction of the image's long edge**,
never in pixels. That is what makes the operations resolution-independent
(docs/SPEC.md section 6.2): the same ``EditParams`` applied to a 1024 px preview and
to the 24 MP original describe the same physical neighbourhood, so the two
renders match (test 1 in section 13).

The guided filter is implemented here rather than taken from ``cv2.ximgproc``
because that module lives in opencv-contrib, which the headless wheel we depend
on does not ship.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "box_filter",
    "gaussian_blur",
    "guided_filter",
    "radius_to_pixels",
    "resize_long_edge",
    "sigma_to_pixels",
]

# Below this the filters degenerate to a no-op; keeping a floor avoids a special
# case in every caller.
_MIN_RADIUS_PX = 1

#: Radius, in pixels, that the guided filter works at once it has downsampled.
#: Twelve is the smallest value at which the box filters still resolve the
#: neighbourhood well enough for the linear coefficients to match the
#: full-resolution ones; below it the approximation starts to show on edges.
_FAST_RADIUS = 12.0


def _working_long_edge(radius_rel: float) -> int:
    """Long edge at which ``radius_rel`` measures ``_FAST_RADIUS`` pixels.

    Depends on the radius alone, never on the input size -- that is the whole
    point: every resolution of the same image lands on the same number.
    """
    return max(2 * int(_FAST_RADIUS), int(round(_FAST_RADIUS / max(radius_rel, 1e-9))))


def long_edge(shape: tuple[int, ...]) -> int:
    return int(max(shape[0], shape[1]))


def radius_to_pixels(radius_rel: float, shape: tuple[int, ...]) -> int:
    """Convert a long-edge-relative radius to an odd pixel radius >= 1."""
    return max(_MIN_RADIUS_PX, int(round(radius_rel * long_edge(shape))))


def sigma_to_pixels(sigma_rel: float, shape: tuple[int, ...]) -> float:
    """Convert a long-edge-relative Gaussian sigma to pixels, with a floor of 0.5."""
    return max(0.5, sigma_rel * long_edge(shape))


def box_filter(img: np.ndarray, radius_px: int) -> np.ndarray:
    ksize = 2 * int(radius_px) + 1
    return cv2.boxFilter(
        img, -1, (ksize, ksize), normalize=True, borderType=cv2.BORDER_REFLECT
    )


def gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur with a kernel sized from sigma, reflecting at the border."""
    if sigma_px < 0.3:
        return img
    ksize = int(2 * round(3.0 * sigma_px) + 1)
    return cv2.GaussianBlur(
        img, (ksize, ksize), sigma_px, borderType=cv2.BORDER_REFLECT
    )


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius_rel: float,
    eps: float,
    shape: tuple[int, ...] | None = None,
    subsample: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Edge-preserving smoothing of ``src`` guided by ``guide`` (He et al., 2010).

    Args:
        guide: single-channel float32 guide, typically the luminance.
        src: float32 image to smooth, ``(H, W)`` or ``(H, W, C)``. May be
            ``guide`` itself. Several channels sharing one guide go in one call:
            the guide is then reduced and its statistics computed once, and
            OpenCV filters the channels with the same arithmetic as one by one.
        radius_rel: filter radius as a fraction of the image long edge.
        eps: regularisation; smaller means more edge-preserving. Expressed in the
            squared units of ``guide``, so callers must pass it in the same range
            the guide lives in.
        shape: shape the relative radius refers to; defaults to ``guide.shape``.
        subsample: use the fast O(N) variant, computing the linear coefficients
            on a downsampled copy. Visually indistinguishable and several times
            faster at full resolution.
        out: where to write the result, shaped and typed like ``src``; may be
            ``src`` itself, which is read only before the result is written.
            Saves a full frame per channel when the caller no longer needs the
            input.

    Returns:
        float32 array shaped like ``src``.
    """

    shape = shape or guide.shape
    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)

    native = long_edge(guide.shape)
    # The working resolution is chosen from the *radius*, not from a subsampling
    # factor: whatever the input size, the filter runs on an image in which the
    # radius is _FAST_RADIUS pixels. Two renders of the same scene at different
    # resolutions therefore reach this point with the same small image and the
    # same integer radius, which is what makes the operation exactly scale
    # invariant instead of merely approximately so (test 1 of section 13). It is
    # also where the fast variant's O(N) behaviour comes from.
    target = min(native, _working_long_edge(radius_rel)) if subsample else native
    downsampled = target < native
    if downsampled:
        scale = target / native
        small_size = (
            max(1, int(round(guide.shape[1] * scale))),
            max(1, int(round(guide.shape[0] * scale))),
        )
        g = cv2.resize(guide, small_size, interpolation=cv2.INTER_AREA)
        r = max(_MIN_RADIUS_PX, int(round(radius_rel * target)))
    else:
        g, r = guide, radius_to_pixels(radius_rel, shape)

    # The guide's statistics, once for every channel of ``src``.
    mean_g = box_filter(g, r)
    var_g = box_filter(g * g, r) - mean_g * mean_g
    var_g += np.float32(eps)
    full_size = (guide.shape[1], guide.shape[0])

    def filter_one(channel: np.ndarray, target_out: np.ndarray | None) -> np.ndarray:
        p = channel
        if downsampled:
            p = cv2.resize(channel, small_size, interpolation=cv2.INTER_AREA)
        mean_p = box_filter(p, r)
        cov_gp = box_filter(g * p, r) - mean_g * mean_p
        a = cov_gp / var_g
        b = mean_p - a * mean_g
        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)
        if downsampled:
            mean_a = cv2.resize(mean_a, full_size, interpolation=cv2.INTER_LINEAR)
            mean_b = cv2.resize(mean_b, full_size, interpolation=cv2.INTER_LINEAR)
        # ``mean_a * guide + mean_b``, in place on arrays of our own: at full
        # resolution each temporary here is a whole frame.
        result = mean_a if target_out is None else target_out
        np.multiply(mean_a, guide, out=result)
        result += mean_b
        return result

    if src.ndim == 2:
        return filter_one(src, out)
    # One channel at a time, the whole filter: OpenCV filters and resizes the
    # channels of an image independently, so this is the same arithmetic --
    # with the temporaries of one channel instead of three. ``out`` may be
    # ``src``: each channel is read before it is written.
    result = out if out is not None else np.empty_like(src)
    for index in range(src.shape[-1]):
        channel = np.ascontiguousarray(src[..., index])
        result[..., index] = filter_one(channel, None)
    return result


def resize_long_edge(img: np.ndarray, target_long_edge: int) -> np.ndarray:
    """Resize so the long edge matches, preserving aspect ratio. No-op if already smaller."""
    height, width = img.shape[:2]
    current = max(height, width)
    if current <= target_long_edge:
        return img
    scale = target_long_edge / current
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    # INTER_AREA is the correct choice for downscaling: it averages, so it does
    # not alias high-frequency detail into the preview.
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

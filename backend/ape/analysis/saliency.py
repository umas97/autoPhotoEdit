# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the eye goes: the saliency map the crop proposal is scored on (section 6.4).

``cv2.saliency`` lives in OpenCV's contrib modules, which the headless wheel of
section 4 does not ship, and an ONNX model for this would be a download for a
feature that only *proposes*. So the map is built here from three classical
cues, each cheap on a 256 px image, each covering what the others miss:

* **colour contrast** (Achanta et al., 2009, "frequency-tuned"): how far each
  pixel's colour is from the frame's average, in Lab. A red jacket on snow.
* **spectral residual** (Hou & Zhang, 2007): what the log-spectrum has beyond
  its smooth trend -- the part of the image that is not "more of the same". A
  lone tree on a hillside, a figure on a beach.
* **focus**: local Laplacian energy. With a lens wide open the subject is the
  sharp part of the frame, and the user's photos are mostly at f/2.8; neither
  cue above knows that a blurred background is background.

Each is normalised to 0..1 and they are averaged. Output: float32, 0..1, the
size asked for.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["saliency_map"]

#: The cues are computed at this long edge. Saliency is about regions, not
#: detail, and every published method works at this scale or below.
_WORK_EDGE = 256

#: Focus counts most: most of the user's frames are shot wide open, and there
#: the sharp region *is* the subject. Colour next; the spectral residual is the
#: weakest cue alone and mostly breaks ties.
_W_FOCUS, _W_COLOUR, _W_SPECTRAL = 0.45, 0.35, 0.20


def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = float(np.percentile(values, 1)), float(np.percentile(values, 99.5))
    if high - low < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _colour_contrast(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    smooth = cv2.GaussianBlur(lab, (0, 0), 1.5)
    mean = np.median(lab.reshape(-1, 3), axis=0)
    # Lightness at a third of the weight of colour: in the user's frames the
    # brightest thing is usually sky or snow, which is far from the average
    # and is background. A red jacket is far from it in a and b.
    diff = (smooth - mean) * np.array([0.35, 1.0, 1.0], dtype=np.float32)
    return np.linalg.norm(diff, axis=2)


def _spectral_residual(grey: np.ndarray) -> np.ndarray:
    small = cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-6).astype(np.float32)
    phase = np.angle(spectrum)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    back = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    back = cv2.GaussianBlur(back.astype(np.float32), (0, 0), 2.5)
    return cv2.resize(back, (grey.shape[1], grey.shape[0]), interpolation=cv2.INTER_LINEAR)


def _focus(grey: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(grey.astype(np.float32), cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(lap * lap, (0, 0), 4.0)
    return np.sqrt(energy)


def _centre_prior(shape: tuple[int, int]) -> np.ndarray:
    """A broad bump in the middle, 0.5 in the corners.

    Photographers put subjects away from the edges far more often than on
    them; every saliency benchmark rewards this prior, and here it mostly keeps
    a bright patch of sky in a corner from outweighing the person in the frame.
    """
    height, width = shape
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    return 0.5 + 0.5 * np.exp(-(xs**2 + ys**2) / (2 * 0.6**2)) / 1.0


def saliency_map(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Saliency of an sRGB image (uint8, or float in 0..1).

    Args:
        image: ``(H, W, 3)`` display-referred RGB, upright.
        size: ``(width, height)`` of the result; the work size when ``None``.

    Returns:
        ``(h, w)`` float32 in 0..1.
    """
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = _WORK_EDGE / max(height, width)
    work = cv2.resize(
        image, (max(8, round(width * scale)), max(8, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    grey = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    combined = (
        _W_COLOUR * _normalise(_colour_contrast(work))
        + _W_SPECTRAL * _normalise(_spectral_residual(grey))
        + _W_FOCUS * _normalise(_focus(grey))
    )
    combined = _normalise(combined * _centre_prior(combined.shape))
    if size is not None and (size[0], size[1]) != (combined.shape[1], combined.shape[0]):
        combined = cv2.resize(combined, size, interpolation=cv2.INTER_LINEAR)
    return combined

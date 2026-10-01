# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic camera previews with known defects, for the culling tests.

Test 9 of section 13 asks for a set with known rejects: ten photos that are
out of focus, moved or blown among a hundred. A real card with that property
would have to be shot on purpose, and even then "known" would mean "judged by
eye". Synthetic frames make the ground truth exact: a frame is out of focus
because it was blurred here, by this much.

The scenes are built to have what a sharpness measure feeds on in a real
photograph: detail at every scale (fractal texture), hard edges in every
orientation (shapes and thin lines), and large smooth areas (sky, gradients).
They are encoded as JPEG at the A7 III preview size, because that is what the
analysis sees in the program -- a camera JPEG with its blocking and its
rounding -- and a test on clean floats would be kinder than reality.

Every defect is chosen on the side of "clearly": section 7.4's conservative
mode discards what is compromised *beyond recovery*, and a test that asked it
to catch a borderline blur would be testing a different mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np

__all__ = [
    "PREVIEW_SHAPE",
    "SyntheticShot",
    "build_culling_set",
    "defocus",
    "move",
    "natural_scene",
    "overexpose",
]

#: The A7 III's embedded preview, landscape.
PREVIEW_SHAPE = (1080, 1616)

_LINEAR = np.where(
    np.arange(256) / 255.0 <= 0.04045,
    (np.arange(256) / 255.0) / 12.92,
    ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4,
)


def _jpeg(image: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _fractal(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """1/f-like texture in [0, 1]: octaves of noise, each twice as fine."""
    height, width = shape
    total = np.zeros(shape, dtype=np.float32)
    amplitude, weight = 1.0, 0.0
    for octave in range(2, 9):
        cells = 2**octave
        coarse = rng.random((cells, max(2, int(cells * width / height)))).astype(np.float32)
        total += amplitude * cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        weight += amplitude
        # Slower than 1/f on purpose: real foliage, stone and fabric carry
        # more fine detail than an ideal 1/f spectrum, and the previews of
        # ``tests/fixtures`` measure about four times the fine variation of a
        # texture that falls off faster.
        amplitude *= 0.85
    total /= weight
    total -= total.min()
    return total / max(float(total.max()), 1e-6)


def natural_scene(seed: int, shape: tuple[int, int] = PREVIEW_SHAPE) -> np.ndarray:
    """A sharp, well exposed 8-bit BGR frame that is not any other seed's frame."""
    rng = np.random.default_rng(seed)
    height, width = shape
    # Drawn at the preview's own size, with antialiased edges: a lens never
    # draws a staircase, and supersampling would cost four times as much for
    # a difference no measure here can see.
    big = (height, width)
    scale = 0.5

    top = rng.uniform(40, 200, 3)
    bottom = rng.uniform(20, 160, 3)
    ramp = np.linspace(0.0, 1.0, big[0], dtype=np.float32)[:, None, None]
    image = (top * (1 - ramp) + bottom * ramp).repeat(big[1], axis=1).astype(np.float32)

    texture = _fractal(rng, big)[..., None]
    tint = rng.uniform(0.6, 1.4, 3).astype(np.float32)
    image = image * (0.25 + 1.5 * texture * tint)

    for _ in range(int(rng.integers(160, 260))):
        colour = tuple(float(c) for c in rng.uniform(5, 255, 3) * rng.uniform(0.3, 1.6))
        centre = (int(rng.integers(0, big[1])), int(rng.integers(0, big[0])))
        kind = rng.integers(0, 3)
        if kind == 0:
            radius = max(2, int(rng.integers(6, 120) * scale))
            cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
        elif kind == 1:
            half = (int(rng.integers(6, 140) * scale), int(rng.integers(6, 140) * scale))
            angle = float(rng.uniform(0, 180))
            box = cv2.boxPoints(((centre[0], centre[1]), (2 * half[0], 2 * half[1]), angle))
            cv2.fillPoly(image, [box.astype(np.int32)], colour, cv2.LINE_AA)
        else:
            end = (int(rng.integers(0, big[1])), int(rng.integers(0, big[0])))
            thickness = max(1, int(rng.integers(2, 9) * scale))
            cv2.line(image, centre, end, colour, thickness, cv2.LINE_AA)

    # Brought down to the exposure of a real frame: the previews of
    # ``tests/fixtures`` sit between -2.8 and +0.8 EV of a normal rendering,
    # and without this the scenes would start a stop and a half brighter.
    image *= 0.6
    return _jpeg(np.clip(image, 0, 255).astype(np.uint8))


def defocus(image: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    """A focus missed by a clear margin: about 13 px of blur on the sensor."""
    return _jpeg(cv2.GaussianBlur(image, (0, 0), sigma))


def move(image: np.ndarray, length: int = 21, angle: float = 30.0) -> np.ndarray:
    """Camera shake: a straight streak, the same over the whole frame."""
    kernel = np.zeros((length, length), dtype=np.float32)
    centre = length // 2
    dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for t in np.linspace(-centre, centre, 4 * length):
        kernel[int(round(centre + t * dy)), int(round(centre + t * dx))] = 1.0
    return _jpeg(cv2.filter2D(image, -1, kernel / kernel.sum()))


def _shift_ev(image: np.ndarray, ev: float) -> np.ndarray:
    linear = np.clip(_LINEAR[image] * 2.0**ev, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return _jpeg((encoded * 255.0 + 0.5).astype(np.uint8))


def overexpose(image: np.ndarray, ev: float = 3.5) -> np.ndarray:
    """Well past the two stops a RAW gives back (section 7.3)."""
    return _shift_ev(image, ev)


def underexpose(image: np.ndarray, ev: float) -> np.ndarray:
    return _shift_ev(image, -ev)


def shallow_depth(image: np.ndarray, seed: int) -> np.ndarray:
    """A sharp subject on a background of pure blur: a *good* f/1.8 portrait."""
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    background = cv2.GaussianBlur(image, (0, 0), 7)
    mask = np.zeros((height, width), dtype=np.float32)
    centre = (int(width * rng.uniform(0.35, 0.65)), int(height * rng.uniform(0.35, 0.65)))
    cv2.ellipse(mask, centre, (width // 7, height // 4), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)[..., None]
    return _jpeg((image * mask + background * (1.0 - mask)).astype(np.uint8))


def noisy(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    grain = np.random.default_rng(seed).normal(0.0, sigma, image.shape)
    return _jpeg(np.clip(image + grain, 0, 255).astype(np.uint8))


def hazy(image: np.ndarray, amount: float = 0.55) -> np.ndarray:
    """Fog: every edge still sharp, and very little contrast left in it."""
    grey = np.full_like(image, 170)
    return _jpeg(cv2.addWeighted(image, 1.0 - amount, grey, amount, 0.0))


def burst_frame(image: np.ndarray, index: int, seed: int) -> np.ndarray:
    """One frame of a burst: the subject a little further on, fresh noise."""
    rng = np.random.default_rng((seed, index))
    width = image.shape[1]
    dx = int(round(width * 0.006 * index + rng.uniform(-2, 2)))
    dy = int(rng.integers(-3, 4))
    shifted = _shift(image, dx, dy)
    frame = _shift_ev(shifted, float(rng.uniform(-0.15, 0.15)))
    return noisy(frame, 1.5, seed * 100 + index)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Whole-pixel translation. A fractional one would interpolate, and
    bilinear interpolation along x is a genuine two-pixel horizontal blur --
    measurably a little motion, which a real burst frame does not have."""
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


@dataclass
class SyntheticShot:
    id: int
    image: np.ndarray
    shot_at: datetime
    #: ``None`` for a good photo, otherwise the defect it was given.
    defect: str | None = None
    #: The burst this frame belongs to, by construction.
    burst: int | None = None
    #: A frame of a burst made worse on purpose: it must not be the one kept.
    weaker: bool = False
    tags: dict = field(default_factory=dict)


def build_culling_set() -> list[SyntheticShot]:
    """The hundred photos of test 9.

    * 70 single shots, each its own scene, among them the hard cases a naive
      culler gets wrong: shallow depth of field, dark but recoverable, noisy,
      foggy;
    * 5 bursts of 4 frames, one frame of each softened slightly -- the frame a
      person would not choose;
    * 10 rejects: 4 out of focus, 3 moved, 3 blown.

    Timestamps are whole seconds, as a Sony writes them. Singles are ten
    seconds apart except for one pair of *different* scenes shot one second
    apart, which must not become a burst.
    """
    shots: list[SyntheticShot] = []
    clock = datetime(2026, 2, 26, 10, 0, 0)
    seed = 1000

    def add(image, **kwargs):
        shots.append(SyntheticShot(id=len(shots) + 1, image=image, shot_at=clock, **kwargs))

    hard_cases = {
        0: lambda img, s: shallow_depth(img, s),
        1: lambda img, s: shallow_depth(img, s),
        2: lambda img, s: shallow_depth(img, s),
        3: lambda img, s: underexpose(img, 2.0),
        4: lambda img, s: underexpose(img, 2.5),
        5: lambda img, s: noisy(img, 2.0, s),
        6: lambda img, s: noisy(img, 2.5, s),
        7: lambda img, s: hazy(img),
    }
    for index in range(70):
        seed += 1
        image = natural_scene(seed)
        if index in hard_cases:
            image = hard_cases[index](image, seed)
        add(image)
        # One pair of different scenes a second apart (indices 40 and 41).
        clock += timedelta(seconds=1 if index == 40 else 10)

    for burst in range(5):
        seed += 1
        base = natural_scene(seed)
        for frame in range(4):
            image = burst_frame(base, frame, seed)
            weaker = frame == 2
            if weaker:
                # Slightly soft, well short of "out of focus": the frame a
                # person would not pick, not a frame anyone would discard.
                image = _jpeg(cv2.GaussianBlur(image, (0, 0), 0.8))
            add(image, burst=burst, weaker=weaker)
            clock += timedelta(seconds=frame % 2)  # 0, 1, 0, 1: all under two seconds
        clock += timedelta(seconds=15)

    rejects = [
        ("out_of_focus", lambda img: defocus(img, 3.5)),
        ("out_of_focus", lambda img: defocus(img, 4.0)),
        ("out_of_focus", lambda img: defocus(img, 3.0)),
        ("out_of_focus", lambda img: defocus(img, 5.0)),
        ("motion_blur", lambda img: move(img, 21, 30)),
        ("motion_blur", lambda img: move(img, 25, 100)),
        ("motion_blur", lambda img: move(img, 19, 160)),
        ("overexposed", lambda img: overexpose(img, 3.5)),
        ("overexposed", lambda img: overexpose(img, 4.0)),
        ("overexposed", lambda img: overexpose(img, 3.5)),
    ]
    for defect, damage in rejects:
        seed += 1
        add(damage(natural_scene(seed)), defect=defect)
        clock += timedelta(seconds=10)
    return shots

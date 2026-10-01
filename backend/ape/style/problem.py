# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One pair as an optimisation problem: caches, renders, losses, render counter.

The per-pixel stages -- white balance to colour -- run on a fixed sample of the
masked pixels, resumed from the denoise's output, and cost about 3 ms; a full
render at 512 px costs about 60. The two spatial controls are approximated on
the sample with what the last full render froze: the base layer of the local
tone as a ratio to each pixel's own luminance, and the gain clarity applied.
See ``learn.py`` for the order in which the stages use it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..pipeline.colorspace import luminance
from ..pipeline.ops.local_contrast import local_tone_gain, tone_base
from ..pipeline.params import EditParams
from ..pipeline.render import resume
from ..raw.decode import DecodedRaw
from . import cmaes
from . import vector as sv
from .colordiff import delta_e_2000, srgb_to_lab
from .loss import (
    W_CHROMA,
    W_CONTRAST,
    W_DELTA_E,
    W_LUMA,
    W_REG,
    blur,
    gradient_spread,
    loss_terms,
    to_display,
)

__all__ = ["SAMPLE_PIXELS", "PairImages", "Problem"]

#: Pixels in the per-pixel sample. 20 000 of a 512 x 341 frame: the ΔE of the
#: sample agrees with the full frame's within 0.05 on the user's pairs.
SAMPLE_PIXELS = 20_000


@dataclass(slots=True)
class PairImages:
    """One pair, ready to be compared. Built by ``style/sample.py``."""

    #: The RAW at the comparison size, as the decoder produced it.
    decoded: DecodedRaw
    #: The reference in the neutral frame, sRGB float in [0, 1].
    reference: np.ndarray
    mask: np.ndarray
    context: sv.StyleContext


class Problem:
    """The pair, the caches, the loss, and the render counter."""

    def __init__(self, pair: PairImages, prior: np.ndarray | None = None) -> None:
        self.pair = pair
        self.prior = sv.NEUTRAL if prior is None else np.asarray(prior, dtype=np.float64)
        self.decoded = pair.decoded
        self.context = pair.context
        neutral = EditParams()
        # The denoise is the last stage a style does not touch: everything
        # before it is computed once. Blurred to the comparison scale, like
        # the reference below (see COMPARE_SIGMA).
        pre = resume(pair.decoded.rgb, pair.decoded, neutral, after=None, until="noise")
        self.pre_sharp = pre
        self.pre = blur(pre)
        reference = blur(pair.reference)
        where = np.flatnonzero(pair.mask.ravel())
        step = max(1, len(where) // SAMPLE_PIXELS)
        self.sample_index = where[::step]
        self.pre_sample = self.pre.reshape(-1, 3)[self.sample_index][:, None, :]
        self.ref_sample = reference.reshape(-1, 3)[self.sample_index]
        self.ref_lab_sample = srgb_to_lab(self.ref_sample)
        self.ref_display_sample = to_display(self.ref_sample)
        self.ref_lab = srgb_to_lab(reference)
        self.ref_spread = gradient_spread(self.ref_lab[..., 0], pair.mask)
        self.clarity_gain = np.ones((len(self.sample_index), 1, 1), np.float32)
        self.base_ratio = np.ones(len(self.sample_index), np.float32)
        self.evaluations = 0
        self.span = sv.BOUNDS[:, 1] - sv.BOUNDS[:, 0]

    # -- rendering -------------------------------------------------------------

    def params(self, vector: np.ndarray) -> EditParams:
        return sv.to_params(vector, self.context)

    def sample_display(self, vector: np.ndarray) -> np.ndarray:
        """Display-referred sample after the local-contrast stage, ``(N, 1, 3)``.

        The two spatial controls are approximated with what the last full render
        froze: the base layer as a ratio to each pixel's own luminance, and the
        gain clarity applied.
        """
        self.evaluations += 1
        out = resume(
            self.pre_sample, self.decoded, self.params(vector), after="noise", until="color"
        )
        shadows = vector[sv.index("local_shadows")]
        highlights = vector[sv.index("local_highlights")]
        if abs(shadows) > 1e-6 or abs(highlights) > 1e-6:
            base = luminance(out)[:, 0] * self.base_ratio
            out = out * local_tone_gain(base, shadows, highlights)[:, None, None]
        return np.clip(out * self.clarity_gain, 0.0, 1.0)

    def freeze_base(self, vector: np.ndarray) -> None:
        """Refresh the base-layer ratio from a full render of ``vector``."""
        self.evaluations += 1
        out = resume(self.pre, self.decoded, self.params(vector), after="noise", until="color")
        luma = luminance(out)
        base = tone_base(luma, out.shape)
        ratio = base / np.maximum(luma, 1e-4)
        self.base_ratio = ratio.reshape(-1)[self.sample_index].astype(np.float32)

    def sample_srgb(self, display: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return resume(display, self.decoded, self.params(vector), after="sharpen")[:, 0, :]

    def full(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full render: sRGB, and the display image before and after clarity."""
        self.evaluations += 1
        params = self.params(vector)
        before = resume(self.pre, self.decoded, params, after="noise", until="color")
        after = resume(before, self.decoded, params, after="color", until="local_contrast")
        srgb = resume(after, self.decoded, params, after="local_contrast")
        return srgb, before, after

    # -- losses ----------------------------------------------------------------

    def regularisation(self, vector: np.ndarray) -> float:
        z = (vector - self.prior) / sv.UNITS
        return float(np.sum(z * z))

    def sample_loss(self, vector: np.ndarray) -> float:
        srgb = self.sample_srgb(self.sample_display(vector), vector)
        terms = loss_terms(srgb_to_lab(srgb), self.ref_lab_sample)
        return (
            W_DELTA_E * terms["delta_e_trimmed"]
            + W_LUMA * terms["luma"]
            + W_CHROMA * terms["chroma"]
            + W_REG * self.regularisation(vector)
        )

    def full_loss(self, vector: np.ndarray) -> tuple[float, dict[str, float]]:
        srgb, before, after = self.full(vector)
        lab = srgb_to_lab(srgb)
        mask = self.pair.mask
        terms = loss_terms(
            lab[mask][:: max(1, mask.sum() // 60_000)],
            self.ref_lab[mask][:: max(1, mask.sum() // 60_000)],
        )
        terms["contrast"] = abs(gradient_spread(lab[..., 0], mask) - self.ref_spread)
        loss = (
            W_DELTA_E * terms["delta_e_trimmed"]
            + W_LUMA * terms["luma"]
            + W_CHROMA * terms["chroma"]
            + W_CONTRAST * terms["contrast"]
            + W_REG * self.regularisation(vector)
        )
        terms["_before"], terms["_after"] = before, after  # type: ignore[assignment]
        return loss, terms

    def strict_delta_e(self, vector: np.ndarray) -> float:
        """Mean ΔE of a full render against the unblurred reference."""
        params = self.params(vector)
        srgb = resume(self.pre_sharp, self.decoded, params, after="noise")
        mask = self.pair.mask
        return float(
            delta_e_2000(srgb_to_lab(srgb[mask]), srgb_to_lab(self.pair.reference[mask])).mean()
        )

    def freeze_clarity(self, vector: np.ndarray, before: np.ndarray, after: np.ndarray) -> None:
        """Freeze clarity's gain: the local-contrast stage's output over what
        the local tone alone would have produced from ``before``."""
        v = vector.copy()
        v[sv.index("clarity")] = 0.0
        toned = resume(before, self.decoded, self.params(v), after="color", until="local_contrast")
        ratio = luminance(after) / np.maximum(luminance(toned), 1e-4)
        self.clarity_gain = ratio.reshape(-1)[self.sample_index][:, None, None].astype(np.float32)

    # -- search helpers --------------------------------------------------------

    def search(
        self,
        vector: np.ndarray,
        names: tuple[str, ...],
        budget: int,
        sigma: float,
        seed: int,
        loss: Callable[[np.ndarray], float] | None = None,
    ) -> tuple[np.ndarray, float]:
        indices = np.array([sv.index(n) for n in names])
        low, span = sv.BOUNDS[indices, 0], self.span[indices]
        objective = loss or self.sample_loss

        def f(u: np.ndarray) -> float:
            candidate = vector.copy()
            candidate[indices] = low + u * span
            return objective(candidate)

        start = (vector[indices] - low) / span
        result = cmaes.minimise(f, start, sigma=sigma, budget=budget, seed=seed)
        best = vector.copy()
        best[indices] = low + result.x * span
        return best, result.value

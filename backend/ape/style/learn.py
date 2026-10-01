# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parameter inversion: which style vector turns this RAW into that edit (section 8.2).

The input is one prepared pair (``style/sample.py``): the RAW decoded at the
comparison size, the user's reference registered onto its frame, and a mask of
where the reference exists. The output is the style vector whose rendering is
closest to the reference, and how close that is.

**The loss** is the composite of section 8.2.3: CIEDE2000 (weight 1.0), a
Wasserstein-1 distance between the L* distributions (0.5) and between the a*
and b* distributions (0.3), the difference in local contrast measured as the
spread of the L* gradient (0.2), and an L2 pull towards the neutral vector
(0.05) that decides between two solutions the eye cannot tell apart. The ΔE is
a trimmed mean -- the worst 3% of pixels are dropped -- because the user's
exports carry a watermark and a lens profile slightly different from ours, and
neither is a colour the pipeline should learn.

**The search** spends at most :data:`BUDGET` renders at 512 px, in stages that
each attack the part of the vector they are good at:

1. exposure and white balance by direct solution: a bisection on the median
   lightness, two Newton steps on the average colour of the near-neutral
   pixels. A dozen renders put the start where a blind search would spend a
   hundred;
2. CMA-ES over the global tone and colour controls, pivot excluded (it is
   degenerate with exposure, and a learned vector full of arbitrary splits
   between the two would average badly across samples);
3. the eight HSL bands by fixed point: the band-weighted ratio of the
   reference's saturation and value to ours, and the circular difference of
   hue, applied with damping and kept only if the loss improves;
4. split toning by a short CMA-ES, kept only if it earns its regularisation;
5. clarity and its radius on full renders, the one spatial control;
6. a narrow CMA-ES polish of the global controls with clarity frozen.

The per-pixel stages -- everything from white balance to colour -- are
evaluated on a fixed sample of the masked pixels (``resume`` from the
denoise's output), which is what makes a render cost 3 ms instead of 60.
Clarity is spatial, so stages 1-4 run without it and stage 6 multiplies the
sample by the per-pixel gain clarity applied in the last full render. Every
final number is measured on a full render.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..pipeline.ops.color import _band_weights  # noqa: PLC2701 - same pipeline, one hue model
from ..pipeline.params import HSL_BANDS
from . import vector as sv
from .colordiff import srgb_to_lab
from .problem import PairImages, Problem

__all__ = ["BUDGET", "InversionResult", "PairImages", "invert"]

#: Section 8.2.4: at most 400 evaluations per pair at 512 px.
BUDGET = 400


#: The controls stage 2 searches. Everything the vector has except the pivot,
#: the HSL bands, split toning and clarity, which have stages of their own.
CORE: tuple[str, ...] = (
    "wb_mired_shift",
    "wb_tint_shift",
    "exposure_offset",
    "highlight_strength",
    "black_point_ev",
    "white_point_ev",
    "contrast",
    "toe",
    "shoulder",
    "shadows",
    "highlights",
    "whites",
    "blacks",
    "vibrance",
    "saturation",
    "local_shadows",
    "local_highlights",
)
SPLIT: tuple[str, ...] = (
    "split_shadow_x",
    "split_shadow_y",
    "split_highlight_x",
    "split_highlight_y",
    "split_balance",
)


@dataclass(slots=True)
class InversionResult:
    vector: np.ndarray
    loss: float
    #: Plain mean CIEDE2000 over the mask, on a full render: the honest number.
    delta_e: float
    delta_e_trimmed: float
    #: The same mean without the comparison blur: pixel-scale detail included.
    delta_e_strict: float
    evaluations: int
    seconds: float
    stages: dict[str, float] = field(default_factory=dict)


def _initial_exposure(problem: Problem, vector: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Bisection on the median L*: the one number a blind search wastes most on."""
    target = float(np.median(problem.ref_lab_sample[:, 0]))
    i = sv.index("exposure_offset")
    low, high = sv.BOUNDS[i]
    v = vector.copy()
    for _ in range(iterations):
        v[i] = 0.5 * (low + high)
        srgb = problem.sample_srgb(problem.sample_display(v), v)
        if float(np.median(srgb_to_lab(srgb)[:, 0])) < target:
            low = v[i]
        else:
            high = v[i]
    v[i] = 0.5 * (low + high)
    return v


def _initial_white_balance(problem: Problem, vector: np.ndarray, steps: int = 2) -> np.ndarray:
    """Newton steps on the mean a*, b* of the pixels the reference shows as near grey."""
    ref = problem.ref_lab_sample
    neutral = (np.hypot(ref[:, 1], ref[:, 2]) < 12.0) & (ref[:, 0] > 20) & (ref[:, 0] < 92)
    if neutral.sum() < 200:
        return vector
    target = ref[neutral, 1:].mean(axis=0)
    names = ("wb_mired_shift", "wb_tint_shift")
    idx = [sv.index(n) for n in names]
    deltas = (6.0, 3.0)

    def mean_ab(v: np.ndarray) -> np.ndarray:
        srgb = problem.sample_srgb(problem.sample_display(v), v)
        return srgb_to_lab(srgb)[neutral, 1:].mean(axis=0)

    v = vector.copy()
    for _ in range(steps):
        current = mean_ab(v)
        jacobian = np.zeros((2, 2))
        for column, (i, d) in enumerate(zip(idx, deltas, strict=True)):
            probe = v.copy()
            probe[i] += d
            jacobian[:, column] = (mean_ab(probe) - current) / d
        try:
            step = np.linalg.solve(jacobian, target - current)
        except np.linalg.LinAlgError:
            break
        for i, s in zip(idx, step, strict=True):
            v[i] = float(np.clip(v[i] + s, *sv.BOUNDS[i]))
    return v


def _hsv(display: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(display.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV)[:, 0, :]


def _fit_hsl(problem: Problem, vector: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Fixed-point estimate of the eight bands, each round kept only if it helps."""
    best, best_loss = vector, problem.sample_loss(vector)
    ref = _hsv(problem.ref_display_sample)
    for _ in range(rounds):
        ours = _hsv(problem.sample_display(best))
        weights = _band_weights(ours[:, 0])  # (N, 8)
        chromatic = (ours[:, 1] > 0.08) & (ref[:, 1] > 0.04)
        lit = (ours[:, 2] > 0.05) & (ours[:, 2] < 0.98)
        candidate = best.copy()
        for band_index, band in enumerate(HSL_BANDS):
            w = weights[:, band_index] * chromatic * lit
            total = float(w.sum())
            if total < 150.0:  # fewer than ~150 pixel-equivalents: no evidence
                continue
            sat_ratio = float((w * (ref[:, 1] / np.maximum(ours[:, 1], 1e-3))).sum() / total)
            val_ratio = float((w * (ref[:, 2] / np.maximum(ours[:, 2], 1e-3))).sum() / total)
            dh = (ref[:, 0] - ours[:, 0] + 180.0) % 360.0 - 180.0
            hue_shift = float((w * dh).sum() / total)
            for channel, update in (
                ("saturation", lambda old, r=sat_ratio: (1 + old) * (1 + 0.7 * (r - 1)) - 1),
                ("luminance", lambda old, r=val_ratio: (1 + old) * (1 + 0.7 * (r - 1)) - 1),
                ("hue", lambda old, h=hue_shift: old + 0.7 * h / 30.0),
            ):
                i = sv.index(f"hsl_{band}_{channel}")
                candidate[i] = float(np.clip(update(candidate[i]), *sv.BOUNDS[i]))
        loss = problem.sample_loss(candidate)
        if loss >= best_loss:
            break
        best, best_loss = candidate, loss
    return best


def _fit_clarity(
    problem: Problem, vector: np.ndarray, budget: int
) -> tuple[np.ndarray, float, dict]:
    """Clarity and radius on full renders: a coarse grid, then the best radius."""
    ci, ri = sv.index("clarity"), sv.index("clarity_radius")
    best = vector.copy()
    best_loss, best_terms = problem.full_loss(best)
    used = 1
    for amount in (-0.4, -0.2, 0.2, 0.4, 0.7):
        if used >= budget:
            break
        candidate = best.copy()
        candidate[ci] = amount
        loss, terms = problem.full_loss(candidate)
        used += 1
        if loss < best_loss:
            best, best_loss, best_terms = candidate, loss, terms
    if abs(best[ci]) > 1e-6:
        for radius in (0.008, 0.045):
            if used >= budget:
                break
            candidate = best.copy()
            candidate[ri] = radius
            loss, terms = problem.full_loss(candidate)
            used += 1
            if loss < best_loss:
                best, best_loss, best_terms = candidate, loss, terms
    return best, best_loss, best_terms


def invert(
    pair: PairImages,
    *,
    budget: int = BUDGET,
    seed: int = 0,
    prior: np.ndarray | None = None,
) -> InversionResult:
    """Find the style vector that best reproduces the reference (section 8.2).

    ``prior`` is what the L2 term pulls towards: neutral by default, the
    profile's typical vector when a profile is retrained on its own samples.
    """
    started = time.perf_counter()
    problem = Problem(pair, prior)
    stages: dict[str, float] = {}

    vector = sv.neutral_vector()
    vector = _initial_exposure(problem, vector)
    vector = _initial_white_balance(problem, vector)
    problem.freeze_base(vector)
    stages["init"] = problem.sample_loss(vector)

    core_budget = int(0.55 * budget)
    vector, stages["core"] = problem.search(vector, CORE, core_budget, sigma=0.12, seed=seed)
    problem.freeze_base(vector)

    vector = _fit_hsl(problem, vector)
    stages["hsl"] = problem.sample_loss(vector)

    split, split_loss = problem.search(vector, SPLIT, int(0.08 * budget), sigma=0.08, seed=seed + 1)
    if split_loss < stages["hsl"]:
        vector = split
    stages["split"] = min(split_loss, stages["hsl"])

    vector, full_loss, terms = _fit_clarity(problem, vector, budget=8)
    problem.freeze_clarity(vector, terms.pop("_before"), terms.pop("_after"))
    problem.freeze_base(vector)
    stages["clarity"] = full_loss

    remaining = budget - problem.evaluations - 1
    if remaining > 20:
        polished, _ = problem.search(vector, CORE, remaining, sigma=0.04, seed=seed + 2)
        loss, polished_terms = problem.full_loss(polished)
        if loss < full_loss:
            vector, full_loss, terms = polished, loss, polished_terms
        terms.pop("_before", None)
        terms.pop("_after", None)
    stages["final"] = full_loss

    return InversionResult(
        vector=vector,
        loss=full_loss,
        delta_e=terms["delta_e"],
        delta_e_trimmed=terms["delta_e_trimmed"],
        delta_e_strict=problem.strict_delta_e(vector),
        evaluations=problem.evaluations,
        seconds=time.perf_counter() - started,
        stages=stages,
    )

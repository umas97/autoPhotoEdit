# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A small CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016).

Section 8.2 names CMA-ES for the global phase of the style inversion. The
``cma`` package is not in the stack of section 4, and the algorithm needed here
-- a dozen to twenty dimensions, a few hundred evaluations, box constraints --
fits in one page of NumPy, so it is written out rather than added as a
dependency.

The search runs in the unit cube. Candidates outside it are evaluated at their
projection onto the box and ranked with a quadratic penalty on the distance,
which keeps the mean inside without distorting the step-size adaptation.
Seeded: the same pair inverts to the same parameters every time (test 4 is
about renders, but a profile that changes when retrained on the same data would
be just as surprising).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["CMAResult", "minimise"]


@dataclass(slots=True)
class CMAResult:
    x: np.ndarray
    value: float
    evaluations: int


def minimise(
    function: Callable[[np.ndarray], float],
    start: np.ndarray,
    *,
    sigma: float = 0.15,
    budget: int = 200,
    seed: int = 0,
    population: int | None = None,
) -> CMAResult:
    """Minimise ``function`` over ``[0, 1]^n`` starting from ``start``.

    Args:
        function: the objective, called on points inside the unit cube.
        start: initial mean, inside the cube.
        sigma: initial step size, in cube units.
        budget: maximum number of evaluations, the start included.
        seed: for the sampling.
        population: offspring per generation; the default ``4 + 3 ln n``.

    Returns:
        The best point seen, its value, and how many evaluations were spent.
    """
    rng = np.random.default_rng(seed)
    n = len(start)
    lam = population or 4 + int(3 * np.log(max(n, 1)))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / float(np.sum(weights**2))

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean = np.clip(np.asarray(start, dtype=np.float64), 0.0, 1.0)
    pc = np.zeros(n)
    ps = np.zeros(n)
    basis = np.eye(n)
    scales = np.ones(n)
    covariance = np.eye(n)

    best_value = float(function(mean))
    best_x = mean.copy()
    used = 1
    generation = 0
    while used + lam <= budget:
        z = rng.standard_normal((lam, n))
        y = z @ (basis * scales).T
        candidates = mean + sigma * y
        inside = np.clip(candidates, 0.0, 1.0)
        values = np.array([float(function(x)) for x in inside])
        used += lam
        ranked = values + np.sum((candidates - inside) ** 2, axis=1)
        order = np.argsort(ranked)
        if values[order[0]] < best_value:
            best_value = float(values[order[0]])
            best_x = inside[order[0]].copy()

        selected = y[order[:mu]]
        step = weights @ selected
        mean = np.clip(mean + sigma * step, 0.0, 1.0)

        inv_sqrt = basis @ np.diag(1.0 / scales) @ basis.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt @ step)
        norm_ps = float(np.linalg.norm(ps))
        hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chi_n < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * step
        covariance = (
            (1 - c1 - cmu) * covariance
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * covariance)
            + cmu * (selected.T * weights) @ selected
        )
        sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1)))
        covariance = np.triu(covariance) + np.triu(covariance, 1).T
        eigenvalues, basis = np.linalg.eigh(covariance)
        scales = np.sqrt(np.maximum(eigenvalues, 1e-20))
        generation += 1
        if sigma * float(scales.max()) < 1e-4:
            break  # converged: further samples would all be the same point
    return CMAResult(x=best_x, value=best_value, evaluations=used)

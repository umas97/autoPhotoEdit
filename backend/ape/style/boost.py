# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The "light gradient boosting" of section 8.3.

Multi-output: every tree predicts the whole style vector, and a split is chosen
on the error summed over all outputs (the caller passes them already in
perceptual units, so white balance in mired does not outvote saturation). One
boosting instead of fifty is what makes a leave-one-out of it affordable at
training time: about 10 s for 65 pairs, against hours for one model per
parameter.

Depth-2 trees, :data:`ROUNDS` rounds, shrinkage :data:`RATE`. Deliberately
small: at 60-100 samples anything deeper memorises them, and the
leave-one-out of ``style/model.py`` would say so and never choose it.

A fitted model serialises to plain float arrays (``to_arrays``): a profile file
never needs pickle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Boosting", "fit"]

ROUNDS = 60
RATE = 0.1
_MIN_LEAF = 3
_DEPTH = 2
#: Nodes of a full binary tree of depth 2: root, two children, four leaves.
_NODES = 7


@dataclass(slots=True)
class Boosting:
    #: ``(outputs,)`` starting value.
    initial: np.ndarray
    #: ``(rounds, 3)`` split feature of the three internal nodes, -1 for "no
    #: split" (the node is a leaf, and both its children carry its value).
    features: np.ndarray
    #: ``(rounds, 3)`` thresholds.
    thresholds: np.ndarray
    #: ``(rounds, 4, outputs)`` leaf values, already multiplied by the rate.
    leaves: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.tile(self.initial, (len(x), 1))
        for r in range(len(self.features)):
            node = np.zeros(len(x), dtype=np.int64)  # 0 = root
            for level in range(_DEPTH):
                feature = self.features[r][node]
                threshold = self.thresholds[r][node]
                go_right = np.where(
                    feature >= 0, x[np.arange(len(x)), np.maximum(feature, 0)] > threshold, False
                )
                node = 2 * node + 1 + go_right.astype(np.int64)
                if level == _DEPTH - 1:
                    out += self.leaves[r][node - 3]
        return out

    def negate_output(self, output: int) -> None:
        """Predict ``-y`` instead of ``y`` for one output, in place."""
        self.initial[output] = 0.0 - self.initial[output]
        self.leaves[:, :, output] = 0.0 - self.leaves[:, :, output]

    def negate_feature(self, feature: int) -> None:
        """Expect ``-x`` instead of ``x`` for one input, in place.

        ``x > t`` becomes ``-x < -t``: the threshold changes sign and the node's
        two children swap. The same predictions for every input but one exactly
        on a threshold, which is a midpoint between two training values.
        """
        for r in range(len(self.features)):
            if self.features[r, 0] == feature:
                self.thresholds[r, 0] = 0.0 - self.thresholds[r, 0]
                self.features[r, [1, 2]] = self.features[r, [2, 1]]
                self.thresholds[r, [1, 2]] = self.thresholds[r, [2, 1]]
                self.leaves[r] = self.leaves[r, [2, 3, 0, 1]]
            for node, pair in ((1, [0, 1]), (2, [2, 3])):
                if self.features[r, node] == feature:
                    self.thresholds[r, node] = 0.0 - self.thresholds[r, node]
                    self.leaves[r, pair] = self.leaves[r, pair[::-1]]

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "boost_initial": self.initial,
            "boost_features": self.features,
            "boost_thresholds": self.thresholds,
            "boost_leaves": self.leaves,
        }

    @classmethod
    def from_arrays(cls, data) -> Boosting:
        # Copies: arrays straight out of an ``np.load`` are read-only.
        return cls(
            initial=np.array(data["boost_initial"]),
            features=np.array(data["boost_features"]),
            thresholds=np.array(data["boost_thresholds"]),
            leaves=np.array(data["boost_leaves"]),
        )


def _best_split(x: np.ndarray, residual: np.ndarray) -> tuple[int, float] | None:
    """Feature and threshold minimising the summed squared error of two halves."""
    n = len(residual)
    if n < 2 * _MIN_LEAF:
        return None
    total = residual.sum(axis=0)
    best: tuple[float, int, float] | None = None
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        xs = x[order, j]
        cumulative = np.cumsum(residual[order], axis=0)
        counts = np.arange(1, n + 1, dtype=np.float64)
        # Gain of splitting after position i: |S_L|^2/n_L + |S_R|^2/n_R.
        left = (cumulative**2).sum(axis=1) / counts
        right_sum = total - cumulative
        right_n = n - counts
        with np.errstate(divide="ignore", invalid="ignore"):
            right = (right_sum**2).sum(axis=1) / right_n
        gain = left + right
        valid = np.zeros(n, dtype=bool)
        valid[_MIN_LEAF - 1 : n - _MIN_LEAF] = True
        valid[:-1] &= xs[1:] > xs[:-1]  # never split between equal values
        valid[-1] = False
        if not valid.any():
            continue
        i = int(np.argmax(np.where(valid, gain, -np.inf)))
        if best is None or gain[i] > best[0]:
            best = (float(gain[i]), j, float(0.5 * (xs[i] + xs[i + 1])))
    return None if best is None else (best[1], best[2])


def fit(x: np.ndarray, y: np.ndarray) -> Boosting:
    """Boost depth-2 multi-output trees on ``y`` (``(n, outputs)``)."""
    initial = y.mean(axis=0)
    prediction = np.tile(initial, (len(y), 1))
    features = np.full((ROUNDS, 3), -1, dtype=np.int64)
    thresholds = np.zeros((ROUNDS, 3))
    leaves = np.zeros((ROUNDS, 4, y.shape[1]))
    for r in range(ROUNDS):
        residual = y - prediction
        members = [np.arange(len(y))]
        for node in range(3):
            rows = members[node]
            split = _best_split(x[rows], residual[rows]) if len(rows) else None
            if split is None:
                left, right = rows, rows[:0]
            else:
                features[r, node], thresholds[r, node] = split
                goes_right = x[rows, split[0]] > split[1]
                left, right = rows[~goes_right], rows[goes_right]
            members += [left, right]
        # members[3..6] are the four leaves, in the order apply() visits them.
        for leaf in range(4):
            rows = members[3 + leaf]
            if len(rows):
                value = RATE * residual[rows].mean(axis=0)
                leaves[r, leaf] = value
                prediction[rows] += value
    return Boosting(initial, features, thresholds, leaves)

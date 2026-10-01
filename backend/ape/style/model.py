# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene -> style vector: the predictive half of a profile (section 8.3).

Two regressions trained on the same samples, and a choice between them made
**per parameter** by leave-one-out on the samples themselves:

* **k-NN** (k = 5, Gaussian kernel on the distance), the one section 8.3
  prefers: with few samples it generalises predictably, and it is what lets
  the interface say "edited like your samples #12 and #31". Its neighbours are
  reported for every prediction, whichever regression the numbers came from;
* **ridge regression** on the standardised scene features plus the exposure
  anchor, the fallback of section 8.3. It wins where the dependence on the
  scene is smooth and monotonic and there are enough samples to see it: on
  the user's 65 pairs, leave-one-out, it predicts the exposure with a 0.32 EV
  RMS error and 0.73 worst, against 0.42 and 1.50 for the k-NN;
* **a line in the exposure anchor** alone -- the same ridge with one input.
  With 30 samples the full ridge is starved (worst exposure miss 3.2 EV over
  200 random 30-pair profiles) and this is the robust choice (1.2 EV).

With more than :data:`BOOSTING_MIN_PAIRS` pairs a small gradient boosting joins
the choice (depth-2 trees, shrinkage 0.1), as section 8.3 asks. Every choice is
measured, not assumed, and recorded in the model so the interface can show it.

The distance of the k-NN is computed on the engineered features, standardised
with their robust spread and the floors of ``scene.FEATURE_SCALES``, plus --
when both photos have one -- the CLIP embedding reduced by a PCA fitted on the
profile's samples (section 8.3: 32 components at most, never more than the
samples allow). How much the embedding counts is also chosen by
leave-one-out.

Serialisation is plain arrays in an ``.npz`` (``allow_pickle=False``) plus
JSON: a profile arrives from other machines in an ``.apestyle`` (section 20.1),
and loading one must never execute anything.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np

from ..analysis.scene import FEATURE_NAMES, FEATURE_SCALES, SCENE_FEATURES_VERSION
from . import boost
from . import vector as sv

__all__ = [
    "BOOSTING_MIN_PAIRS",
    "MODEL_VERSION",
    "Neighbour",
    "Prediction",
    "StyleModel",
    "TrainingSample",
]

#: 2: the tint changed sign (``raw/whitepoint.py``); version 1 loads flipped.
MODEL_VERSION = 2

K_NEIGHBOURS = 5
PCA_COMPONENTS = 32
BOOSTING_MIN_PAIRS = 60

#: Candidates the leave-one-out chooses among.
_EMBEDDING_WEIGHTS = (0.0, 0.5, 1.0)
_RIDGE_LAMBDAS = (5.0, 20.0, 60.0)
#: One standardised input: a light touch, just enough to keep a profile whose
#: samples all share one anchor from dividing by nothing.
_ANCHOR_LAMBDA = 1.0

_FEATURE_FLOORS = np.array([FEATURE_SCALES[n] for n in FEATURE_NAMES], dtype=np.float64)

KNN, RIDGE, BOOST, ANCHOR = 0, 1, 2, 3
METHOD_NAMES = ("knn", "ridge", "boost", "anchor")


@dataclass(slots=True)
class TrainingSample:
    sample_id: int
    vector: np.ndarray
    features: np.ndarray
    embedding: np.ndarray | None
    exposure_anchor_ev: float


@dataclass(frozen=True, slots=True)
class Neighbour:
    sample_id: int
    weight: float
    distance: float


@dataclass(slots=True)
class Prediction:
    vector: np.ndarray
    neighbours: list[Neighbour]
    #: Distance to the nearest sample, in the model's standardised units.
    nearest: float
    #: Spread of the neighbours' vectors, in perceptual units (section 9.1).
    dispersion: float
    used_embedding: bool
    #: The neighbours' weighted mean alone, whatever method won each
    #: parameter: "edited like your nearest samples", one of the variants the
    #: review of section 9.2 offers when the methods disagree.
    knn_vector: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Small numeric pieces
# --------------------------------------------------------------------------- #


def _knn_weights(distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` nearest and their Gaussian-kernel weights.

    The kernel's width is the median distance of the k neighbours, so it adapts
    to how densely the samples cover this part of the scene space.
    """
    k = min(k, len(distances))
    order = np.argsort(distances, kind="stable")[:k]
    near = distances[order]
    width = max(float(np.median(near)), 1e-6)
    weights = np.exp(-0.5 * (near / width) ** 2)
    return order, weights / weights.sum()


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> dict[str, np.ndarray]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (x - mean) / scale
    y_mean = y.mean(axis=0)
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y_mean))
    return {"mean": mean, "scale": scale, "w": w, "y_mean": y_mean}


def _ridge_apply(fit: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return ((x - fit["mean"]) / fit["scale"]) @ fit["w"] + fit["y_mean"]


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class StyleModel:
    sample_ids: np.ndarray
    vectors: np.ndarray  # (n, len(sv.NAMES))
    features_z: np.ndarray  # (n, n_features), standardised
    feature_centre: np.ndarray
    feature_scale: np.ndarray
    anchors: np.ndarray
    embeddings_z: np.ndarray | None  # (n, k), PCA scores scaled
    pca_mean: np.ndarray | None
    pca_components: np.ndarray | None
    pca_scale: np.ndarray | None
    embedding_weight: float
    ridge: dict[str, np.ndarray]
    ridge_lambda: float
    methods: np.ndarray  # (len(sv.NAMES),) KNN / RIDGE / BOOST / ANCHOR
    #: Ridge on the anchor alone: the ANCHOR method.
    anchor_fit: dict[str, np.ndarray] = field(default_factory=dict)
    #: Fitted on the vectors in perceptual units (``sv.UNITS``).
    boost: boost.Boosting | None = None
    #: Leave-one-out RMS error per parameter, in perceptual units, per method.
    validation: dict[str, list[float]] = field(default_factory=dict)

    # -- inputs ----------------------------------------------------------------

    def standardise(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=np.float64) - self.feature_centre) / self.feature_scale
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def project(self, embedding: np.ndarray | None) -> np.ndarray | None:
        if embedding is None or self.pca_components is None:
            return None
        return (
            (np.asarray(embedding, dtype=np.float64) - self.pca_mean) @ self.pca_components.T
        ) / self.pca_scale

    def _ridge_input(self, features_z: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        return np.hstack([features_z, np.asarray(anchors, dtype=np.float64).reshape(-1, 1)])

    def _distances(self, z: np.ndarray, e: np.ndarray | None) -> tuple[np.ndarray, bool]:
        d2 = ((self.features_z - z) ** 2).sum(axis=1) / self.features_z.shape[1]
        use = e is not None and self.embeddings_z is not None and self.embedding_weight > 0
        if use:
            d2 = (
                d2
                + self.embedding_weight**2
                * ((self.embeddings_z - e) ** 2).sum(axis=1)
                / (self.embeddings_z.shape[1])
            )
        return np.sqrt(d2), use

    # -- prediction ------------------------------------------------------------

    def _dispersion(self, order: np.ndarray, weights: np.ndarray, knn: np.ndarray) -> float:
        """RMS spread of the neighbours' vectors around their mean, perceptual units."""
        spread = np.sqrt(weights @ (((self.vectors[order] - knn) / sv.UNITS) ** 2))
        return float(np.sqrt(np.mean(spread**2)))

    def spacing(self) -> tuple[np.ndarray, np.ndarray]:
        """Each sample's distance to its nearest other sample, and the dispersion
        of its neighbours, both leave-one-out.

        The yardstick of the confidence of section 9.1: "far from the samples"
        and "the neighbours disagree" are relative to how far apart, and how
        different, the samples of *this* profile are among themselves.
        """
        count = len(self.sample_ids)
        nearest = np.full(count, np.nan)
        dispersion = np.full(count, np.nan)
        if count < 2:
            return nearest, dispersion
        for i in range(count):
            own = None if self.embeddings_z is None else self.embeddings_z[i]
            distances, _used = self._distances(self.features_z[i], own)
            distances[i] = np.inf
            order, weights = _knn_weights(distances, min(K_NEIGHBOURS, count - 1))
            nearest[i] = distances[order[0]]
            dispersion[i] = self._dispersion(order, weights, weights @ self.vectors[order])
        return nearest, dispersion

    def predict(
        self, features: np.ndarray, embedding: np.ndarray | None, exposure_anchor_ev: float
    ) -> Prediction:
        z = self.standardise(features)
        distances, used = self._distances(z, self.project(embedding))
        order, weights = _knn_weights(distances, K_NEIGHBOURS)
        knn = weights @ self.vectors[order]
        ridge = _ridge_apply(self.ridge, self._ridge_input(z[None], [exposure_anchor_ev]))[0]
        out = np.where(self.methods == RIDGE, ridge, knn)
        if self.anchor_fit:
            line = _ridge_apply(self.anchor_fit, np.array([[exposure_anchor_ev]]))[0]
            out = np.where(self.methods == ANCHOR, line, out)
        if self.boost is not None and np.any(self.methods == BOOST):
            x = self._ridge_input(z[None], [exposure_anchor_ev])
            boosted = self.boost.apply(x)[0] * sv.UNITS
            out = np.where(self.methods == BOOST, boosted, out)
        return Prediction(
            vector=sv.clip(out),
            neighbours=[
                Neighbour(int(self.sample_ids[i]), float(w), float(distances[i]))
                for i, w in zip(order, weights, strict=True)
            ],
            nearest=float(distances[order[0]]),
            dispersion=self._dispersion(order, weights, knn),
            used_embedding=used,
            knn_vector=sv.clip(knn),
        )

    # -- persistence -----------------------------------------------------------

    def to_bytes(self) -> bytes:
        arrays = {
            "sample_ids": self.sample_ids,
            "vectors": self.vectors,
            "features_z": self.features_z,
            "feature_centre": self.feature_centre,
            "feature_scale": self.feature_scale,
            "anchors": self.anchors,
            "methods": self.methods,
            "ridge_mean": self.ridge["mean"],
            "ridge_scale": self.ridge["scale"],
            "ridge_w": self.ridge["w"],
            "ridge_y_mean": self.ridge["y_mean"],
        }
        if self.anchor_fit:
            arrays.update({f"anchor_{k}": v for k, v in self.anchor_fit.items()})
        if self.embeddings_z is not None:
            arrays.update(
                embeddings_z=self.embeddings_z,
                pca_mean=self.pca_mean,
                pca_components=self.pca_components,
                pca_scale=self.pca_scale,
            )
        if self.boost is not None:
            arrays.update(self.boost.to_arrays())
        meta = {
            "model_version": MODEL_VERSION,
            "vector_names": list(sv.NAMES),
            "feature_names": list(FEATURE_NAMES),
            "scene_features_version": SCENE_FEATURES_VERSION,
            "embedding_weight": self.embedding_weight,
            "ridge_lambda": self.ridge_lambda,
            "validation": self.validation,
        }
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> StyleModel:
        """Load a model. Raises ``ValueError`` on a model this build cannot use."""
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            legacy_tint = meta.get("model_version") == 1
            if meta.get("model_version") != MODEL_VERSION and not legacy_tint:
                raise ValueError("modello di stile di una versione diversa: va riaddestrato")
            if meta.get("vector_names") != list(sv.NAMES) or meta.get("feature_names") != list(
                FEATURE_NAMES
            ):
                raise ValueError("modello di stile con parametri diversi: va riaddestrato")
            has_pca = "pca_components" in data
            trees = boost.Boosting.from_arrays(data) if "boost_initial" in data else None
            model = cls(
                sample_ids=data["sample_ids"],
                vectors=data["vectors"],
                features_z=data["features_z"],
                feature_centre=data["feature_centre"],
                feature_scale=data["feature_scale"],
                anchors=data["anchors"],
                embeddings_z=data["embeddings_z"] if has_pca else None,
                pca_mean=data["pca_mean"] if has_pca else None,
                pca_components=data["pca_components"] if has_pca else None,
                pca_scale=data["pca_scale"] if has_pca else None,
                embedding_weight=float(meta["embedding_weight"]),
                ridge={
                    "mean": data["ridge_mean"],
                    "scale": data["ridge_scale"],
                    "w": data["ridge_w"],
                    "y_mean": data["ridge_y_mean"],
                },
                ridge_lambda=float(meta["ridge_lambda"]),
                anchor_fit={k: data[f"anchor_{k}"] for k in ("mean", "scale", "w", "y_mean")}
                if "anchor_w" in data
                else {},
                methods=data["methods"],
                boost=trees,
                validation=meta.get("validation", {}),
            )
        if legacy_tint:
            model.flip_tint()
        return model

    def flip_tint(self) -> None:
        """Turn a version 1 model into the same model in the new tint sign, in place.

        The tint is both an output (``wb_tint_shift``) and an input (the
        as-shot ``tint`` scene feature). Every regression here is linear in
        both, or a tree over them, so negating the right rows, columns and
        thresholds gives the same predictions, negated tint included, without
        retraining (``tests/test_tint_sign.py``).
        """
        out = sv.index("wb_tint_shift")
        feature = FEATURE_NAMES.index("tint")
        self.vectors = np.array(self.vectors, dtype=np.float64)
        self.vectors[:, out] = 0.0 - self.vectors[:, out]
        self.features_z = np.array(self.features_z, dtype=np.float64)
        self.features_z[:, feature] = 0.0 - self.features_z[:, feature]
        self.feature_centre = np.array(self.feature_centre, dtype=np.float64)
        self.feature_centre[feature] = 0.0 - self.feature_centre[feature]
        # The ridge's inputs are the standardised features and then the anchor,
        # so the feature's index is also its row.
        for fit, has_feature in ((self.ridge, True), (self.anchor_fit, False)):
            if not fit:
                continue
            fit["mean"] = np.array(fit["mean"], dtype=np.float64)
            fit["w"] = np.array(fit["w"], dtype=np.float64)
            fit["y_mean"] = np.array(fit["y_mean"], dtype=np.float64)
            if has_feature:
                fit["mean"][feature] = 0.0 - fit["mean"][feature]
                fit["w"][feature] = 0.0 - fit["w"][feature]
            fit["w"][:, out] = 0.0 - fit["w"][:, out]
            fit["y_mean"][out] = 0.0 - fit["y_mean"][out]
        if self.boost is not None:
            self.boost.negate_output(out)
            self.boost.negate_feature(feature)

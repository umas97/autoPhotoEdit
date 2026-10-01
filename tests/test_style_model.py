# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The predictive half of a profile (section 8.3), coherence (8.4), pairing (8.1).

All on synthetic samples: the point here is the arithmetic -- which regression
wins when, that a model survives serialisation bit for bit, that coherence
moves what it should -- not the look, which ``test_style_flow.py`` covers on
the user's files.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ape.analysis.scene import FEATURE_NAMES
from ape.raw.metadata import ReferenceIdentity
from ape.style import boost, builtin, coherence
from ape.style import vector as sv
from ape.style.model import ANCHOR, KNN, StyleModel, TrainingSample
from ape.style.pairing import pair_files
from ape.style.train import train

_EXPOSURE = sv.index("exposure_offset")
_CONTRAST = sv.index("contrast")
_VIBRANCE = sv.index("vibrance")


def _samples(n: int, seed: int = 0, embeddings: bool = True) -> list[TrainingSample]:
    """Two kinds of scene, a style that depends on which, an exposure that
    depends linearly on the anchor -- the structure of the user's pairs."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        indoor = i % 2 == 0
        features = rng.normal(0, 0.02, len(FEATURE_NAMES)).astype(np.float32)
        features[FEATURE_NAMES.index("cct_mired")] = 310.0 if indoor else 180.0
        features[FEATURE_NAMES.index("lum_median")] = 0.3 if indoor else 0.5
        features[FEATURE_NAMES.index("ev100")] = 6.0 if indoor else 12.0
        anchor = float(rng.uniform(-1.5, 2.5))
        vector = sv.neutral_vector()
        vector[_CONTRAST] = 1.1 if indoor else 1.5
        vector[_VIBRANCE] = 0.1 if indoor else 0.4
        vector[_EXPOSURE] = -0.6 * anchor + 0.3 + rng.normal(0, 0.03)
        embedding = None
        if embeddings:
            embedding = rng.normal(0, 1, 512)
            embedding[0] += 8.0 if indoor else -8.0
            embedding /= np.linalg.norm(embedding)
        out.append(TrainingSample(100 + i, vector, features, embedding, anchor))
    return out


def test_model_learns_scene_dependence_and_the_exposure_line():
    samples = _samples(30)
    model = train(samples)
    # The exposure is a line in the anchor: the anchor regression must win it.
    assert model.methods[_EXPOSURE] == ANCHOR
    indoor, outdoor = samples[0], samples[1]
    for s, contrast in ((indoor, 1.1), (outdoor, 1.5)):
        p = model.predict(s.features, s.embedding, 1.0)
        assert p.vector[_CONTRAST] == pytest.approx(contrast, abs=0.05)
        assert p.vector[_EXPOSURE] == pytest.approx(-0.3, abs=0.1)
        assert len(p.neighbours) == 5
        assert all(n.sample_id >= 100 for n in p.neighbours)
    # A parameter that no scene moves is left to the explainable k-NN.
    assert model.methods[sv.index("hsl_red_hue")] == KNN


def test_model_survives_serialisation_exactly():
    samples = _samples(64, seed=2)  # > 60 pairs: the boosting joins the choice
    model = train(samples)
    assert "boost" in model.validation
    again = StyleModel.from_bytes(model.to_bytes())
    rng = np.random.default_rng(9)
    for s in samples[:10]:
        features = s.features + rng.normal(0, 0.01, s.features.shape).astype(np.float32)
        a = model.predict(features, s.embedding, 0.4)
        b = again.predict(features, s.embedding, 0.4)
        assert np.array_equal(a.vector, b.vector)
        assert [n.sample_id for n in a.neighbours] == [n.sample_id for n in b.neighbours]


def test_model_predicts_without_an_embedding():
    model = train(_samples(12))
    s = _samples(1, seed=5)[0]
    p = model.predict(s.features, None, 0.0)
    assert not p.used_embedding
    assert np.all(np.isfinite(p.vector))


def test_boosting_round_trips_as_arrays():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(70, 5))
    y = np.stack([np.where(x[:, 0] > 0, 1.0, -1.0), x[:, 1] ** 2], axis=1)
    fitted = boost.fit(x, y)
    again = boost.Boosting.from_arrays(fitted.to_arrays())
    assert np.array_equal(fitted.apply(x), again.apply(x))
    # It has learned the step: the sign of the first feature.
    assert np.corrcoef(fitted.apply(x)[:, 0], y[:, 0])[0, 1] > 0.9


def test_builtin_rules_are_valid_and_distinct():
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[FEATURE_NAMES.index("lum_p99")] = 0.9
    vectors = {
        name: builtin.predict(
            builtin.RULES[name],
            features,
            exposure_anchor_ev=1.0,
            auto_wb_mired=10.0,
            auto_wb_tint=-2.0,
        )
        for name in builtin.BUILTIN_NAMES
    }
    context = sv.StyleContext(5000.0, 0.0, 1.0)
    for vector in vectors.values():
        sv.to_params(vector, context)  # validates every range
    assert len({v.tobytes() for v in vectors.values()}) == 4
    # Ritratto: clarity never positive (section 22), skin bands held back.
    ritratto = vectors["Ritratto"]
    assert ritratto[sv.index("clarity")] <= 0
    assert ritratto[sv.index("hsl_orange_saturation")] < 0
    # Paesaggio recovers highlights the most.
    assert vectors["Paesaggio"][sv.index("local_highlights")] == min(
        v[sv.index("local_highlights")] for v in vectors.values()
    )


def _item(photo_id, mired_shift, as_shot_k, *, cluster=1, second=0, exposure=0.0):
    vector = sv.neutral_vector()
    vector[sv.index("wb_mired_shift")] = mired_shift
    vector[_EXPOSURE] = exposure
    return coherence.CoherenceItem(
        photo_id,
        vector,
        as_shot_k,
        0.0,
        cluster,
        datetime(2026, 3, 14, 18, 0, 0) + timedelta(seconds=second),
        50.0,
    )


def test_coherence_off_changes_nothing():
    items = [_item(1, 10.0, 5000.0), _item(2, -10.0, 5200.0, second=5)]
    out = coherence.regularise(items, 0.0)
    assert np.array_equal(out[1], items[0].vector) and np.array_equal(out[2], items[1].vector)


def test_sequence_evens_out_the_absolute_temperature():
    """Two frames five seconds apart, the camera's AWB wandered by 300 K, the
    prediction shifted both the same: coherence must bring the *results*
    together, not the shifts."""
    items = [_item(1, 5.0, 5000.0), _item(2, 5.0, 5300.0, second=5)]
    out = coherence.regularise(items, coherence.DEFAULT_LAMBDA)

    def target_mired(item, vector):
        return 1e6 / item.as_shot_temperature_k - vector[sv.index("wb_mired_shift")]

    before = abs(target_mired(items[0], items[0].vector) - target_mired(items[1], items[1].vector))
    after = abs(target_mired(items[0], out[1]) - target_mired(items[1], out[2]))
    # The scene pull (0.35) then the sequence pull (0.6) of section 8.4.
    expected = (1 - coherence.DEFAULT_LAMBDA) * (1 - coherence.SEQUENCE_LAMBDA) * before
    assert after == pytest.approx(expected, rel=0.02)


def test_coherence_pulls_inside_a_scene_and_not_across():
    items = [
        _item(1, 0.0, 5000.0, exposure=0.0, second=0),
        _item(2, 0.0, 5000.0, exposure=1.0, second=600),
        _item(3, 0.0, 5000.0, exposure=0.5, second=1200),
        _item(4, 0.0, 5000.0, exposure=2.0, cluster=2, second=1800),
    ]
    out = coherence.regularise(items, coherence.DEFAULT_LAMBDA)
    assert out[2][_EXPOSURE] < 1.0 and out[1][_EXPOSURE] > 0.0
    assert out[4][_EXPOSURE] == pytest.approx(2.0)  # alone in its scene


def test_pairing_by_name_then_xmp_then_time(tmp_path: Path):
    raws = [tmp_path / f"DSC0{i}.ARW" for i in range(1, 6)]
    refs = [
        tmp_path / "DSC01.jpg",  # base name
        tmp_path / "evento_002.jpg",  # XMP original name
        tmp_path / "evento_003.jpg",  # shooting time
        tmp_path / "evento_004.jpg",  # a time two RAWs share: not paired
        tmp_path / "copia_DSC01.jpg",  # a second edit of DSC01: a duplicate
    ]
    t = datetime(2026, 3, 14, 18, 0, 0)
    identities = {
        refs[1]: ReferenceIdentity("DSC02.ARW", None, None, "ILCE-7M3"),
        refs[2]: ReferenceIdentity(None, t, None, "ILCE-7M3"),
        refs[3]: ReferenceIdentity(None, t + timedelta(seconds=9), None, "ILCE-7M3"),
        refs[4]: ReferenceIdentity("DSC01.ARW", None, None, "ILCE-7M3"),
    }
    raw_times = {
        raws[2]: (t, "", "ILCE-7M3"),
        raws[3]: (t + timedelta(seconds=9), "", "ILCE-7M3"),
        raws[4]: (t + timedelta(seconds=9), "", "ILCE-7M3"),
    }
    result = pair_files(raws, refs, identities=identities, raw_times=raw_times)
    methods = {p.raw.name: p.method for p in result.pairs}
    assert methods == {"DSC01.ARW": "name", "DSC02.ARW": "xmp", "DSC03.ARW": "time"}
    assert result.unpaired_references == [refs[3]]
    assert [d.reference for d in result.duplicates] == [refs[4]]


def test_spacing_is_leave_one_out_and_flags_what_is_far():
    """The yardstick of the confidence (section 9.1) from the samples themselves."""
    from ape.review.confidence import ConfidenceInputs, score
    from ape.style.profile import LoadedProfile

    samples = _samples(30)
    model = train(samples)
    nearest, dispersion = model.spacing()
    # Without leaving itself out every sample would be at distance 0.
    assert np.all(nearest > 0) and np.all(np.isfinite(dispersion))
    profile = LoadedProfile(1, "prova", False, None, None, model)
    calibration = profile.calibration
    assert calibration.nearest_p90 == pytest.approx(np.percentile(nearest, 90))
    assert calibration.anchor_min >= -1.5 and calibration.anchor_max <= 2.5

    def confidence(sample: TrainingSample) -> float:
        p = model.predict(sample.features, sample.embedding, sample.exposure_anchor_ev)
        return score(
            ConfidenceInputs(
                nearest=p.nearest, nearest_p90=calibration.nearest_p90,
                dispersion=p.dispersion, dispersion_p90=calibration.dispersion_p90,
                exposure_anchor_ev=sample.exposure_anchor_ev,
                anchor_range=(calibration.anchor_min, calibration.anchor_max),
            )
        ).value

    # A new photo like the samples is confident; one from another world is not.
    like = _samples(2, seed=7)[0]
    assert confidence(like) > 0.9
    alien = _samples(1, seed=8)[0]
    alien.features = alien.features.copy()
    alien.features[FEATURE_NAMES.index("ev100")] = -4.0
    alien.features[FEATURE_NAMES.index("cct_mired")] = 500.0
    alien.embedding = -alien.embedding
    alien.exposure_anchor_ev = 4.5
    assert confidence(alien) < 0.55
    # The prediction also reports the neighbours' own vector, for the variants.
    assert model.predict(like.features, like.embedding, 0.0).knn_vector is not None

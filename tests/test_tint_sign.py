# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The tint's change of sign of parameters version 2.

Two claims. The slider now goes Lightroom's way: raising it makes the picture
magenta. And the migrations change nothing but the sign: a photo edited before
renders identically to the bit, a style model trained before predicts the same
edits, and a catalogue or an ``.apestyle`` written before comes out consistent.

"Before" is reproduced by putting the old constant back into
``raw/whitepoint.py``, the one place the convention lives.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from ape.analysis.scene import _MAGIC, FEATURE_NAMES, decode_features
from ape.pipeline.ops import white_balance
from ape.pipeline.params import PARAMS_VERSION, EditParams, migrate
from ape.raw import whitepoint
from ape.style import boost
from ape.style import model as style_model
from ape.style import vector as sv
from ape.style.model import TrainingSample
from ape.style.train import train

FIXTURES = Path(__file__).parent / "fixtures"
_TINT = sv.index("wb_tint_shift")
_FEATURE = FEATURE_NAMES.index("tint")


@pytest.fixture
def old_sign(monkeypatch):
    """The convention of parameters version 1, for the length of a test."""

    def use(old: bool) -> None:
        monkeypatch.setattr(whitepoint, "TINT_PER_DUV", -3000.0 if old else 3000.0)
        white_balance.adaptation_matrix.cache_clear()

    yield use
    white_balance.adaptation_matrix.cache_clear()


def _positive_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) > 0


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tint", [15.0, 50.0])
def test_raising_the_tint_turns_a_grey_magenta(tint):
    grey = np.ones(3)
    up = white_balance.adaptation_matrix(5500.0, tint, 5500.0, 0.0) @ grey
    down = white_balance.adaptation_matrix(5500.0, -tint, 5500.0, 0.0) @ grey
    assert up[1] < min(up[0], up[2]), "più tinta = più magenta, come in Lightroom"
    assert down[1] > max(down[0], down[2]), "meno tinta = più verde"


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_version_one_parameters_are_negated():
    custom = {"params_version": 1, "white_balance": {"mode": "custom", "tint": 7.5}}
    assert EditParams.from_dict(custom).white_balance.tint == -7.5
    assert EditParams.from_dict({"params_version": 1, "exposure": {"ev": 0.5}}).exposure.ev == 0.5
    zero = migrate({"params_version": 1, "white_balance": {"mode": "as_shot", "tint": 0.0}})
    # A negative zero would change the style signature of an untouched photo.
    assert _positive_zero(zero["white_balance"]["tint"])


@pytest.mark.slow
def test_migrated_parameters_render_identically(old_sign):
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.decode import decode_linear

    source = FIXTURES / "DSC05618.ARW"
    v1 = {
        "params_version": 1,
        "white_balance": {"mode": "custom", "temperature_k": 4700.0, "tint": 9.0},
        "exposure": {"ev": 0.3},
    }
    options = RenderOptions(long_edge=512)

    old_sign(True)
    decoded = decode_linear(source, half_size=True)
    old_as_shot = decoded.camera.as_shot_tint
    before = render(decoded, EditParams.model_validate(v1), options)

    old_sign(False)
    decoded = decode_linear(source, half_size=True)
    assert decoded.camera.as_shot_tint == -old_as_shot
    after = render(decoded, EditParams.from_dict(v1), options)
    assert np.array_equal(before, after)


# --------------------------------------------------------------------------- #
# Scene features and models
# --------------------------------------------------------------------------- #


def test_version_one_scene_features_decode_flipped():
    vector = np.arange(len(FEATURE_NAMES), dtype=np.float32)
    vector[_FEATURE] = -21.5
    decoded = decode_features(_MAGIC + bytes([1]) + vector.astype("<f4").tobytes())
    assert decoded[_FEATURE] == 21.5
    others = np.arange(len(FEATURE_NAMES)) != _FEATURE
    assert np.array_equal(decoded[others], vector[others])


def test_boosting_negation_predicts_the_same():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (80, 5))
    y = np.column_stack([np.sign(x[:, 2]) + 0.3 * x[:, 0], x[:, 2] ** 2 - x[:, 1]])
    trees = boost.fit(x, y)
    assert np.any(trees.features == 2), "il test deve dividere sulla feature negata"
    flipped = boost.Boosting.from_arrays(trees.to_arrays())
    flipped.negate_feature(2)
    flipped.negate_output(0)
    query = rng.normal(0, 1, (200, 5))
    mirrored = query.copy()
    mirrored[:, 2] *= -1
    expected = trees.apply(query)
    expected[:, 0] *= -1
    assert np.allclose(flipped.apply(mirrored), expected, rtol=0, atol=1e-12)


def _samples(n: int, seed: int) -> list[TrainingSample]:
    """Scenes whose tint shift follows the camera's tint, as a user's might."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        features = rng.normal(0, 0.05, len(FEATURE_NAMES)).astype(np.float32)
        features[_FEATURE] = rng.uniform(-25, 25)
        features[FEATURE_NAMES.index("cct_mired")] = rng.uniform(150, 330)
        vector = sv.neutral_vector()
        vector[_TINT] = 0.4 * features[_FEATURE] + rng.normal(0, 1)
        vector[sv.index("contrast")] = 1.2 + 0.01 * features[_FEATURE]
        anchor = float(rng.uniform(-1, 2))
        vector[sv.index("exposure_offset")] = -0.5 * anchor
        embedding = rng.normal(0, 1, 512)
        embedding /= np.linalg.norm(embedding)
        out.append(TrainingSample(i + 1, vector, features, embedding, anchor))
    return out


def test_a_version_one_model_predicts_the_same_edits(monkeypatch):
    samples = _samples(70, seed=5)  # past BOOSTING_MIN_PAIRS: every method exists
    model = train(samples)
    assert model.boost is not None
    assert np.any(model.boost.features[:, 0] == _FEATURE), "alberi divisi sulla tinta"
    # Every method on some parameter, so every piece of the migration is used.
    model.methods = np.arange(len(sv.NAMES)) % 4
    monkeypatch.setattr(style_model, "MODEL_VERSION", 1)
    blob = model.to_bytes()
    monkeypatch.undo()

    migrated = style_model.StyleModel.from_bytes(blob)
    for sample in _samples(12, seed=8):
        mirrored = sample.features.copy()
        mirrored[_FEATURE] = -mirrored[_FEATURE]
        old = model.predict(sample.features, sample.embedding, sample.exposure_anchor_ev)
        new = migrated.predict(mirrored, sample.embedding, sample.exposure_anchor_ev)
        expected = old.vector.copy()
        expected[_TINT] = -expected[_TINT]
        assert np.allclose(new.vector, expected, rtol=0, atol=1e-9)
        assert [n.sample_id for n in new.neighbours] == [n.sample_id for n in old.neighbours]
        assert new.nearest == pytest.approx(old.nearest, abs=1e-12)
    # And it saves as the current version.
    meta_ok = style_model.StyleModel.from_bytes(migrated.to_bytes())
    assert np.array_equal(meta_ok.vectors, migrated.vectors)


# --------------------------------------------------------------------------- #
# The catalogue and the .apestyle
# --------------------------------------------------------------------------- #


def test_a_schema_six_catalogue_is_converted(tmp_path: Path):
    from sqlalchemy.orm import sessionmaker

    from ape.db.models import EditVersion, Photo, Project, StyleProfile, StyleSample
    from ape.db.session import SCHEMA_VERSION, engine_for, init_db, session_scope
    from ape.style.apply import style_signature

    engine = engine_for(tmp_path / "vecchio.db")
    init_db(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    styled = {
        "params_version": 1,
        "white_balance": {"mode": "custom", "temperature_k": 5100.0, "tint": 6.0},
    }
    neutral = {"params_version": 1, "white_balance": {"mode": "as_shot", "tint": 0.0}}
    vector = [float(x) for x in sv.neutral_vector()]
    vector[_TINT] = 4.0
    context = {"as_shot_temperature_k": 5000.0, "as_shot_tint": -12.0, "exposure_anchor_ev": 0.4}
    with session_scope(maker) as session:
        project = Project(name="vecchio", source_dir="/non/esiste")
        session.add(project)
        session.flush()
        photos = []
        for name, params in (("A.ARW", styled), ("B.ARW", neutral)):
            photo = Photo(project_id=project.id, filename=name, path=f"/non/esiste/{name}")
            photo.analysis = {
                "as_shot": {"temperature_k": 5000.0, "tint": -12.0},
                "auto": {"version": 3, "wb_mired_shift": 2.0, "wb_tint_shift": 1.5},
            }
            photo.prediction = {
                "vector": vector,
                "knn_vector": vector,
                "applied_vector": vector,
                "context": context,
                "applied_signature": style_signature(EditParams.model_validate(params)),
            }
            session.add(photo)
            session.flush()
            session.add(
                EditVersion(photo_id=photo.id, params=params, params_version=1, is_current=True)
            )
            photos.append(photo.id)
        profile = StyleProfile(name="Copia", builtin_rules={"base": {"wb_tint_shift": 3.0}})
        session.add(profile)
        session.flush()
        session.add(
            StyleSample(profile_id=profile.id, params=styled, vector=vector, context=context)
        )
    with engine.begin() as connection:
        connection.execute(text("UPDATE setting SET value = '6' WHERE key = 'schema_version'"))

    init_db(engine)
    with session_scope(maker) as session:
        version = session.query(EditVersion).filter_by(photo_id=photos[0]).one()
        assert version.params_version == PARAMS_VERSION
        params = EditParams.from_dict(version.params)
        assert params.white_balance.tint == -6.0
        for photo_id in photos:
            photo = session.get(Photo, photo_id)
            assert photo.analysis["as_shot"]["tint"] == 12.0
            assert photo.analysis["auto"]["wb_tint_shift"] == -1.5
            for key in ("vector", "knn_vector", "applied_vector"):
                assert photo.prediction[key][_TINT] == -4.0
            assert photo.prediction["context"]["as_shot_tint"] == 12.0
            # Still the program's own style, so the profile keeps following it.
            current = session.query(EditVersion).filter_by(photo_id=photo_id).one()
            assert photo.prediction["applied_signature"] == style_signature(
                EditParams.from_dict(current.params)
            )
        untouched = session.query(EditVersion).filter_by(photo_id=photos[1]).one()
        assert _positive_zero(untouched.params["white_balance"]["tint"])
        sample = session.query(StyleSample).one()
        assert sample.vector[_TINT] == -4.0 and sample.context["as_shot_tint"] == 12.0
        assert sample.params["white_balance"]["tint"] == -6.0
        assert session.query(StyleProfile).one().builtin_rules["base"]["wb_tint_shift"] == -3.0
        assert int(session.execute(
            text("SELECT value FROM setting WHERE key = 'schema_version'")
        ).scalar_one()) == SCHEMA_VERSION


def test_a_version_one_apestyle_imports_converted(tmp_path: Path):
    from sqlalchemy.orm import sessionmaker

    from ape.db.models import StyleSample
    from ape.db.session import engine_for, init_db, session_scope
    from ape.style import portable

    vector = [float(x) for x in sv.neutral_vector()]
    vector[_TINT] = 5.0
    features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, _FEATURE] = -8.0
    document = {
        "name": "Vecchio",
        "samples": [
            {
                "id": 1,
                "params": {"params_version": 1, "white_balance": {"mode": "custom", "tint": 2.0}},
                "vector": vector,
                "context": {"as_shot_temperature_k": 5200.0, "as_shot_tint": -9.0},
                "has_features": True,
            }
        ],
    }
    arrays = io.BytesIO()
    np.savez_compressed(arrays, features=features, embeddings=np.zeros((1, 512), np.float32))
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("format.json", json.dumps({"format": "apestyle", "format_version": 1}))
        archive.writestr("profile.json", json.dumps(document))
        archive.writestr("samples.npz", arrays.getvalue())

    engine = engine_for(tmp_path / "a.db")
    init_db(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with session_scope(maker) as session:
        portable.import_profile(session, data.getvalue())
    with session_scope(maker) as session:
        sample = session.query(StyleSample).one()
        assert sample.vector[_TINT] == -5.0
        assert sample.context["as_shot_tint"] == 9.0
        assert EditParams.from_dict(sample.params).white_balance.tint == -2.0
        assert decode_features(sample.scene_features)[_FEATURE] == 8.0

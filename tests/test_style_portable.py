# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 15 of section 13: a profile's round trip through an ``.apestyle``.

Export a trained profile, import it into an *empty* catalogue on which none of
the sample RAWs exist, predict ten photos with both: the parameters must be
identical. Plus the refusals: not a profile, a newer format.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from ape.analysis.embed import encode_embedding
from ape.analysis.scene import encode_features
from ape.db.models import StyleProfile, StyleSample, StyleSampleStatus
from ape.db.session import engine_for, init_db, session_scope
from ape.style import auto, portable
from ape.style import vector as sv
from ape.style.profile import load, retrain
from ape.style.sample import SceneDescription
from test_style_model import _samples


def _catalogue(path: Path) -> sessionmaker:
    engine = engine_for(path)
    init_db(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _scene(sample, anchor: float) -> SceneDescription:
    return SceneDescription(
        features=sample.features,
        embedding=sample.embedding,
        context=sv.StyleContext(5200.0, 3.0, anchor),
        auto=auto.AutoResult(-1.0, anchor, 4.0, -1.0),
    )


def _trained_profile(maker: sessionmaker, n: int = 40) -> int:
    with session_scope(maker) as session:
        profile = StyleProfile(name="Evento", notes="30 coppie", raw_dir="/non/esiste")
        session.add(profile)
        session.flush()
        for s in _samples(n, seed=4):
            session.add(
                StyleSample(
                    profile_id=profile.id,
                    raw_path=f"/non/esiste/DSC{s.sample_id:05d}.ARW",
                    reference_path=f"/non/esiste/evento_{s.sample_id:03d}.jpg",
                    params={},
                    vector=[float(x) for x in s.vector],
                    context={
                        "as_shot_temperature_k": 5000.0,
                        "as_shot_tint": 0.0,
                        "exposure_anchor_ev": s.exposure_anchor_ev,
                    },
                    scene_features=encode_features(s.features),
                    embedding=encode_embedding(s.embedding),
                    thumbnail=b"\xff\xd8 finta miniatura \xff\xd9",
                    status=StyleSampleStatus.READY,
                    delta_e=3.0,
                    pairing="xmp",
                )
            )
        session.flush()
        assert retrain(session, profile)
        return profile.id


def test_profile_round_trip_predicts_identically(tmp_path: Path):
    source = _catalogue(tmp_path / "a.db")
    profile_id = _trained_profile(source)
    with session_scope(source) as session:
        data = portable.export_profile(session, session.get(StyleProfile, profile_id))
        original = load(session, profile_id)

    target = _catalogue(tmp_path / "vuoto.db")  # another machine, no RAWs anywhere
    with session_scope(target) as session:
        imported = portable.import_profile(session, data)
        imported_id = imported.id
        assert imported.name == "Evento" and not imported.builtin
        assert imported.embedding_centroid is not None
    with session_scope(target) as session:
        copy = load(session, imported_id)
        rows = session.query(StyleSample).filter_by(profile_id=imported_id).all()
        assert len(rows) == 40 and all(r.thumbnail for r in rows)
        ids = {r.id for r in rows}

    rng = np.random.default_rng(11)
    for sample in _samples(10, seed=99):
        anchor = float(rng.uniform(-1, 2))
        a, b = original.predict(_scene(sample, anchor)), copy.predict(_scene(sample, anchor))
        assert np.array_equal(a.vector, b.vector)
        context = _scene(sample, anchor).context
        assert sv.to_params(a.vector, context) == sv.to_params(b.vector, context)
        # The neighbours point at the imported rows, whose thumbnails are there.
        assert {i for i, _w in b.neighbours} <= ids
        assert [w for _i, w in a.neighbours] == [w for _i, w in b.neighbours]


def test_import_never_overwrites(tmp_path: Path):
    maker = _catalogue(tmp_path / "a.db")
    profile_id = _trained_profile(maker, n=10)
    with session_scope(maker) as session:
        data = portable.export_profile(session, session.get(StyleProfile, profile_id))
        again = portable.import_profile(session, data)
        assert again.id != profile_id and again.name == "Evento (2)"


def test_import_refuses_what_is_not_a_profile(tmp_path: Path):
    maker = _catalogue(tmp_path / "a.db")
    with session_scope(maker) as session:
        with pytest.raises(portable.PortableError, match="non è un profilo"):
            portable.import_profile(session, b"PK non proprio uno zip")
        newer = io.BytesIO()
        with zipfile.ZipFile(newer, "w") as archive:
            archive.writestr(
                "format.json", json.dumps({"format": "apestyle", "format_version": 99})
            )
        with pytest.raises(portable.PortableError, match="più recente"):
            portable.import_profile(session, newer.getvalue())

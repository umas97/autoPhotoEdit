# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 10 of docs/SPEC_rimozione.md: a removal is its photo's own (R6).

"Applica alla scena" carries the representative's correction, not its erased
cables, and leaves each photo's removals where they were; a style vector
neither reads nor writes them, so learning, predicting, the built-in presets
and the review's variants keep a photo's removals and never add any; a
correction proposed to the style, and a ``.apestyle``, carry none.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from ape.pipeline.params import EditParams
from ape.style import vector as sv

_HEAL = {"kind": "heal", "id": "h1", "cx": 0.3, "cy": 0.4, "sx": 0.35, "sy": 0.4}
_ERASE = {"kind": "erase", "id": "e1", "area": "ab" * 32}
_CONTEXT = sv.StyleContext(as_shot_temperature_k=5200.0, as_shot_tint=3.0)


def _with(*items, **sections) -> EditParams:
    return EditParams.model_validate({**sections, "retouch": list(items)})


def test_a_style_vector_neither_reads_nor_writes_removals():
    plain = _with(exposure={"ev": 0.4})
    retouched = _with(_HEAL, _ERASE, exposure={"ev": 0.4})
    assert np.array_equal(sv.from_params(plain, _CONTEXT), sv.from_params(retouched, _CONTEXT))
    vector = sv.from_params(plain, _CONTEXT)
    assert sv.to_params(vector, _CONTEXT).retouch == []
    assert sv.to_params(vector, _CONTEXT, base=retouched).retouch == retouched.retouch


def test_the_style_signature_ignores_removals():
    """A removal is not a hand-made style change: predictions keep coming."""
    from ape.style.apply import style_signature

    assert style_signature(_with(_HEAL)) == style_signature(_with())


def test_a_preset_keeps_the_photo_s_removals_and_adds_none():
    """Presets and learned profiles alike reach a photo through ``params_for``."""
    from ape.style.builtin import BuiltinRules
    from ape.style.predict import params_for

    class _Preset:
        def predict(self, _scene):
            class _Prediction:
                vector = sv.from_params(EditParams(), _CONTEXT) + 0.1 * sv.UNITS

            return _Prediction()

    class _Scene:
        context = _CONTEXT

    assert BuiltinRules is not None
    current = _with(_ERASE)
    assert params_for(_Preset(), _Scene(), current).retouch == current.retouch
    assert params_for(_Preset(), _Scene(), EditParams()).retouch == []


def test_applying_to_the_scene_leaves_each_photo_s_removals(catalog):
    from ape.api.routes_photos import add_version
    from ape.db.models import EditVersion, EditVersionSource, Photo, PhotoStatus, Project
    from ape.db.style import StyleProfile
    from ape.review.decisions import apply_to_scene

    with catalog() as session:
        profile = StyleProfile(name="Stile")
        session.add(profile)
        session.flush()
        project = Project(name="scena", source_dir="/nonexistent", style_profile_id=profile.id)
        session.add(project)
        session.flush()
        photos = []
        for rank, own in enumerate(([_HEAL], [_ERASE])):
            photo = Photo(
                project_id=project.id, filename=f"DSC0{rank}.ARW", path=f"/nonexistent/{rank}",
                status=PhotoStatus.PREDICTED, cluster_id=1, cluster_rank=rank,
                prediction={"profile_id": profile.id, "applied": True,
                            "vector": [float(v) for v in sv.from_params(EditParams(), _CONTEXT)],
                            "context": {"as_shot_temperature_k": 5200.0, "as_shot_tint": 3.0}},
            )
            session.add(photo)
            session.flush()
            add_version(session, photo, _with(*own), EditVersionSource.PREDICTED)
            photos.append(photo)
        session.flush()
        representative, other = photos
        correction = _with(_HEAL, {**_ERASE, "id": "e9"}, exposure={"ev": 0.6})
        result = apply_to_scene(session, project, 1, correction)
        assert result["propagated"] == 1

        def current(photo):
            version = session.query(EditVersion).filter_by(photo_id=photo.id,
                                                           is_current=True).one()
            return EditParams.from_dict(version.params)

        assert current(representative).retouch == correction.retouch
        moved = current(other)
        assert [item.model_dump() for item in moved.retouch] == [
            EditParams.model_validate({"retouch": [_ERASE]}).retouch[0].model_dump()
        ]
        assert moved.exposure.ev > 0.3, "la correzione di stile è arrivata"


def test_a_correction_proposed_to_the_style_carries_no_removal(catalog):
    """``review/feedback.sync`` turns an approved correction into a sample."""
    from ape.api.routes_photos import add_version
    from ape.db.models import EditVersionSource, Photo, PhotoStatus, Project
    from ape.db.style import StyleProfile
    from ape.review import feedback

    with catalog() as session:
        profile = StyleProfile(name="Stile")
        session.add(profile)
        session.flush()
        project = Project(name="f", source_dir="/nonexistent", style_profile_id=profile.id)
        session.add(project)
        session.flush()
        photo = Photo(
            project_id=project.id, filename="DSC0.ARW", path="/nonexistent/0",
            status=PhotoStatus.APPROVED, review={"decision": "approved", "by": "photo"},
            prediction={"applied_signature": "predetta",
                        "context": {"as_shot_temperature_k": 5200.0, "as_shot_tint": 3.0}},
        )
        session.add(photo)
        session.flush()
        add_version(session, photo, _with(_HEAL, _ERASE, exposure={"ev": 0.7}),
                    EditVersionSource.USER_EDITED)
        sample = feedback.sync(session, project, photo)
        assert sample is not None, "la correzione deve diventare un campione"
        assert sample.params["retouch"] == []
        assert sample.params["exposure"]["ev"] == 0.7


def test_an_apestyle_carries_no_removal(catalog):
    from ape.db.models import StyleSampleStatus
    from ape.db.style import StyleProfile, StyleSample
    from ape.style.portable import export_profile, import_profile

    with catalog() as session:
        profile = StyleProfile(name="Portato")
        session.add(profile)
        session.flush()
        params = _with(_HEAL, exposure={"ev": 0.2}).model_dump(mode="json")
        session.add(StyleSample(profile_id=profile.id, params=params,
                                status=StyleSampleStatus.READY, vector=[0.0] * len(sv.NAMES),
                                context={"as_shot_temperature_k": 5200.0, "as_shot_tint": 0.0}))
        session.flush()
        data = export_profile(session, profile)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            text = "".join(
                archive.read(name).decode() for name in archive.namelist()
                if name.endswith(".json")
            )
        assert "retouch" not in text and '"heal"' not in text
        imported = import_profile(session, data, name="Importato")
        session.flush()
        samples = session.query(StyleSample).filter_by(profile_id=imported.id).all()
        assert samples and all("retouch" not in (s.params or {}) for s in samples)
        assert json.dumps(samples[0].params).count("exposure") == 1

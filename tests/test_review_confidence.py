# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The confidence score of section 9.1, term by term, and the variants of 9.2.

Pure functions, so every case is a few numbers: each term raises its doubt at
the level its comment names and not before; weights and the product behave;
the variants answer the reason the photo is in the queue.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.pipeline.params import EditParams
from ape.review.confidence import (
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    ConfidenceInputs,
    score,
    weights_from,
)
from ape.review.variants import MAX_VARIANTS, VariantInputs, variants
from ape.style import vector as sv


def _codes(inputs: ConfidenceInputs) -> list[str]:
    return [t.code for t in score(inputs).reasons()]


def test_nothing_known_is_full_confidence():
    result = score(ConfidenceInputs())
    assert result.value == pytest.approx(1.0)
    assert result.reasons() == []


def test_an_ordinary_photo_of_a_covered_scene_passes():
    inputs = ConfidenceInputs(
        nearest=0.8, nearest_p90=1.0, dispersion=0.9, dispersion_p90=1.0,
        exposure_anchor_ev=0.4, anchor_range=(-0.5, 1.5),
        wb_spread_mired=25.0, burnt=0.004, crushed=0.01,
        straighten={"outcome": "rotated", "confidence": 0.8, "rotation_deg": 0.6},
        lens_profile=True,
    )
    result = score(inputs)
    assert result.value == pytest.approx(1.0)
    assert result.value >= DEFAULT_THRESHOLD


@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        (ConfidenceInputs(nearest=2.5, nearest_p90=1.0), "far"),
        (ConfidenceInputs(dispersion=2.1, dispersion_p90=1.0), "ambiguous"),
        (ConfidenceInputs(exposure_anchor_ev=2.8, anchor_range=(-0.5, 1.5)), "extrapolation"),
        (ConfidenceInputs(wb_spread_mired=130.0), "white_balance"),
        (ConfidenceInputs(burnt=0.1), "burnt"),
        (ConfidenceInputs(crushed=0.2), "crushed"),
        (ConfidenceInputs(straighten={"outcome": "contradictory"}), "geometry"),
    ],
)
def test_each_doubt_alone_at_full_risk_escalates(inputs, code):
    result = score(inputs)
    assert result.value < DEFAULT_THRESHOLD
    assert _codes(inputs) == [code]


def test_below_the_ramps_nothing_is_raised():
    inputs = ConfidenceInputs(
        nearest=1.0, nearest_p90=1.0, dispersion=1.0, dispersion_p90=1.0,
        exposure_anchor_ev=1.7, anchor_range=(-0.5, 1.5),  # 0.2 EV out: noise
        wb_spread_mired=40.0, burnt=0.02, crushed=0.05,
    )
    assert score(inputs).value == pytest.approx(1.0)


def test_a_missing_lens_profile_alone_never_escalates():
    result = score(ConfidenceInputs(lens_profile=False))
    assert result.value == pytest.approx(1.0 - DEFAULT_WEIGHTS["lens"])
    assert result.value > DEFAULT_THRESHOLD
    assert _codes(ConfidenceInputs(lens_profile=False)) == ["lens"]


def test_small_doubts_compound():
    one = ConfidenceInputs(wb_spread_mired=70.0)  # half risk each
    two = ConfidenceInputs(wb_spread_mired=70.0, burnt=0.05)
    three = ConfidenceInputs(wb_spread_mired=70.0, burnt=0.05, crushed=0.10)
    assert score(one).value == pytest.approx(0.75)
    assert score(two).value == pytest.approx(0.75 * 0.75)
    assert score(two).value > DEFAULT_THRESHOLD
    assert score(three).value < DEFAULT_THRESHOLD


def test_a_rotation_the_user_set_is_not_a_doubt():
    level = {"outcome": "contradictory"}
    assert score(ConfidenceInputs(straighten=level, rotation_by_user=True)).value == 1.0


def test_geometry_doubt_grows_with_the_straightening_uncertainty():
    sure = score(ConfidenceInputs(straighten={"outcome": "rotated", "confidence": 0.7}))
    unsure = score(ConfidenceInputs(straighten={"outcome": "rotated", "confidence": 0.35}))
    assert sure.value == pytest.approx(1.0)
    assert unsure.value < sure.value


def test_weights_are_the_users_and_bounded():
    weights = weights_from({"white_balance": 0.0, "lens": 3.0, "unknown": 1.0})
    assert weights["white_balance"] == 0.0 and weights["lens"] == 1.0
    assert "unknown" not in weights
    assert score(ConfidenceInputs(wb_spread_mired=200.0), weights).value == 1.0
    assert score(ConfidenceInputs(lens_profile=False), weights).value == 0.0


def test_reasons_are_heaviest_first():
    inputs = ConfidenceInputs(wb_spread_mired=70.0, burnt=0.2, lens_profile=False)
    assert _codes(inputs) == ["burnt", "lens", "white_balance"]


# --- variants ---------------------------------------------------------------


def _context() -> sv.StyleContext:
    return sv.StyleContext(as_shot_temperature_k=5200.0, as_shot_tint=4.0, exposure_anchor_ev=0.6)


def _current() -> EditParams:
    vector = sv.neutral_vector()
    vector[sv.index("wb_mired_shift")] = 12.0
    vector[sv.index("contrast")] = 1.4
    params = sv.to_params(vector, _context())
    params.geometry.rotation_deg = 1.2
    params.sharpen.amount = 0.8
    return params


def _keys(reasons, **extra) -> list[str]:
    inputs = VariantInputs(current=_current(), context=_context(), reasons=reasons, **extra)
    return [v.key for v in variants(inputs)]


def test_variants_answer_the_reason():
    neutral = sv.neutral_vector()
    assert _keys(["white_balance"], auto_wb=(-8.0, 2.0))[:2] == ["camera_wb", "auto_wb"]
    assert _keys(["burnt"])[0] == "darker"
    assert _keys(["crushed"])[0] == "brighter"
    assert _keys(["geometry"])[0] == "no_rotation"
    knn = sv.from_params(_current(), _context())
    knn[sv.index("exposure_offset")] += 0.8
    assert _keys(["far"], knn_vector=knn, neutral_vector=neutral) == [
        "neighbours", "brighter", "darker"
    ]


def test_there_are_always_two_or_three_distinct_variants():
    for reasons in ([], ["lens"], ["far", "white_balance"], ["geometry", "burnt"]):
        found = variants(
            VariantInputs(
                current=_current(), context=_context(), reasons=reasons,
                neutral_vector=sv.neutral_vector(),
            )
        )
        assert 2 <= len(found) <= MAX_VARIANTS
        vectors = [sv.from_params(v.params, _context()) for v in found]
        base = sv.from_params(_current(), _context())
        for i, vector in enumerate(vectors):
            same_geometry = found[i].params.geometry == _current().geometry
            assert not same_geometry or np.max(np.abs(vector - base) / sv.UNITS) >= 0.5


def test_variants_keep_what_is_not_style():
    for variant in variants(
        VariantInputs(current=_current(), context=_context(), reasons=["far"])
    ):
        assert variant.params.sharpen == _current().sharpen
        if variant.key != "no_rotation":
            assert variant.params.geometry == _current().geometry
    assert variants(
        VariantInputs(current=_current(), context=_context(), reasons=["geometry"])
    )[0].params.geometry.rotation_deg == 0.0


def test_a_variant_equal_to_the_current_edit_is_dropped():
    current = _current()
    same = sv.from_params(current, _context())
    keys = _keys(["far"], knn_vector=same)
    assert "neighbours" not in keys


def test_a_catalogue_of_schema_4_is_migrated(xdg_home):
    """Schema 5: the review's columns arrive, and the old 0.6 default becomes 0.55."""
    from sqlalchemy import inspect, text

    from ape.db.models import Project
    from ape.db.session import SCHEMA_VERSION, get_engine, init_db, read_setting, session_scope

    engine = get_engine()
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE photo DROP COLUMN review"))
        connection.execute(text("ALTER TABLE project DROP COLUMN confidence_weights"))
        # SQLite cannot drop a column with a foreign key: the table of schema 4
        # is rebuilt without the two columns instead.
        kept = [
            c["name"]
            for c in inspect(connection).get_columns("style_sample")
            if c["name"] not in ("project_id", "photo_id")
        ]
        connection.execute(
            text(f"CREATE TABLE old_sample AS SELECT {', '.join(kept)} FROM style_sample")
        )
        connection.execute(text("DROP TABLE style_sample"))
        connection.execute(text("ALTER TABLE old_sample RENAME TO style_sample"))
        connection.execute(
            text(
                "INSERT INTO project (name, source_dir, coherence_lambda, confidence_threshold,"
                " culling_enabled, culling_mode, culling_aggressiveness, culling_weights,"
                " culling_criteria, export_template, export_on_conflict, export_strip_gps,"
                " source_missing, merge_detection_enabled, crop_proposals_paused, created_at,"
                " updated_at, status) VALUES ('vecchio', '/x', 0.35, 0.6, 0, 'conservative',"
                " 0.5, '{}', '{}', '{basename}.{ext}', 'ask', 0, 0, 1, 0, '2026-01-01',"
                " '2026-01-01', 'new')"
            )
        )
        connection.execute(text("UPDATE setting SET value = '4' WHERE key = 'schema_version'"))
    init_db(engine)
    inspector = inspect(engine)
    assert "review" in {c["name"] for c in inspector.get_columns("photo")}
    assert {"project_id", "photo_id"} <= {c["name"] for c in inspector.get_columns("style_sample")}
    with session_scope() as session:
        assert int(read_setting(session, "schema_version")) == SCHEMA_VERSION
        old = session.query(Project).one()
        assert old.confidence_threshold == pytest.approx(0.55)
        assert old.confidence_weights == {}


def test_the_style_tolerance_comes_from_the_samples_own_frames(xdg_home):
    """``reference.sample_clipping``: the user's edit against each sample's tails.

    A sample without tails (prepared before they were kept) does not vote.
    """
    from ape.db.models import StyleProfile, StyleSample, StyleSampleStatus
    from ape.db.session import get_engine, init_db, session_scope
    from ape.review import measure
    from ape.review.reference import sample_clipping, style_clipping

    init_db(get_engine())
    rng = np.random.default_rng(3)
    frame = (rng.uniform(0.02, 0.5, size=(64, 96, 3)) * 255).astype(np.uint8)
    tails = measure.neutral_tails(frame)
    brighter = EditParams()
    brighter.exposure.ev = 5.0
    with session_scope() as session:
        profile = StyleProfile(name="prova")
        session.add(profile)
        session.flush()
        for params, context in (
            (brighter, {"exposure_anchor_ev": 0.0, "tails": tails}),
            (EditParams(), {"exposure_anchor_ev": 0.0, "tails": tails}),
            (brighter, {"exposure_anchor_ev": 0.0}),
        ):
            session.add(
                StyleSample(
                    profile_id=profile.id,
                    params=params.model_dump(mode="json"),
                    context=context,
                    status=StyleSampleStatus.READY,
                )
            )
        session.flush()
        table = sample_clipping(session, profile)
        ids = sorted(table)
        assert len(ids) == 2
        assert table[ids[0]][0] > 0.02  # five stops brighter burns the frame
        assert table[ids[1]] == (0.0, 0.0)  # the neutral adds nothing
        assert style_clipping([[ids[0], 1.0], [ids[1], 1.0]], table)[0] == pytest.approx(
            table[ids[0]][0] / 2
        )
        assert style_clipping([[999, 1.0]], table) is None

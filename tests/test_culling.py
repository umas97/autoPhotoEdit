# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 9 of section 13: culling finds what is known to be wrong.

    "su un set con scarti noti (10 foto volutamente mosse / fuori fuoco /
    bruciate tra 100), la modalità conservativa identifica almeno 8 dei 10
    scarti e non ne scarta più di 3 valide; le raffiche note vengono
    raggruppate correttamente."

The set is synthetic (``culling_scenes.py``) so that "known" means known by
construction. One reading of the sentence needs stating: conservative mode
discards burst duplicates by design (section 7.4), and those are valid photos
too. What the test bounds at three is the valid photos discarded *as defective*
-- out of focus, moved, blown -- which is the mistake the sentence is about.
Burst duplicates are checked separately: exactly the bursts that were built,
exactly one frame proposed from each, never the frame made worse on purpose.

The second half runs on the real ARW files of ``tests/fixtures``: they are good
photographs, taken by the user, and none of them may be discarded as defective.
"""

from __future__ import annotations

import pytest

from ape.culling.burst import Shot, group_bursts
from ape.culling.features import analyse_preview
from ape.culling.select import Candidate, CullingSettings, Reason, select
from ape.raw.embedded import EmbeddedPreview
from culling_scenes import build_culling_set, defocus, overexpose, underexpose

_DEFECTS = {Reason.OUT_OF_FOCUS, Reason.MOTION_BLUR, Reason.OVEREXPOSED, Reason.UNDEREXPOSED}


def _decide(shots, analyses, groups, settings=None):
    group_of = {photo_id: members[0] for members in groups for photo_id in members}
    candidates = [
        Candidate(
            id=shot.id,
            sharpness=analyses[shot.id].scores.focus,
            motion=analyses[shot.id].scores.motion,
            exposure=analyses[shot.id].scores.exposure,
            exposure_side=analyses[shot.id].scores.exposure_side,
            laplacian=analyses[shot.id].scores.sharpness.laplacian,
            burst_group=group_of.get(shot.id),
            order=(shot.shot_at, shot.id),
        )
        for shot in shots
    ]
    return {d.id: d for d in select(candidates, settings or CullingSettings())}


@pytest.fixture(scope="module")
def culled_set():
    shots = build_culling_set()
    analyses = {s.id: analyse_preview(EmbeddedPreview(s.image, "embedded")) for s in shots}
    groups = group_bursts(
        Shot(s.id, s.shot_at, (s.id,), analyses[s.id].features["signature"]) for s in shots
    )
    return shots, analyses, groups, _decide(shots, analyses, groups)


def _defective(decision) -> bool:
    return bool(_DEFECTS & set(decision.reasons))


def test_the_set_is_what_the_specification_describes(culled_set):
    shots, *_ = culled_set
    assert len(shots) == 100
    assert sum(1 for s in shots if s.defect) == 10


def test_conservative_mode_finds_at_least_eight_of_the_ten_rejects(culled_set):
    shots, analyses, _, decisions = culled_set
    rejects = [s for s in shots if s.defect]
    found = [s.id for s in rejects if decisions[s.id].culled and _defective(decisions[s.id])]
    missed = {
        s.id: (s.defect, analyses[s.id].scores.focus, analyses[s.id].scores.motion)
        for s in rejects if s.id not in found
    }
    assert len(found) >= 8, f"trovati {len(found)} scarti su 10; mancati: {missed}"


def test_conservative_mode_discards_at_most_three_valid_photos_as_defective(culled_set):
    shots, analyses, _, decisions = culled_set
    wrong = {
        s.id: decisions[s.id].reasons
        for s in shots
        if s.defect is None and _defective(decisions[s.id])
    }
    assert len(wrong) <= 3, f"foto valide scartate come difettose: {wrong}"


def test_the_reason_names_the_defect(culled_set):
    """A photo discarded for the wrong reason is only half right: the user
    fixes a missed focus and a shaken hand differently."""
    shots, _, _, decisions = culled_set
    for shot in shots:
        if shot.defect and _defective(decisions[shot.id]):
            reasons = set(decisions[shot.id].reasons)
            if shot.defect == "overexposed":
                assert reasons == {Reason.OVEREXPOSED}, (shot.id, reasons)
            elif shot.defect == "out_of_focus":
                assert Reason.OUT_OF_FOCUS in reasons, (shot.id, reasons)


def test_the_known_bursts_are_grouped_exactly(culled_set):
    shots, _, groups, _ = culled_set
    expected = {}
    for shot in shots:
        if shot.burst is not None:
            expected.setdefault(shot.burst, []).append(shot.id)
    assert sorted(groups) == sorted(expected.values())


def test_each_burst_proposes_one_frame_and_never_the_weaker_one(culled_set):
    shots, _, groups, decisions = culled_set
    weaker = {s.id for s in shots if s.weaker}
    for members in groups:
        kept = [m for m in members if not decisions[m].culled]
        assert len(kept) == 1, (members, kept)
        assert kept[0] not in weaker
        for other in set(members) - set(kept):
            assert Reason.BURST_DUPLICATE in decisions[other].reasons


def test_two_different_scenes_a_second_apart_are_not_a_burst(culled_set):
    """Time alone is not enough: shots 41 and 42 are a second apart and unrelated."""
    shots, _, groups, _ = culled_set
    assert (shots[41].shot_at - shots[40].shot_at).total_seconds() == 1
    grouped = {photo_id for members in groups for photo_id in members}
    assert shots[40].id not in grouped and shots[41].id not in grouped


def test_the_hard_valid_cases_survive(culled_set):
    """Shallow depth of field, dark but recoverable, noisy, foggy: all keepers."""
    shots, analyses, _, decisions = culled_set
    for shot in shots[:8]:
        assert not decisions[shot.id].culled, (
            shot.id, decisions[shot.id].reasons, analyses[shot.id].scores.focus,
        )


def test_the_most_aggressive_setting_still_spares_the_valid_singles(culled_set):
    """The slider moves thresholds; even at the far end a sharp, well exposed
    single shot is not a technical failure."""
    shots, analyses, groups, _ = culled_set
    decisions = _decide(shots, analyses, groups, CullingSettings(aggressiveness=1.0))
    plain = [s for s in shots[8:70]]
    wrong = [s.id for s in plain if _defective(decisions[s.id])]
    assert len(wrong) <= 3, wrong


# --- the user's own photographs --------------------------------------------


@pytest.fixture(scope="module")
def real_previews(raw_fixtures):
    from ape.raw.embedded import read_embedded_preview

    return {path.name: read_embedded_preview(path) for path in raw_fixtures}


@pytest.mark.fixtures
def test_no_real_photograph_is_discarded_as_defective(real_previews):
    """The 24 A7 III files are good photos: macro with nothing but bokeh around
    the subject, a dark forest, snow under a bright sky. None is a reject."""
    for name, preview in real_previews.items():
        scores = analyse_preview(preview).scores
        candidate = Candidate(
            id=1, sharpness=scores.focus, motion=scores.motion, exposure=scores.exposure,
            exposure_side=scores.exposure_side,
        )
        decision = select([candidate], CullingSettings())[0]
        assert not decision.culled, (name, decision.reasons, decision.criteria)


@pytest.mark.fixtures
@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (lambda image: defocus(image, 3.5), Reason.OUT_OF_FOCUS),
        (lambda image: underexpose(image, 5.5), Reason.UNDEREXPOSED),
    ],
    ids=["fuori-fuoco", "sottoesposta"],
)
def test_real_photographs_damaged_on_purpose_are_discarded(real_previews, damage, reason):
    missed = []
    for name, preview in real_previews.items():
        scores = analyse_preview(EmbeddedPreview(damage(preview.image), "embedded")).scores
        candidate = Candidate(
            id=1, sharpness=scores.focus, motion=scores.motion, exposure=scores.exposure,
            exposure_side=scores.exposure_side,
        )
        decision = select([candidate], CullingSettings())[0]
        if reason not in decision.reasons:
            missed.append(name)
    assert not missed, f"non riconosciute: {missed}"


@pytest.mark.fixtures
def test_real_photographs_blown_by_three_and_a_half_stops_are_mostly_discarded(real_previews):
    """Past the two stops a RAW recovers. Mostly, not always: a dark forest
    pushed three and a half stops has nothing left to blow, and is merely
    bright -- which the RAW does recover."""
    found = 0
    for preview in real_previews.values():
        scores = analyse_preview(EmbeddedPreview(overexpose(preview.image, 3.5), "embedded")).scores
        candidate = Candidate(id=1, exposure=scores.exposure, exposure_side=scores.exposure_side)
        found += select([candidate], CullingSettings())[0].culled
    assert found >= 0.75 * len(real_previews), f"{found} su {len(real_previews)}"

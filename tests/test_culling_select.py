# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The selection of section 7.4, as the pure function it is.

No pixels and no catalogue here: candidates are built by hand, with the scores
a real analysis would have produced, so that each rule of section 7.4 can be
checked in isolation and in milliseconds.

Test 10 of section 13 is here in its purest form -- a user's decision survives
every change of mode, slider, weight and toggle -- and again, through the
catalogue and the HTTP API, in ``test_culling_api.py``.
"""

from __future__ import annotations

import time

import pytest

from ape.culling.select import (
    Candidate,
    CullingSettings,
    Reason,
    select,
    target_count,
    threshold,
)
from ape.db.enums import CullingMode


def good(id_: int, **kwargs) -> Candidate:
    values = {"sharpness": 0.95, "motion": 0.98, "exposure": 1.0, "laplacian": 10.0}
    values.update(kwargs)
    return Candidate(id=id_, order=(id_,), **values)


def by_id(decisions):
    return {d.id: d for d in decisions}


def test_conservative_mode_keeps_everything_that_is_fine():
    decisions = select([good(i) for i in range(1, 11)], CullingSettings())
    assert not any(d.culled for d in decisions)
    assert all(d.reasons == () for d in decisions)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("sharpness", 0.05, Reason.OUT_OF_FOCUS),
        ("motion", 0.05, Reason.MOTION_BLUR),
        ("exposure", 0.05, Reason.OVEREXPOSED),
    ],
)
def test_conservative_mode_discards_a_technical_failure_and_says_why(field, value, reason):
    decisions = by_id(select([good(1), good(2, **{field: value})], CullingSettings()))
    assert not decisions[1].culled
    assert decisions[2].culled
    assert decisions[2].reasons == (reason,)


def test_an_underexposed_photo_is_named_as_such():
    photo = good(1, exposure=0.02, exposure_side="under")
    assert select([photo], CullingSettings())[0].reasons == (Reason.UNDEREXPOSED,)


def test_a_photo_that_could_not_be_judged_is_kept():
    """No structure to judge (a clear sky): kept, never discarded by default."""
    sky = good(1, sharpness=None, motion=None)
    decision = select([sky], CullingSettings())[0]
    assert not decision.culled
    assert decision.criteria == {"exposure": 1.0}


def test_an_unanalysed_photo_is_kept():
    decision = select([Candidate(id=1, analysed=False)], CullingSettings())[0]
    assert not decision.culled and decision.score is None


def test_more_aggressiveness_never_keeps_what_less_discarded():
    photos = [good(i, sharpness=i / 20.0, motion=1 - i / 40.0) for i in range(1, 21)]
    previous: set[int] = set()
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        culled = {d.id for d in select(photos, CullingSettings(aggressiveness=level)) if d.culled}
        assert previous <= culled, f"aggressività {level}: tenute foto prima scartate"
        previous = culled
    assert threshold(0.0) < threshold(0.5) < threshold(1.0)


def test_a_switched_off_criterion_neither_discards_nor_scores():
    photos = [good(1), good(2, sharpness=0.01)]
    settings = CullingSettings(criteria={"sharpness": False, "motion": True,
                                         "exposure": True, "burst": True})
    decisions = by_id(select(photos, settings))
    assert not decisions[2].culled
    assert "sharpness" not in decisions[2].criteria


def test_the_score_is_the_weighted_mean_the_interface_shows():
    photo = good(1, sharpness=0.5, motion=1.0, exposure=0.8)
    decision = select([photo], CullingSettings())[0]
    weights = CullingSettings().weights
    expected = sum(weights[k] * v for k, v in decision.criteria.items()) / sum(
        weights[k] for k in decision.criteria
    )
    assert decision.score == pytest.approx(expected, abs=1e-4)


def test_a_burst_proposes_exactly_one_frame_the_sharpest():
    frames = [good(i, laplacian=value, burst_group=10) for i, value in
              ((10, 8.0), (11, 12.0), (12, 9.0), (13, 11.5))]
    decisions = by_id(select(frames, CullingSettings()))
    kept = [i for i, d in decisions.items() if not d.culled]
    assert kept == [11]
    assert decisions[11].rank == 0
    for other in (10, 12, 13):
        assert decisions[other].reasons == (Reason.BURST_DUPLICATE,)
    assert sorted(d.rank for d in decisions.values()) == [0, 1, 2, 3]


def test_bursts_off_means_no_duplicates():
    frames = [good(i, burst_group=1) for i in range(1, 5)]
    settings = CullingSettings(criteria={"burst": False, "sharpness": True,
                                         "motion": True, "exposure": True})
    assert not any(d.culled for d in select(frames, settings))


def test_the_proposed_frame_is_never_a_technical_failure():
    frames = [good(1, burst_group=1, laplacian=50.0, sharpness=0.02), good(2, burst_group=1)]
    decisions = by_id(select(frames, CullingSettings()))
    assert decisions[2].culled is False
    assert Reason.OUT_OF_FOCUS in decisions[1].reasons


def test_a_user_choice_inside_a_burst_is_the_proposed_frame():
    frames = [good(1, burst_group=1, laplacian=20.0), good(2, burst_group=1, user_keep=True)]
    decisions = by_id(select(frames, CullingSettings()))
    assert decisions[2].culled is False and decisions[2].rank == 0
    assert decisions[1].culled is True


def test_user_decisions_survive_every_setting(settings_variants):
    """Test 10 of section 13, on the pure function."""
    photos = [
        good(1, sharpness=0.01, user_keep=True),    # blurred, and the user wants it
        good(2, user_keep=False),                   # perfect, and the user does not
        good(3, burst_group=5, laplacian=1.0, user_keep=True),
        good(4, burst_group=5, laplacian=99.0),
        good(5),
    ]
    for settings in settings_variants:
        decisions = by_id(select(photos, settings))
        assert decisions[1].culled is False, settings
        assert decisions[2].culled is True and decisions[2].reasons == (Reason.USER,), settings
        assert decisions[3].culled is False, settings


@pytest.fixture
def settings_variants() -> list[CullingSettings]:
    variants = []
    for aggressiveness in (0.0, 0.5, 1.0):
        variants.append(CullingSettings(aggressiveness=aggressiveness))
        variants.append(
            CullingSettings(
                mode=CullingMode.TARGET_PERCENT, target=10, aggressiveness=aggressiveness
            )
        )
        variants.append(CullingSettings(mode=CullingMode.TARGET_COUNT, target=1))
    variants.append(CullingSettings(weights={"sharpness": 0.0, "motion": 1.0, "exposure": 0.0}))
    variants.append(CullingSettings(criteria={k: False for k in ("sharpness", "motion",
                                                                   "exposure", "burst")}))
    return variants


def test_protected_frames_are_never_judged_as_single_shots():
    """Section 25.7: a bracketing's dark frame is dark on purpose."""
    frame = good(1, exposure=0.0, exposure_side="under", protected=True, burst_group=3)
    other = good(2, burst_group=3)
    decisions = by_id(select([frame, other], CullingSettings(aggressiveness=1.0)))
    assert decisions[1].culled is False
    # And it is not a burst member either: the other frame is on its own.
    assert decisions[2].culled is False


def test_target_mode_covers_every_moment_before_a_second_frame_of_any():
    """Section 7.4: two hundred moments, not two hundred variants of one."""
    photos = [good(i, burst_group=1, laplacian=100.0 - i) for i in range(1, 6)]  # one burst
    photos += [good(i, sharpness=0.7 + i / 1000.0) for i in range(10, 14)]      # four singles
    settings = CullingSettings(mode=CullingMode.TARGET_COUNT, target=5)
    decisions = by_id(select(photos, settings))
    kept = {i for i, d in decisions.items() if not d.culled}
    assert len(kept) == 5
    assert {10, 11, 12, 13} <= kept, "ogni momento prima di un secondo fotogramma"
    assert len(kept & {1, 2, 3, 4, 5}) == 1

    denser = by_id(select(photos, CullingSettings(mode=CullingMode.TARGET_COUNT, target=7)))
    assert len([d for d in denser.values() if not d.culled]) == 7


def test_target_mode_never_fills_the_quota_with_failures():
    photos = [good(1), good(2), good(3, sharpness=0.01)]
    settings = CullingSettings(mode=CullingMode.TARGET_COUNT, target=3)
    decisions = by_id(select(photos, settings))
    assert decisions[3].culled and decisions[3].reasons == (Reason.OUT_OF_FOCUS,)
    assert sum(not d.culled for d in decisions.values()) == 2


def test_target_percent_counts_the_whole_project():
    settings = CullingSettings(mode=CullingMode.TARGET_PERCENT, target=30)
    assert target_count(settings, 10) == 3
    assert target_count(settings, 1180) == 354
    decisions = select([good(i, sharpness=0.5 + i / 100) for i in range(1, 11)], settings)
    kept = sorted(d.id for d in decisions if not d.culled)
    assert kept == [8, 9, 10], "le migliori per punteggio"
    assert all(d.reasons == (Reason.BELOW_TARGET,) for d in decisions if d.culled)


def test_user_kept_photos_count_towards_the_target():
    photos = [good(1, sharpness=0.4, user_keep=True)] + [good(i) for i in range(2, 6)]
    settings = CullingSettings(mode=CullingMode.TARGET_COUNT, target=2)
    kept = [d.id for d in select(photos, settings) if not d.culled]
    assert len(kept) == 2 and 1 in kept


def test_selecting_two_thousand_photos_is_instant():
    """Section 14, phase 4: changing mode or slider must not be felt (< 100 ms).

    This is the arithmetic alone; ``test_culling_api.py`` measures the whole
    request, catalogue writes included.
    """
    photos = []
    for i in range(2000):
        group = i // 5 if i % 3 == 0 else None
        photos.append(good(i, sharpness=(i % 97) / 97, motion=(i % 89) / 89,
                           exposure=(i % 83) / 83, burst_group=group, laplacian=float(i % 13)))
    settings = [CullingSettings(aggressiveness=a / 10) for a in range(11)]
    settings.append(CullingSettings(mode=CullingMode.TARGET_PERCENT, target=30))
    started = time.perf_counter()
    for variant in settings:
        select(photos, variant)
    per_selection = (time.perf_counter() - started) / len(settings)
    assert per_selection < 0.05, f"{per_selection * 1000:.1f} ms per selezione"


def test_equal_scores_are_broken_by_the_measured_sharpness():
    """Every clearly sharp photo scores 100; a target must still pick the crispest."""
    photos = [good(i, sharpness=1.0, motion=1.0, acuity=value)
              for i, value in ((1, 0.70), (2, 0.82), (3, 0.75), (4, 0.80))]
    decisions = select(photos, CullingSettings(mode=CullingMode.TARGET_COUNT, target=2))
    assert sorted(d.id for d in decisions if not d.culled) == [2, 4]


def test_a_merge_frame_the_user_discarded_does_not_count_towards_the_target():
    """Regression: a protected frame discarded by hand was still counted as
    kept, so the target mode kept one photo fewer than asked."""
    photos = [good(i) for i in range(1, 11)]
    photos += [good(20, protected=True), good(21, protected=True),
               good(22, protected=True, user_keep=False)]
    decisions = select(photos, CullingSettings(mode=CullingMode.TARGET_COUNT, target=5))
    assert sum(not d.culled for d in decisions) == 5

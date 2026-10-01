# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bracketings are found before culling, and culling leaves them alone.

The part of test 17 (section 13) that phase 4 is responsible for:

    "un bracketing di 3 scatti EXIF-coerenti viene rilevato come gruppo HDR e
    **non** come raffica, e nessuno dei suoi frame viene scartato dalla
    cernita per esposizione"

and, from section 25.2, "un gruppo rifiutato resta rifiutato". The merges
themselves are phase 11; what is checked here is the proposal and its effect on
the selection.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from ape.culling import service
from ape.culling.burst import visual_signature
from ape.culling.features import camera_details
from ape.culling.select import CullingSettings
from ape.db.models import MergeDecision, MergeGroup, MergeKind, Photo, Project
from ape.merge.detect import BracketShot, detect_brackets, exposure_brightness
from ape.raw.metadata import PhotoMetadata
from culling_catalog import AnalysedShot, insert_analysed, make_project
from culling_scenes import natural_scene, overexpose, underexpose

T0 = datetime(2026, 2, 26, 12, 0, 0)

#: The far end of the aggressiveness slider: the harshest the selection gets.
_STRICT = CullingSettings(aggressiveness=1.0)


def shot(id_: int, seconds: float, shutter: float, **kwargs) -> BracketShot:
    values = dict(
        id=id_, shot_at=T0 + timedelta(seconds=seconds), order=(id_,), focal_length=35.0,
        aperture=8.0, iso=100, shutter=shutter, exposure_bias=None, release_mode=None,
        signature=kwargs.pop("signature", {"phash_eq": "0" * 16, "phash": "0" * 16,
                                           "hist": [1.0] + [0.0] * 63}),
    )
    values.update(kwargs)
    return BracketShot(**values)


def test_brightness_follows_shutter_iso_and_aperture():
    base = shot(1, 0, 1 / 250)
    assert exposure_brightness(shot(2, 0, 1 / 125)) - exposure_brightness(base) == pytest.approx(1)
    assert exposure_brightness(shot(3, 0, 1 / 250, iso=400)) - exposure_brightness(
        base
    ) == pytest.approx(2)
    assert exposure_brightness(shot(4, 0, 1 / 250, aperture=5.6)) > exposure_brightness(base)


def test_sonys_own_order_is_recognised():
    """Metered frame first, then under, then over: 0, -2, +2 EV."""
    found = detect_brackets([shot(1, 0, 1 / 250), shot(2, 0, 1 / 1000), shot(3, 1, 1 / 60)])
    assert len(found) == 1
    group = found[0]
    assert group.kind == "hdr"
    assert group.members == (1, 2, 3)
    assert group.reference == 1
    assert group.ev_offsets == pytest.approx((0.0, -2.0, 2.06), abs=0.1)
    assert group.reasons["frames"] == 3


def test_the_longest_sequence_wins():
    frames = [shot(i, i * 0.5, 1 / 250 * 2 ** (i - 2)) for i in range(5)]
    found = detect_brackets(frames)
    assert [g.members for g in found] == [(0, 1, 2, 3, 4)]


@pytest.mark.parametrize(
    "frames",
    [
        # Same exposure three times: a burst, not a bracketing.
        [shot(1, 0, 1 / 250), shot(2, 0, 1 / 250), shot(3, 0, 1 / 250)],
        # A change of focal length: three compositions.
        [shot(1, 0, 1 / 250), shot(2, 0, 1 / 1000, focal_length=50), shot(3, 1, 1 / 60)],
        # Too far apart in time.
        [shot(1, 0, 1 / 250), shot(2, 5, 1 / 1000), shot(3, 10, 1 / 60)],
        # Uneven steps: someone adjusting the exposure by hand.
        [shot(1, 0, 1 / 250), shot(2, 0, 1 / 320), shot(3, 1, 1 / 30)],
        # Not the same picture once brightness is taken out.
        [shot(1, 0, 1 / 250), shot(2, 0, 1 / 1000, signature={"phash_eq": "f" * 16}),
         shot(3, 1, 1 / 60)],
    ],
    ids=["esposizione-costante", "focale-diversa", "troppo-distanti", "passi-irregolari",
         "inquadrature-diverse"],
)
def test_what_is_not_a_bracketing(frames):
    assert detect_brackets(frames) == []


#: The user's own bracketings (``tests/fixtures/fase11``): five frames a third
#: of a stop apart -- the A7 III's default step -- metered frame first, by ISO
#: at a fixed shutter and by shutter at a fixed ISO, with the EXIF values as
#: the camera rounded them.
_BY_ISO = [(1 / 125, 2500), (1 / 125, 2000), (1 / 125, 3200), (1 / 125, 1600), (1 / 125, 4000)]
_BY_SHUTTER = [(1 / 15, 6400), (1 / 20, 6400), (1 / 13, 6400), (1 / 25, 6400), (1 / 10, 6400)]


@pytest.mark.parametrize("exposures", [_BY_ISO, _BY_SHUTTER], ids=["iso", "tempo"])
def test_a_third_of_a_stop_is_a_bracketing_when_the_camera_says_so(exposures):
    frames = [shot(i, i * 0.25, t, iso=iso, aperture=2.8, camera_bracket=True)
              for i, (t, iso) in enumerate(exposures)]
    found = detect_brackets(frames)
    assert [g.members for g in found] == [(0, 1, 2, 3, 4)]
    assert found[0].reference == 0
    assert found[0].reasons["step_ev"] == pytest.approx(1 / 3, abs=0.03)
    assert found[0].reasons["camera_bracket"] is True
    # Without the camera's word a third of a stop is an exposure being
    # corrected -- and no three of the five pass for a bracketing either
    # (07310-07312 once did, at "half a stop").
    assert detect_brackets([replace(f, camera_bracket=False) for f in frames]) == []


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"Exif.SonyMisc3c.ReleaseMode2": "2", "Exif.Sony2.ReleaseMode": "5",
          "Exif.Photo.ExposureMode": "2"}, True),  # 07309: all three agree
        ({"Exif.SonyMisc3c.ReleaseMode2": "26", "Exif.Sony2.ReleaseMode": "2",
          "Exif.Photo.ExposureMode": "0"}, False),  # 07319: a continuous burst
        ({"Exif.Sony2.ReleaseMode": "2"}, False),  # continuous in this tag's own table
        ({"Exif.Sony2.ReleaseMode": "5"}, True),
        ({"Exif.Photo.ExposureMode": "2"}, True),  # the standard tag, any maker
        ({}, None),
    ],
    ids=["bracketing", "raffica", "raffica-sony2", "bracketing-sony2", "exif", "ignoto"],
)
def test_the_camera_says_when_it_was_bracketing(tags, expected):
    assert camera_details(PhotoMetadata(raw_tags=tags))["camera_bracket"] is expected


def _bracket(seed: int, when: datetime) -> list[AnalysedShot]:
    base = natural_scene(seed)
    # +-4 EV. At the most aggressive setting each outer frame, judged alone,
    # fails the exposure criterion -- which is what makes the test mean
    # something: the frames are kept *because* they are a bracketing.
    frames = [(base, 1 / 250), (underexpose(base, 4.0), 1 / 4000), (overexpose(base, 4.0), 1 / 15)]
    return [
        AnalysedShot(image=image, shot_at=when + timedelta(seconds=i // 2), shutter=shutter)
        for i, (image, shutter) in enumerate(frames)
    ]


@pytest.fixture
def bracket_project(catalog, tmp_path):
    with catalog() as session:
        project = make_project(session, tmp_path / "card")
        ids = insert_analysed(session, project, _bracket(7001, T0))
        # A plain photo ten seconds later, for contrast.
        ids += insert_analysed(
            session, project, [AnalysedShot(natural_scene(7002), T0 + timedelta(seconds=10))]
        )
        session.commit()
        return project.id, ids


def test_a_bracketing_is_a_merge_proposal_and_not_a_burst(catalog, bracket_project):
    project_id, ids = bracket_project
    with catalog() as session:
        project = session.get(Project, project_id)
        assert service.ensure_grouped(session, project)
        state = service.apply_selection(session, project, _STRICT)
        session.commit()

        groups = session.scalars(select(MergeGroup)).all()
        assert len(groups) == 1
        group = groups[0]
        assert group.kind is MergeKind.HDR and group.decision is MergeDecision.PROPOSED
        assert sorted(m.photo_id for m in group.members) == ids[:3]

        photos = {p.id: p for p in session.scalars(select(Photo))}
        assert all(photos[i].burst_group_id is None for i in ids[:3]), "non è una raffica"

        decisions = {d.id: d for d in state.decisions}
        for frame in ids[:3]:
            assert not decisions[frame].culled, decisions[frame].reasons


def test_the_outer_frames_would_fail_alone(catalog, bracket_project):
    """The control: without the proposal, the -3 and +3 frames are rejects."""
    project_id, ids = bracket_project
    with catalog() as session:
        project = session.get(Project, project_id)
        project.merge_detection_enabled = False
        service.ensure_grouped(session, project)
        state = service.apply_selection(session, project, _STRICT)
        decisions = {d.id: d for d in state.decisions}
        assert decisions[ids[1]].culled and decisions[ids[2]].culled
        assert not decisions[ids[3]].culled


def test_a_rejected_group_is_not_proposed_again(catalog, bracket_project):
    project_id, ids = bracket_project
    with catalog() as session:
        project = session.get(Project, project_id)
        service.ensure_grouped(session, project)
        group = session.scalars(select(MergeGroup)).one()
        group.decision = MergeDecision.REJECTED
        session.commit()

        # A re-analysis marks everything stale; the next look regroups.
        for photo in session.scalars(select(Photo)):
            photo.burst_rank = None
        session.commit()
        assert service.ensure_grouped(session, project)
        session.commit()

        groups = session.scalars(select(MergeGroup)).all()
        assert len(groups) == 1 and groups[0].decision is MergeDecision.REJECTED
        # Rejected means "these are ordinary photos": judged as such again.
        state = service.apply_selection(session, project, _STRICT)
        decisions = {d.id: d for d in state.decisions}
        assert decisions[ids[2]].culled


def test_the_equalised_hash_survives_the_exposure_steps():
    base = natural_scene(7100)
    from ape.culling.burst import hamming

    reference = visual_signature(base)["phash_eq"]
    for image in (underexpose(base, 2.0), overexpose(base, 2.0)):
        assert hamming(visual_signature(image)["phash_eq"], reference) <= 16


def test_a_bracketing_is_not_cut_short_by_an_analysis_still_running(catalog, tmp_path):
    """Regression: grouping three frames of a five-frame bracketing while the
    other two were still being analysed recorded a three-frame group for good,
    and the two late frames were then judged as single shots."""
    from ape.db.models import Job, JobKind, JobState

    base = natural_scene(7001)
    evs = [-4, -2, 0, 2, 4]
    frames = [
        AnalysedShot(
            image=underexpose(base, -ev) if ev < 0 else overexpose(base, ev) if ev else base,
            shot_at=T0 + timedelta(seconds=i // 2),
            shutter=1 / 250 * 2**ev,
        )
        for i, ev in enumerate(evs)
    ]
    with catalog() as session:
        project = make_project(session, tmp_path / "card")
        ids = insert_analysed(session, project, frames)
        # The last two are still in the queue when the screen first looks.
        late = [Job(project_id=project.id, kind=JobKind.CULL, payload={"photo_id": i},
                    dedupe_key=f"cull:{i}") for i in ids[3:]]
        session.add_all(late)
        session.commit()

        assert service.ensure_grouped(session, project) is False
        service.apply_selection(session, project, _STRICT)
        session.commit()
        assert session.scalars(select(MergeGroup)).all() == []

        for job in late:
            job.state = JobState.DONE
        session.commit()
        assert service.ensure_grouped(session, project)
        state = service.apply_selection(session, project, _STRICT)
        session.commit()
        groups = session.scalars(select(MergeGroup)).all()
        assert [sorted(m.photo_id for m in g.members) for g in groups] == [ids]
        assert not any(d.culled for d in state.decisions if d.id in ids)

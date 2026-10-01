# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Failed merges in the Problems panel, and "Riprova" (section 25.6)."""

from __future__ import annotations

from sqlalchemy import select

from ape import problems
from ape.db.models import (
    Job,
    JobKind,
    JobState,
    MergeDecision,
    MergeGroup,
    MergeKind,
    Photo,
    Project,
)
from ape.merge import virtual


def test_a_failed_merge_is_listed_counted_and_retried(catalog, project):
    project_id, _ = project
    with catalog() as session:
        ids = session.scalars(select(Photo.id).where(Photo.project_id == project_id)).all()
        owner = session.get(Project, project_id)
        group = virtual.create_group(session, owner, MergeKind.PANORAMA, ids)
        virtual.record_failure(session, group, "DSC00003.ARW non si sovrappone abbastanza")
        session.commit()
        before = problems.count(session, project_id)
        listed = problems.overview(session, project_id)["merges"]
        assert [(m["kind"], m["frames"]) for m in listed] == [("panorama", 3)]
        assert "non si sovrappone" in listed[0]["reason"]
        assert before >= 1

        assert problems.retry(session, merge_ids=[group.id]) == 1
        session.commit()
        assert session.get(MergeGroup, group.id).decision is MergeDecision.ACCEPTED
        queued = session.scalars(
            select(Job).where(Job.kind == JobKind.MERGE, Job.state == JobState.QUEUED)
        ).all()
        assert len(queued) == 1
        assert problems.count(session, project_id) == before - 1


def test_a_panoramas_crop_is_proposed_by_the_merge_and_kept_by_the_analysis(catalog, project,
                                                                             tmp_path):
    """Section 25.5.5: proposed, not applied; and not replaced by a composition."""
    from ape.analysis.crop import CropSuggestion
    from ape.db.models import CropDecision, CropProposal
    from ape.jobs.handlers_analysis import _write_proposal

    project_id, _ = project
    crop = {"x": 0.02, "y": 0.05, "width": 0.95, "height": 0.88}
    with catalog() as session:
        ids = session.scalars(select(Photo.id).where(Photo.project_id == project_id)).all()
        owner = session.get(Project, project_id)
        group = virtual.create_group(session, owner, MergeKind.PANORAMA, ids)
        intermediate = tmp_path / "pano.tif"
        intermediate.write_bytes(b"stand-in")
        merged = virtual.record_success(session, group, intermediate, {"crop": crop}, (9000, 3000))
        session.flush()
        composition = CropSuggestion(x=0.1, y=0.1, width=0.8, height=0.8, aspect="4:5",
                                     score=0.9, baseline=0.5)
        _write_proposal(session, merged.id, composition)  # what the analysis would do
        session.commit()
        proposals = session.scalars(
            select(CropProposal).where(CropProposal.photo_id == merged.id)
        ).all()
        assert [(p.aspect, p.decision, p.rect) for p in proposals] == [
            ("borders", CropDecision.PENDING, crop)
        ]

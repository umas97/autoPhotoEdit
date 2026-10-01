# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A merge from group to export, through the catalogue and the real jobs.

Section 25.1: a merged photo "da lì in poi è una foto come tutte le altre" --
proxy, analysis, working set, export -- and its frames step aside. And the
parts of test 17 that are about the catalogue:

* "§2 vale anche qui: dopo rilevamento, fusione ed export, gli hash dei RAW
  membri sono invariati";
* "cancellando l'intermedio in cache, la foto derivata si rigenera identica
  (test di determinismo) senza toccare gli originali".

The RAWs are stand-in files -- the importer reads bytes, not pixels -- and the
merge decodes them, through a patched decoder, into a synthetic bracketing of a
real A7 III frame. Everything downstream of the merge reads the intermediate
with the real decoder.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from ape.db.models import (
    ExportItem,
    MergeDecision,
    MergeGroup,
    MergeKind,
    Photo,
    PhotoKind,
    Project,
)
from ape.db.workset import editing_set
from ape.merge import virtual
from export_helpers import run_queue, start_batch
from merge_bracketing import T0


def _hashes(folder: Path) -> dict[str, tuple[str, float]]:
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime)
        for p in sorted(folder.iterdir())
    }


def _group(catalog, project_id: int) -> int:
    with catalog() as session:
        project = session.get(Project, project_id)
        ids = session.scalars(select(Photo.id).where(Photo.project_id == project_id)).all()
        group = virtual.create_group(session, project, MergeKind.HDR, ids)
        session.commit()
        return group.id


def test_a_merge_goes_from_group_to_export_and_back(catalog, bracketing, project, tmp_path):
    project_id, source = project
    before = _hashes(source)
    group_id = _group(catalog, project_id)
    with catalog() as session:
        group = session.get(MergeGroup, group_id)
        offsets = {m.photo_id: m.ev_offset for m in group.members}
        reference = next(m for m in group.members if m.role.value == "reference")
        # The metered frame is the reference, and the offsets come from EXIF.
        assert session.get(Photo, reference.photo_id).filename == "DSC00001.ARW"
        assert sorted(offsets.values()) == pytest.approx([-2.0, 0.0, 2.0], abs=0.1)

        virtual.request_preview(session, group)
        session.commit()
    run_queue()
    with catalog() as session:
        preview = session.get(MergeGroup, group_id).report["preview"]
    assert "error" not in preview, preview
    from ape.config import get_settings

    assert (get_settings().merge_preview_dir / preview["file"]).is_file()

    with catalog() as session:
        virtual.accept(session, session.get(MergeGroup, group_id))
        session.commit()
    run_queue()

    with catalog() as session:
        group = session.get(MergeGroup, group_id)
        assert group.decision is MergeDecision.ACCEPTED and group.error is None
        merged = session.get(Photo, group.result_photo_id)
        assert merged.kind is PhotoKind.MERGED and merged.path is None
        assert merged.filename == "DSC00001_HDR"
        assert merged.shot_at.replace(tzinfo=None) == T0 + timedelta(seconds=1)  # the first
        assert merged.proxy_path and Path(merged.proxy_path).is_file()
        assert merged.analysis is not None  # analysed like any photo
        working = session.scalars(select(Photo.id).where(*editing_set(project_id))).all()
        assert working == [merged.id]  # the frames stepped aside
        merged_id, intermediate = merged.id, Path(merged.intermediate_path)

    # Export: the merged photo, not its frames, and no XMP for it.
    batch = start_batch(catalog, project_id, output_dir=str(tmp_path / "out"),
                        xmp_darktable=True, xmp_adobe=True, long_edge=800)
    run_queue()
    with catalog() as session:
        items = session.scalars(select(ExportItem).where(ExportItem.batch_id == batch)).all()
        assert [(i.photo_id, i.state.value) for i in items] == [(merged_id, "done")]
        assert not items[0].sidecars
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["DSC00001_HDR.jpg"]

    # Determinism: the cache loses the intermediate, a worker rebuilds it
    # byte for byte, the originals untouched.
    original = intermediate.read_bytes()
    intermediate.unlink()
    from ape.raw.source import make_available

    rebuilt = make_available(catalog, merged_id, intermediate)
    assert rebuilt == intermediate and rebuilt.read_bytes() == original

    # Undo, then undo the undo: the same merged photo comes back, history and all.
    with catalog() as session:
        virtual.undo(session, session.get(MergeGroup, group_id))
        session.commit()
        working = set(session.scalars(select(Photo.id).where(*editing_set(project_id))).all())
        assert merged_id not in working and len(working) == 3
        virtual.accept(session, session.get(MergeGroup, group_id))
        session.commit()
    run_queue()
    with catalog() as session:
        group = session.get(MergeGroup, group_id)
        assert group.result_photo_id == merged_id
        working = session.scalars(select(Photo.id).where(*editing_set(project_id))).all()
        assert working == [merged_id]

    assert _hashes(source) == before


def test_a_merge_that_cannot_be_made_fails_without_a_photo(catalog, bracketing, project):
    project_id, source = project
    group_id = _group(catalog, project_id)
    (source / "DSC00003.ARW").unlink()  # a frame left the card before the merge ran
    with catalog() as session:
        virtual.accept(session, session.get(MergeGroup, group_id))
        session.commit()
    run_queue()
    with catalog() as session:
        group = session.get(MergeGroup, group_id)
        assert group.decision is MergeDecision.FAILED
        assert "DSC00003.ARW" in group.error and group.result_photo_id is None
        assert not session.scalars(select(Photo).where(Photo.kind == PhotoKind.MERGED)).all()
        assert not session.scalars(select(Photo).where(Photo.superseded.is_(True))).all()


def test_groups_made_by_hand_are_checked(catalog, project):
    project_id, _ = project
    with catalog() as session:
        project = session.get(Project, project_id)
        ids = session.scalars(select(Photo.id).where(Photo.project_id == project_id)).all()
        with pytest.raises(ValueError, match="almeno due"):
            virtual.create_group(session, project, MergeKind.HDR, ids[:1])
        group = virtual.create_group(session, project, MergeKind.PANORAMA, ids, ids[0])
        assert {m.role.value for m in group.members if m.photo_id == ids[0]} == {"reference"}
        virtual.edit_group(session, group, ids[:2], None)
        assert sorted(m.photo_id for m in group.members) == sorted(ids[:2])
        virtual.reject(session, group)
        with pytest.raises(ValueError, match="proposta"):
            virtual.edit_group(session, group, ids, None)

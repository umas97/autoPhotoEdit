# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export batches: the plan, the questions, pause and resume, and test 11.

Test 11 of section 13: "corrupt RAW, missing EXIF, unknown lens, a completely
blown or completely black picture: no worker crash, the photo goes to state
``failed`` with a readable message". Here, for the export: an unreadable RAW
fails *that photo*, with a sentence in Italian, and the rest of the batch is
exported; the other four are not errors and must export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ape.db.enums import ExportConflict


@pytest.fixture
def project(catalog, tmp_path, monkeypatch):
    from export_helpers import make_project, synthetic_decode

    synthetic_decode(monkeypatch)
    return make_project(catalog, tmp_path / "card", count=4)


def _summary(catalog, batch_id):
    from ape.db.models import ExportBatch
    from ape.export.batch import batch_summary

    with catalog() as session:
        return batch_summary(session, session.get(ExportBatch, batch_id), with_items=True)


def _plan(catalog, project_id, **changes):
    from ape.db.models import Project
    from ape.export import service
    from ape.export.settings import ExportSettings, load_settings

    with catalog() as session:
        project = session.get(Project, project_id)
        settings = ExportSettings.model_validate(
            load_settings(project).model_dump(mode="json") | changes
        )
        return service.plan(session, project, settings)


# --------------------------------------------------------------------------- #
# The plan refuses what would go wrong later.


def test_no_destination_is_refused(catalog, project):
    from ape.export.service import PlanError

    with pytest.raises(PlanError, match="destinazione"):
        _plan(catalog, project)


def test_the_source_folder_is_refused_as_destination(catalog, project, tmp_path):
    from ape.export.service import PlanError

    for inside in (tmp_path / "card", tmp_path / "card" / "export"):
        with pytest.raises(PlanError, match="RAW"):
            _plan(catalog, project, output_dir=str(inside))


def test_duplicate_names_are_refused_before_the_queue(catalog, project, tmp_path):
    from ape.db.models import Job
    from ape.export.service import PlanError

    with pytest.raises(PlanError, match="stesso nome"):
        _plan(catalog, project, output_dir=str(tmp_path / "out"), template="{camera}.{ext}")
    with catalog() as session:
        assert session.query(Job).count() == 0


def test_the_plan_lists_names_and_collisions(catalog, project, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "DSC00003.jpg").write_bytes(b"x")
    (out / "DSC00001.ARW.xmp").write_bytes(b"x")
    plan = _plan(catalog, project, output_dir=str(out), xmp_darktable=True).describe()
    assert plan["count"] == 4
    assert [n["name"] for n in plan["names"]] == [f"DSC0000{i}.jpg" for i in range(1, 5)]
    assert [(c["filename"], c["files"]) for c in plan["conflicts"]] == [
        ("DSC00001.ARW", ["DSC00001.ARW.xmp"]),
        ("DSC00003.ARW", ["DSC00003.jpg"]),
    ]


def test_the_plan_says_the_sidecars_leave_the_masks_out(catalog, project, tmp_path):
    from ape.api.routes_photos import add_version
    from ape.db.models import EditVersionSource, Photo
    from ape.pipeline.params import EditParams

    with catalog() as session:
        photo = session.query(Photo).filter_by(project_id=project).order_by(Photo.id).first()
        masked = EditParams.model_validate(
            {"masks": [{"kind": "radial", "definition": {}, "exposure": {"ev": 0.5}}]}
        )
        add_version(session, photo, masked, EditVersionSource.USER_EDITED)
        session.commit()
    out = str(tmp_path / "out")
    with_sidecar = _plan(catalog, project, output_dir=out, xmp_adobe=True).describe()
    assert any("1 foto hanno maschere" in w for w in with_sidecar["warnings"])
    images_only = _plan(catalog, project, output_dir=out).describe()
    assert not any("maschere" in w for w in images_only["warnings"])


# --------------------------------------------------------------------------- #
# "Ask": nothing starts until every collision has an answer.


def test_ask_refuses_to_start_until_answered(catalog, project, tmp_path):
    from ape.db.models import Project
    from ape.export import service
    from ape.export.settings import load_settings
    from export_helpers import run_queue

    out = tmp_path / "out"
    out.mkdir()
    (out / "DSC00002.jpg").write_bytes(b"mine")
    (out / "DSC00004.jpg").write_bytes(b"mine too")
    with catalog() as session:
        project_row = session.get(Project, project)
        settings = load_settings(project_row).model_copy(update={"output_dir": str(out)})
        assert settings.on_conflict is ExportConflict.ASK
        with pytest.raises(service.ConflictsPending) as raised:
            service.start(session, project_row, settings)
        assert [c["filename"] for c in raised.value.conflicts] == ["DSC00002.ARW", "DSC00004.ARW"]
        # One answered, one not: still refused, and only the other is asked.
        with pytest.raises(service.ConflictsPending) as again:
            service.start(
                session, project_row, settings, decisions={2: ExportConflict.SKIP}
            )
        assert [c["filename"] for c in again.value.conflicts] == ["DSC00004.ARW"]
        batch = service.start(
            session,
            project_row,
            settings,
            decisions={2: ExportConflict.SKIP, 4: ExportConflict.OVERWRITE},
        )
        # Single answers do not become the project's policy.
        assert project_row.export_on_conflict is ExportConflict.ASK
        session.commit()
        batch_id = batch.id
    run_queue()
    assert (out / "DSC00002.jpg").read_bytes() == b"mine"
    assert (out / "DSC00004.jpg").read_bytes() != b"mine too"
    assert _summary(catalog, batch_id)["counts"] | {} == {
        "queued": 0,
        "done": 3,
        "skipped": 1,
        "failed": 0,
        "cancelled": 0,
    }


def test_apply_to_all_becomes_the_projects_policy(catalog, project, tmp_path):
    from ape.db.models import Project
    from ape.export import service
    from ape.export.settings import load_settings

    out = tmp_path / "out"
    out.mkdir()
    (out / "DSC00001.jpg").write_bytes(b"mine")
    with catalog() as session:
        row = session.get(Project, project)
        settings = load_settings(row).model_copy(update={"output_dir": str(out)})
        service.start(session, row, settings, apply_to_all=ExportConflict.RENAME)
        assert row.export_on_conflict is ExportConflict.RENAME
        session.commit()


# --------------------------------------------------------------------------- #
# Pause, resume, cancel, retry.


def test_pause_holds_the_queue_and_resume_finishes_it(catalog, project, tmp_path):
    from ape.db.models import ExportBatch, Job, JobState
    from ape.export import batch as batches
    from export_helpers import run_queue, start_batch

    out = tmp_path / "out"
    batch_id = start_batch(catalog, project, output_dir=str(out), on_conflict="rename")
    with catalog() as session:
        batch = session.get(ExportBatch, batch_id)
        assert batches.pause(session, batch) == 4
        session.commit()
    run_queue()  # nothing to take
    assert not out.exists() or not list(out.iterdir())
    with catalog() as session:
        assert session.query(Job).filter_by(state=JobState.CANCELLED).count() == 4
        assert batches.resume(session, session.get(ExportBatch, batch_id)) == 4
        session.commit()
    run_queue()
    summary = _summary(catalog, batch_id)
    assert summary["state"] == "done" and summary["counts"]["done"] == 4
    assert len(list(out.glob("*.jpg"))) == 4


def test_cancel_keeps_what_was_written(catalog, project, tmp_path):
    from ape.db.models import ExportBatch
    from ape.export import batch as batches
    from export_helpers import start_batch

    batch_id = start_batch(catalog, project, output_dir=str(tmp_path / "o"), on_conflict="rename")
    with catalog() as session:
        assert batches.cancel(session, session.get(ExportBatch, batch_id)) == 4
        session.commit()
    summary = _summary(catalog, batch_id)
    assert summary["state"] == "cancelled" and summary["counts"]["cancelled"] == 4


def test_a_corrupt_raw_fails_alone_and_can_be_retried(catalog, project, tmp_path):
    """Test 11: one unreadable file, a readable message, the rest exported."""
    from ape.db.models import ExportBatch, Photo, PhotoStatus
    from ape.export import batch as batches
    from export_helpers import run_queue, start_batch

    card = tmp_path / "card"
    original = (card / "DSC00002.ARW").read_bytes()
    (card / "DSC00002.ARW").write_bytes(b"CORRUPT" + original[7:])
    out = tmp_path / "out"
    batch_id = start_batch(catalog, project, output_dir=str(out), on_conflict="rename")
    run_queue()

    summary = _summary(catalog, batch_id)
    assert summary["counts"]["done"] == 3 and summary["counts"]["failed"] == 1
    [problem] = summary["problems"]
    assert problem["filename"] == "DSC00002.ARW"
    assert "illeggibile" in problem["error"]
    with catalog() as session:
        photo = session.get(Photo, problem["photo_id"])
        assert photo.status is PhotoStatus.FAILED and "illeggibile" in photo.error

    # The card is put back as it was; "Riprova" exports it.
    (card / "DSC00002.ARW").write_bytes(original)
    with catalog() as session:
        assert batches.retry_failed(session, session.get(ExportBatch, batch_id)) == 1
        session.commit()
    run_queue()
    summary = _summary(catalog, batch_id)
    assert summary["counts"]["done"] == 4 and summary["state"] == "done"
    assert (out / "DSC00002.jpg").is_file()


def test_a_full_disk_fails_the_item_not_the_photo(catalog, project, tmp_path, monkeypatch):
    import errno

    from ape.db.models import Photo, PhotoStatus
    from export_helpers import run_queue, start_batch

    def full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("ape.export.run.write_exclusive", full)
    batch_id = start_batch(catalog, project, output_dir=str(tmp_path / "o"), on_conflict="rename")
    run_queue()
    summary = _summary(catalog, batch_id)
    assert summary["counts"]["failed"] == 4
    assert all("spazio esaurito" in p["error"] for p in summary["problems"])
    with catalog() as session:
        assert all(p.status is not PhotoStatus.FAILED for p in session.query(Photo))


@pytest.mark.parametrize("fill", [0.0, 1.0, 40.0])
def test_black_and_blown_frames_export_cleanly(catalog, tmp_path, monkeypatch, fill):
    """Test 11: a black frame, a white one, and one far past the white point."""
    from export_helpers import make_project, run_queue, start_batch, synthetic_decode

    synthetic_decode(monkeypatch, fill=fill)
    project = make_project(catalog, tmp_path / "card", count=1)
    out = tmp_path / "out"
    batch_id = start_batch(catalog, project, output_dir=str(out), on_conflict="rename")
    run_queue()
    assert _summary(catalog, batch_id)["counts"]["done"] == 1
    from PIL import Image

    pixels = np.asarray(Image.open(out / "DSC00001.jpg"))
    assert pixels.shape == (96, 128, 3) and np.isfinite(pixels).all()


def test_missing_exif_and_unknown_lens_export(catalog, tmp_path, monkeypatch):
    """Test 11: no EXIF at all, no lens anyone knows -- exported, untagged."""
    from ape.raw.metadata import PhotoMetadata
    from export_helpers import make_project, run_queue, start_batch, synthetic_decode

    synthetic_decode(monkeypatch)
    monkeypatch.setattr("ape.raw.metadata.read_metadata", lambda _p: PhotoMetadata())
    project = make_project(catalog, tmp_path / "card", count=1)
    out = tmp_path / "out"
    batch_id = start_batch(
        catalog, project, output_dir=str(out), on_conflict="rename", xmp_darktable=True,
        xmp_adobe=True,
    )
    run_queue()
    assert _summary(catalog, batch_id)["counts"]["done"] == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "DSC00001.ARW.xmp",
        "DSC00001.jpg",
        "DSC00001.xmp",
    ]


def test_the_edit_exported_is_the_one_current_at_the_start(catalog, project, tmp_path):
    """A version made while the batch runs belongs to the next export."""
    from ape.db.models import EditVersion, ExportItem, Photo
    from ape.pipeline.params import EditParams
    from export_helpers import start_batch

    with catalog() as session:
        photo = session.query(Photo).filter_by(filename="DSC00001.ARW").one()
        first = EditVersion(photo_id=photo.id, params=EditParams().model_dump(), is_current=True)
        session.add(first)
        session.commit()
        first_id = first.id
    batch_id = start_batch(catalog, project, output_dir=str(tmp_path / "o"), on_conflict="rename")
    with catalog() as session:
        item = session.query(ExportItem).filter_by(batch_id=batch_id, position=1).one()
        assert item.version_id == first_id


def test_the_queue_caps_concurrent_exports(catalog, project, tmp_path):
    from ape.db.models import JobKind
    from ape.jobs import queue as jobq
    from export_helpers import start_batch

    start_batch(catalog, project, output_dir=str(tmp_path / "o"), on_conflict="rename")
    limits = {JobKind.EXPORT: 2}
    with catalog() as session:
        claimed = [jobq.claim_job(session, worker_pid=1, limits=limits) for _ in range(3)]
    assert [c is not None for c in claimed] == [True, True, False]
    with catalog() as session:
        jobq.complete_job(session, claimed[0].id)
        session.commit()
        assert jobq.claim_job(session, worker_pid=1, limits=limits) is not None


# --------------------------------------------------------------------------- #
# The screen's API.


def test_the_api_asks_with_409_and_starts_with_answers(catalog, project, tmp_path):
    from fastapi.testclient import TestClient

    from ape.api.app import create_app

    out = tmp_path / "out"
    out.mkdir()
    (out / "DSC00001.jpg").write_bytes(b"mine")
    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as client:
        screen = client.patch(f"/api/projects/{project}/export", json={"output_dir": str(out)})
        assert screen.status_code == 200
        body = screen.json()
        assert body["plan"]["ok"] and body["plan"]["count"] == 4
        assert body["plan"]["conflicts"][0]["files"] == ["DSC00001.jpg"]
        preview = client.get(
            f"/api/projects/{project}/export/name", params={"template": "{counter:03}_{seq}"}
        ).json()
        assert preview == {"name": "001_00001.jpg", "error": None, "filename": "DSC00001.ARW"}

        refused = client.post(f"/api/projects/{project}/export", json={})
        assert refused.status_code == 409
        assert refused.json()["detail"]["conflicts"][0]["filename"] == "DSC00001.ARW"

        started = client.post(
            f"/api/projects/{project}/export",
            json={"answers": [{"photo_id": 1, "policy": "rename"}]},
        )
        assert started.status_code == 201, started.text
        batch = started.json()
        assert batch["total"] == 4 and batch["state"] == "running"
        paused = client.post(f"/api/exports/{batch['id']}/pause").json()
        assert paused["state"] == "paused"
        assert client.get("/api/exports/active").json()["batches"][0]["state"] == "paused"
        assert client.post(f"/api/exports/{batch['id']}/resume").json()["state"] == "running"

        bad = client.post(
            f"/api/projects/{project}/export",
            json={"settings": {"output_dir": str(Path(tmp_path / "card"))}},
        )
        assert bad.status_code == 400

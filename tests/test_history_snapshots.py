# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 16 of section 13: the history of a photo and the snapshots of a project.

Twenty edits, a restore to the fifth, a snapshot, more edits, a rollback to the
snapshot: no version is ever lost, exactly one is current per photo, and the
rollback is itself undoable. Plus what section 23.2 adds: the snapshot carries
the culling and review decisions, automatic ones are capped at ten and not
repeated when nothing changed, and nothing of the source folder is touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from ape.api.app import create_app
from conftest_catalog import file_state


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _project(client: TestClient, card: Path) -> tuple[int, list[int]]:
    project = client.post("/api/projects", json={"name": "storico", "source_dir": str(card)})
    assert project.status_code == 201, project.text
    project_id = project.json()["id"]
    items = client.get(f"/api/projects/{project_id}/photos").json()["items"]
    return project_id, [item["id"] for item in items]


def _edit(client: TestClient, photo_id: int, ev: float) -> dict:
    response = client.put(
        f"/api/photos/{photo_id}/params", json={"params": {"exposure": {"ev": ev}}}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _check_history(catalog, photo_ids: list[int]) -> int:
    """Exactly one current version per photo; returns how many versions exist."""
    from ape.db.models import EditVersion

    with catalog() as session:
        for photo_id in photo_ids:
            current = session.scalar(
                select(func.count()).where(
                    EditVersion.photo_id == photo_id, EditVersion.is_current.is_(True)
                )
            )
            assert current == 1, f"foto {photo_id}: {current} versioni correnti"
        return session.scalar(select(func.count()).select_from(EditVersion))


def _ev(client: TestClient, photo_id: int) -> float:
    return client.get(f"/api/photos/{photo_id}").json()["params"]["exposure"]["ev"]


def test_twenty_edits_a_restore_a_snapshot_and_an_undoable_rollback(
    client: TestClient, catalog, card: Path
):
    before_files = file_state(card)
    project_id, photos = _project(client, card)
    photo, other = photos[0], photos[1]

    versions = [_edit(client, photo, round(0.1 * n, 2))["current_version_id"] for n in range(1, 21)]
    assert len(set(versions)) == 20
    _edit(client, other, -1.0)
    total = _check_history(catalog, [photo, other])

    # Back to the fifth: a new version, the twenty stay.
    restored = client.post(f"/api/photos/{photo}/versions/{versions[4]}/restore").json()
    assert restored["params"]["exposure"]["ev"] == pytest.approx(0.5)
    assert restored["versions"][0]["source"] == "reverted"
    assert _check_history(catalog, [photo, other]) == total + 1

    taken = client.post(f"/api/projects/{project_id}/snapshots", json={"name": "prima di tutto"})
    assert taken.status_code == 201, taken.text
    snapshot = taken.json()["id"]

    for n in range(3):
        _edit(client, photo, 2.0 + n / 10)
    _edit(client, other, 1.5)
    listed = client.get(f"/api/projects/{project_id}/snapshots").json()
    assert listed[0]["name"] == "prima di tutto" and listed[0]["differs"] == 2
    count_before_rollback = _check_history(catalog, [photo, other])

    rollback = client.post(f"/api/projects/{project_id}/snapshots/{snapshot}/restore")
    assert rollback.status_code == 200, rollback.text
    body = rollback.json()
    assert body["restored"] == 2 and body["missing"] == 0
    assert _ev(client, photo) == pytest.approx(0.5) and _ev(client, other) == -1.0
    detail = client.get(f"/api/photos/{photo}").json()
    assert detail["versions"][0]["source"] == "snapshot_restored"
    # Nothing lost: two versions added, none removed.
    assert _check_history(catalog, [photo, other]) == count_before_rollback + 2

    # The rollback is undoable: its "before" snapshot takes the project back.
    undo = client.post(f"/api/projects/{project_id}/snapshots/{body['undo']}/restore").json()
    assert undo["restored"] == 2
    assert _ev(client, photo) == pytest.approx(2.2) and _ev(client, other) == 1.5
    assert _check_history(catalog, [photo, other]) == count_before_rollback + 4
    # ...and so is the undo.
    again = client.post(f"/api/projects/{project_id}/snapshots/{undo['undo']}/restore").json()
    assert again["restored"] == 2 and _ev(client, photo) == pytest.approx(0.5)

    # A snapshot of another project, or none at all, is refused.
    assert client.post(f"/api/projects/{project_id}/snapshots/99999/restore").status_code == 404
    assert (
        client.post(f"/api/projects/{project_id}/snapshots", json={"name": "  "}).status_code == 400
    )

    assert file_state(card) == before_files, "§2 violato nella cartella sorgente"


def test_the_snapshot_carries_the_decisions_and_the_automatic_ones_are_capped(
    client: TestClient, catalog, card: Path
):
    from ape import snapshots
    from ape.db.enums import CullDecidedBy, PhotoStatus, SnapshotKind
    from ape.db.models import Photo, Project, ProjectSnapshot

    project_id, photos = _project(client, card)
    with catalog() as session:
        project = session.get(Project, project_id)
        first = session.get(Photo, photos[0])
        first.culled, first.cull_decided_by = True, CullDecidedBy.USER
        second = session.get(Photo, photos[1])
        second.review = {"decision": "approved", "by": "photo"}
        second.status = PhotoStatus.APPROVED
        session.flush()
        marked = snapshots.take_auto(session, project, snapshots.AUTO_REVIEW)
        assert marked is not None
        # The same passage again, nothing changed: no copy of the last snapshot.
        assert snapshots.take_auto(session, project, snapshots.AUTO_REVIEW) is None

        first.culled, first.cull_decided_by = False, None
        second.review, second.status = None, PhotoStatus.PREDICTED
        session.flush()
        result = snapshots.restore(session, project, marked.id)
        assert result["restored"] == 2
        assert first.culled and first.cull_decided_by is CullDecidedBy.USER
        assert second.review == {"decision": "approved", "by": "photo"}
        assert second.status is PhotoStatus.APPROVED

        # A failure describes the file: a rollback does not un-fail it.
        second.status = PhotoStatus.FAILED
        session.flush()
        snapshots.restore(session, project, result["undo"])
        assert second.status is PhotoStatus.FAILED

        for n in range(snapshots.AUTO_KEEP + 5):
            first.culled = not first.culled
            session.flush()
            assert snapshots.take_auto(session, project, f"passo {n}") is not None
        manual = snapshots.take(session, project, "mio")
        session.commit()
        kinds = session.execute(
            select(ProjectSnapshot.kind, func.count())
            .where(ProjectSnapshot.project_id == project_id)
            .group_by(ProjectSnapshot.kind)
        ).all()
        assert dict(kinds) == {SnapshotKind.AUTO: snapshots.AUTO_KEEP, SnapshotKind.MANUAL: 1}

        # Rolling back to the oldest automatic one does not prune it away.
        oldest = session.scalars(
            select(ProjectSnapshot)
            .where(ProjectSnapshot.kind == SnapshotKind.AUTO)
            .order_by(ProjectSnapshot.id)
        ).first()
        snapshots.restore(session, project, oldest.id)
        assert session.get(ProjectSnapshot, oldest.id) is not None
        assert session.get(ProjectSnapshot, manual.id) is not None
        session.commit()

    deleted = client.delete(f"/api/projects/{project_id}/snapshots/{manual.id}")
    assert deleted.status_code == 204
    names = {s["name"] for s in client.get(f"/api/projects/{project_id}/snapshots").json()}
    assert "mio" not in names

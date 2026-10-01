# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The merges screen over HTTP (section 25.6), with the real jobs run by hand.

The contract the frontend is written against: a group made by hand from the
grid, its preview and when the preview no longer describes the group, the full
merge and the badge in the grid, "Mostra scatti sorgente", undo and reject. And
that nothing merges without a click: every state change below is a request.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from ape.api.app import create_app
from ape.db.models import Photo
from export_helpers import run_queue


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _ids(catalog, project_id: int) -> dict[str, int]:
    with catalog() as session:
        rows = session.execute(
            select(Photo.filename, Photo.id).where(Photo.project_id == project_id)
        ).all()
    return dict(rows)


def _grid(client: TestClient, project_id: int, **params) -> list[dict]:
    response = client.get(f"/api/projects/{project_id}/photos", params=params)
    assert response.status_code == 200, response.text
    return response.json()["items"]


def _only_group(client: TestClient, project_id: int) -> dict:
    body = client.get(f"/api/projects/{project_id}/merges").json()
    assert len(body["groups"]) == 1
    return body["groups"][0]


def test_a_group_goes_from_the_grid_to_a_merged_photo_and_back(
    client, catalog, bracketing, project
):
    project_id, _ = project
    body = client.get(f"/api/projects/{project_id}/merges").json()
    assert body["enabled"] is True and body["groups"] == []
    ids = _ids(catalog, project_id)

    response = client.post(f"/api/projects/{project_id}/merges",
                           json={"kind": "hdr", "photo_ids": list(ids.values())})
    assert response.status_code == 201, response.text
    group = response.json()
    assert group["decision"] == "proposed" and group["reasons"]["manual"] is True
    assert [m["filename"] for m in group["members"]] == sorted(ids)
    assert [m["filename"] for m in group["members"] if m["reference"]] == ["DSC00001.ARW"]
    assert sorted(m["ev_offset"] for m in group["members"]) == pytest.approx([-2, 0, 2], abs=0.1)
    # Creating a group asks for its preview, and only for that.
    assert group["preview"]["state"] == "queued" and group["merge_job"] is None

    run_queue()
    group = _only_group(client, project_id)
    assert group["preview"]["state"] == "ready", group["preview"]
    image = client.get(group["preview"]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"
    assert "immutable" in image.headers["cache-control"]

    # A different reference is a different merge: the preview is out of date.
    edited = client.patch(f"/api/merges/{group['id']}",
                          json={"reference_id": ids["DSC00002.ARW"]}).json()
    assert edited["preview"]["state"] == "stale" and edited["preview"]["url"] is None
    client.patch(f"/api/merges/{group['id']}", json={"reference_id": ids["DSC00001.ARW"]})
    assert _only_group(client, project_id)["preview"]["state"] == "ready"

    accepted = client.post(f"/api/merges/{group['id']}/accept").json()
    assert accepted["decision"] == "accepted" and accepted["merge_job"]["state"] == "queued"
    assert accepted["result_photo_id"] is None  # nothing changes until the job has run
    run_queue()
    group = _only_group(client, project_id)
    assert group["decision"] == "accepted" and group["error"] is None
    merged_id = group["result_photo_id"]
    assert merged_id is not None and group["full_report"]

    # The grid: the merged photo with its badge, the frames behind the filter.
    grid = _grid(client, project_id)
    assert [(p["id"], p["merge"]) for p in grid] == [(merged_id, {"kind": "hdr", "sources": 3})]
    sources = _grid(client, project_id, include_sources=True)
    assert len(sources) == 4
    assert sorted(p["superseded"] for p in sources) == [False, True, True, True]
    assert client.patch(f"/api/merges/{group['id']}",
                        json={"photo_ids": list(ids.values())[:2]}).status_code == 400

    undone = client.post(f"/api/merges/{group['id']}/undo").json()
    assert undone["decision"] == "proposed" and undone["result_photo_id"] is None
    assert sorted(p["filename"] for p in _grid(client, project_id)) == sorted(ids)
    # "Mostra scatti sorgente" never brings back a merged photo that was undone.
    assert len(_grid(client, project_id, include_sources=True)) == 3


def test_a_rejected_group_stays_rejected_until_undone(client, catalog, project):
    project_id, _ = project
    ids = list(_ids(catalog, project_id).values())
    group = client.post(f"/api/projects/{project_id}/merges",
                        json={"kind": "panorama", "photo_ids": ids}).json()
    rejected = client.post(f"/api/merges/{group['id']}/reject").json()
    assert rejected["decision"] == "rejected"
    assert client.post(f"/api/merges/{group['id']}/preview").status_code == 409
    assert client.patch(f"/api/merges/{group['id']}", json={"photo_ids": ids}).status_code == 400
    # Reopening the screen proposes nothing again (section 25.2).
    assert _only_group(client, project_id)["decision"] == "rejected"
    assert client.post(f"/api/merges/{group['id']}/undo").json()["decision"] == "proposed"


def test_what_the_screen_refuses(client, catalog, project):
    project_id, _ = project
    ids = list(_ids(catalog, project_id).values())
    one = client.post(f"/api/projects/{project_id}/merges",
                      json={"kind": "hdr", "photo_ids": [ids[0], ids[0]]})
    assert one.status_code == 400 and "almeno due" in one.json()["detail"]
    foreign = client.post(f"/api/projects/{project_id}/merges",
                          json={"kind": "hdr", "photo_ids": [ids[0], 99999]})
    assert foreign.status_code == 400
    assert client.post(f"/api/projects/{project_id}/merges",
                       json={"kind": "tonemap", "photo_ids": ids}).status_code == 422
    assert client.post("/api/merges/99999/accept").status_code == 404
    assert client.patch("/api/merges/99999", json={}).status_code == 404
    for name in ("../catalog.db", "a" * 64 + ".png", "0" * 64 + ".jpg"):
        assert client.get(f"/api/merges/previews/{name}").status_code == 404

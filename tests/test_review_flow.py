# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 7 end to end, over the API, on the user's files (section 9.2).

A project of real ARWs gets a built-in style; the review screen then shows its
scenes and queue. A correction of a scene's representative reaches the rest of
the scene as a delta, a scene approval approves what is not in doubt, a
rejected photo goes back to neutral and stays in the queue, a variant is
approved, Ctrl+Z undoes it, the threshold moves photos in and out of the queue
but never out of "approved", and the corrections found a new learned profile
when the user confirms. Nothing of the source folder is touched (section 2).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ape.api.app import create_app
from conftest import available_raws
from conftest_catalog import file_state
from test_style_flow import _run_all

pytestmark = [pytest.mark.fixtures, pytest.mark.slow]

#: Three frames at the log pile, two of the pond shore, one of the forest: two
#: scenes of more than one photo, whatever the clustering is computed on.
NAMES = ["DSC05618", "DSC05629", "DSC05630", "DSC05631", "DSC05634", "DSC05635"]


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


@pytest.fixture
def card(tmp_path: Path) -> Path:
    raws = {p.stem: p for p in available_raws()}
    if not all(name in raws for name in NAMES):
        pytest.skip("servono i file ARW di tests/fixtures/")
    folder = tmp_path / "scheda"
    folder.mkdir()
    for name in NAMES:
        shutil.copy2(raws[name], folder)
    return folder


def _vector(client: TestClient, photo_id: int) -> np.ndarray:
    from ape.pipeline.params import EditParams
    from ape.style import vector as sv

    detail = client.get(f"/api/photos/{photo_id}").json()
    return sv.from_params(EditParams.from_dict(detail["params"]), _context(photo_id))


def _context(photo_id: int):
    from ape.db.models import Photo
    from ape.db.session import session_scope
    from ape.style import vector as sv

    with session_scope() as session:
        return sv.StyleContext(**session.get(Photo, photo_id).prediction["context"])


def test_review_scene_queue_undo_feedback(client: TestClient, catalog, card):
    before = file_state(card)
    project = client.post("/api/projects", json={"name": "revisione", "source_dir": str(card)})
    project_id = project.json()["id"]
    _run_all(catalog)

    empty = client.get(f"/api/projects/{project_id}/review").json()
    assert empty["profile"] is None and empty["counts"]["total"] == 0

    natural = next(p for p in client.get("/api/styles").json() if p["name"] == "Naturale")
    client.put(f"/api/projects/{project_id}/style", json={"profile_id": natural["id"]})
    _run_all(catalog)
    review = client.get(f"/api/projects/{project_id}/review").json()
    assert review["pending"] == 0 and review["counts"]["total"] == len(NAMES)
    assert review["threshold"] == pytest.approx(0.55)
    photos = {p["id"]: p for p in review["photos"]}
    assert all(p["confidence"] is not None for p in photos.values())
    queued = {q["photo_id"] for q in review["queue"]}
    assert all(photos[i]["status"] == "needs_review" for i in queued)
    # The queue is ordered by confidence, lowest first (section 9.2.3).
    confidences = [q["confidence"] for q in review["queue"]]
    assert confidences == sorted(confidences)

    # --- a scene: correct the representative, the delta reaches the rest -----
    scene = max(review["scenes"], key=lambda s: len(s["photos"]))
    assert len(scene["photos"]) >= 2, review["scenes"]
    representative, others = scene["photos"][0], scene["photos"][1:]
    assert representative == scene["representative"]
    old = {pid: _vector(client, pid) for pid in scene["photos"]}
    start = client.get(f"/api/photos/{representative}").json()
    params = start["params"]
    params["exposure"]["ev"] += 0.4
    # The editor saves by itself (2026-09-29): the correction is already the
    # representative's current version when "Applica alla scena" is pressed,
    # and the screen says where it started.
    saved = client.put(f"/api/photos/{representative}/params", json={"params": params})
    assert saved.status_code == 200, saved.text
    without_base = client.post(
        f"/api/projects/{project_id}/review/scenes/{scene['cluster']}/apply",
        json={"params": params},
    ).json()
    assert without_base["propagated"] == 0  # measured from the saved correction: nothing
    stranger = client.post(
        f"/api/projects/{project_id}/review/scenes/{scene['cluster']}/apply",
        json={"params": params, "base_version_id": 10**6},
    )
    assert stranger.status_code == 400
    body = {"params": params, "base_version_id": start["current_version_id"]}
    applied = client.post(
        f"/api/projects/{project_id}/review/scenes/{scene['cluster']}/apply", json=body
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["propagated"] == len(others)
    versions = client.get(f"/api/photos/{representative}").json()["versions"]
    assert versions[0]["id"] == saved.json()["current_version_id"]  # not written twice
    from ape.style import vector as sv

    exposure = sv.index("exposure_offset")
    for pid in others:
        moved = _vector(client, pid) - old[pid]
        assert moved[exposure] == pytest.approx(0.4, abs=1e-3)
        assert np.max(np.abs(np.delete(moved, exposure))) < 1e-3
        versions = client.get(f"/api/photos/{pid}").json()["versions"]
        assert versions[0]["source"] == "cluster_applied"

    # Undoing the application puts the representative back where it started too.
    client.post(f"/api/projects/{project_id}/review/undo", json={"undo": applied.json()["undo"]})
    for pid in scene["photos"]:
        assert np.max(np.abs(_vector(client, pid) - old[pid])) < 1e-6
    again = client.post(
        f"/api/projects/{project_id}/review/scenes/{scene['cluster']}/apply",
        json={"params": params, "base_version_id": start["current_version_id"]},
    ).json()
    assert again["propagated"] == len(others)

    approved = client.post(
        f"/api/projects/{project_id}/review/scenes/{scene['cluster']}/approve", json={}
    ).json()
    review = client.get(f"/api/projects/{project_id}/review").json()
    statuses = {p["id"]: p["status"] for p in review["photos"]}
    assert statuses[representative] == "approved"
    assert approved["approved"] == sum(statuses[pid] == "approved" for pid in scene["photos"])
    assert all(statuses[pid] in ("approved", "needs_review") for pid in scene["photos"])
    # The representative was corrected: one proposed training pair.
    assert review["feedback"]["proposed"] == 1

    # --- a photo: reject it, it goes neutral and stays in the queue ------------
    target = next(pid for pid in photos if pid not in scene["photos"])
    rejected = client.post(f"/api/projects/{project_id}/review/photos/{target}/reject")
    assert rejected.status_code == 200, rejected.text
    detail = client.get(f"/api/photos/{target}/review").json()
    assert detail["status"] == "needs_review" and detail["review"]["decision"] == "rejected"
    overview = client.get(f"/api/projects/{project_id}/review").json()
    entry = next(q for q in overview["queue"] if q["photo_id"] == target)
    assert entry["reasons"][0] == "rejected"

    # ...then choose a variant and approve it; Ctrl+Z puts it back.
    assert 2 <= len(detail["variants"]) <= 3
    before_approval = client.get(f"/api/photos/{target}").json()["params"]
    choice = detail["variants"][0]["params"]
    result = client.post(
        f"/api/projects/{project_id}/review/photos/{target}/approve", json={"params": choice}
    ).json()
    overview = client.get(f"/api/projects/{project_id}/review").json()
    assert next(p for p in overview["photos"] if p["id"] == target)["status"] == "approved"
    assert overview["feedback"]["proposed"] == 2
    restored = client.post(
        f"/api/projects/{project_id}/review/undo", json={"undo": result["undo"]}
    ).json()
    assert restored["restored"] == 1
    back = client.get(f"/api/photos/{target}").json()
    assert back["params"] == before_approval and back["versions"][0]["source"] == "reverted"
    overview = client.get(f"/api/projects/{project_id}/review").json()
    assert next(p for p in overview["photos"] if p["id"] == target)["status"] == "needs_review"
    assert overview["feedback"]["proposed"] == 1
    client.post(
        f"/api/projects/{project_id}/review/photos/{target}/approve", json={"params": choice}
    )

    # --- the threshold moves the queue, never the approvals --------------------
    everything = client.put(
        f"/api/projects/{project_id}/review/settings", json={"threshold": 1.0}
    ).json()
    for photo in everything["photos"]:
        assert photo["status"] in ("approved", "needs_review")
    kept = {p["id"] for p in everything["photos"] if p["status"] == "approved"}
    assert representative in kept and target in kept
    nothing = client.put(
        f"/api/projects/{project_id}/review/settings", json={"threshold": 0.0}
    ).json()
    assert {p["id"] for p in nothing["photos"] if p["status"] == "approved"} == kept
    assert nothing["counts"]["queue"] == 0
    bad = client.put(f"/api/projects/{project_id}/review/settings", json={"weights": {"x": 1}})
    assert bad.status_code == 400

    # --- the corrections found a learned profile, on explicit confirmation ------
    feedback = client.get(f"/api/projects/{project_id}/feedback").json()
    assert feedback["proposed"] == 2 and feedback["profile"]["builtin"]
    library = {p["name"] for p in client.get("/api/styles").json()}
    assert not any("revisione" in name for name in library)
    _run_all(catalog)  # the thumbnails of the proposed pairs
    feedback = client.get(f"/api/projects/{project_id}/feedback").json()
    assert all(s["has_thumbnail"] for s in feedback["samples"])
    created = client.post(f"/api/projects/{project_id}/feedback/incorporate", json={})
    assert created.status_code == 200, created.text
    profile = created.json()
    assert profile["name"] == feedback["suggested_name"]
    assert profile["trained"] and profile["n_pairs"] == 2 and not profile["builtin"]
    assert client.get(f"/api/projects/{project_id}/feedback").json()["proposed"] == 0

    assert file_state(card) == before, "§2 violato nella cartella sorgente"


def test_developed_renders_and_the_grid(client: TestClient, catalog, card):
    """Each photo's current version is rendered small, and re-rendered when it
    changes; the grid approves several scenes at once, and one undo takes it back.
    """
    from ape.review.developed import developed_path

    before = file_state(card)
    project_id = client.post(
        "/api/projects", json={"name": "griglia", "source_dir": str(card)}
    ).json()["id"]
    _run_all(catalog)
    natural = next(p for p in client.get("/api/styles").json() if p["name"] == "Naturale")
    client.put(f"/api/projects/{project_id}/style", json={"profile_id": natural["id"]})
    _run_all(catalog)

    review = client.get(f"/api/projects/{project_id}/review").json()
    assert review["developing"] == len(NAMES)
    assert all(p["developed"] is None for p in review["photos"])
    _run_all(catalog)
    review = client.get(f"/api/projects/{project_id}/review").json()
    assert review["developing"] == 0
    target = review["photos"][0]
    version = target["developed"]
    assert version is not None
    served = client.get(f"/api/photos/{target['id']}/developed/{version}")
    assert served.status_code == 200 and served.headers["content-type"] == "image/jpeg"
    assert "immutable" in served.headers["cache-control"]
    assert client.get(f"/api/photos/{target['id']}/developed/999999").status_code == 404

    # A new version is a new render; the old one goes once the new one is there.
    params = client.get(f"/api/photos/{target['id']}").json()["params"]
    params["exposure"]["ev"] += 1.0
    client.put(f"/api/photos/{target['id']}/params", json={"params": params})
    review = client.get(f"/api/projects/{project_id}/review").json()
    assert review["developing"] == 1
    assert next(p for p in review["photos"] if p["id"] == target["id"])["developed"] is None
    _run_all(catalog)
    review = client.get(f"/api/projects/{project_id}/review").json()
    newer = next(p for p in review["photos"] if p["id"] == target["id"])["developed"]
    assert newer not in (None, version)
    assert developed_path(target["id"], newer).is_file()
    assert not developed_path(target["id"], version).is_file()

    # --- the grid: every scene approved at once, and undone at once ------------
    clusters = [s["cluster"] for s in review["scenes"]]
    statuses = {p["id"]: p["status"] for p in review["photos"]}
    done = client.post(
        f"/api/projects/{project_id}/review/scenes/approve", json={"clusters": clusters}
    )
    assert done.status_code == 200, done.text
    assert done.json()["scenes"] == len(clusters)
    after = client.get(f"/api/projects/{project_id}/review").json()
    assert after["counts"]["scenes_done"] == len(clusters)
    queued = {q["photo_id"] for q in review["queue"]}
    representatives = {s["representative"] for s in review["scenes"]}
    for photo in after["photos"]:
        if photo["id"] not in queued or photo["id"] in representatives:
            assert photo["status"] == "approved", photo
    client.post(f"/api/projects/{project_id}/review/undo", json={"undo": done.json()["undo"]})
    undone = client.get(f"/api/projects/{project_id}/review").json()
    assert {p["id"]: p["status"] for p in undone["photos"]} == statuses
    empty = client.post(f"/api/projects/{project_id}/review/scenes/approve", json={"clusters": []})
    assert empty.status_code == 422

    assert file_state(card) == before, "§2 violato nella cartella sorgente"

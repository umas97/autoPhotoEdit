# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Culling through the catalogue and the HTTP API (sections 7 and 14).

Three acceptance criteria of phase 4 are checked end to end here:

* test 10 -- a photo promoted or discarded by the user keeps that decision
  through a recomputation of the scores and through every change of toggle,
  mode and slider;
* "cambiare modalità o slider aggiorna la selezione senza rianalisi percepibile
  (< 100 ms)" -- measured on two thousand photos, catalogue writes included,
  and checked to queue no job at all;
* section 7.6's "il programma non inizia mai a elaborare senza consenso
  esplicito" -- an import that chose culling builds no proxy until the user
  confirms, and confirming builds them for the selection only.
"""

from __future__ import annotations

import shutil
import statistics
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, insert, select

from ape.api.app import create_app
from ape.db.models import Job, JobKind, JobState, Photo, PhotoKind, PhotoStatus, Project
from conftest_catalog import write_raw
from culling_catalog import AnalysedShot, insert_analysed, make_project
from culling_scenes import defocus, natural_scene

T0 = datetime(2026, 2, 26, 9, 0, 0)


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _jobs(session, kind: JobKind) -> int:
    return int(
        session.scalar(
            select(func.count(Job.id)).where(Job.kind == kind, Job.state == JobState.QUEUED)
        )
        or 0
    )


@pytest.fixture
def small_project(catalog, tmp_path):
    """Six analysed photos: four good singles, one blurred, one burst pair."""
    sharp = [natural_scene(8000 + i) for i in range(5)]
    shots = [AnalysedShot(sharp[i], T0 + timedelta(seconds=10 * i)) for i in range(4)]
    shots.append(AnalysedShot(defocus(natural_scene(8100), 4.0), T0 + timedelta(seconds=50)))
    burst = natural_scene(8200)
    shots.append(AnalysedShot(burst, T0 + timedelta(seconds=70)))
    shots.append(AnalysedShot(burst, T0 + timedelta(seconds=71)))
    with catalog() as session:
        project = make_project(session, tmp_path / "card")
        ids = insert_analysed(session, project, shots)
        session.commit()
        return project.id, ids


def test_import_with_culling_builds_no_proxy(client: TestClient, card: Path, catalog):
    """Section 7.2: culling reads the embedded previews; nothing is developed."""
    created = client.post(
        "/api/projects", json={"name": "Scheda", "source_dir": str(card), "import_now": False}
    ).json()
    body = client.post(f"/api/projects/{created['id']}/import", json={"culling": True}).json()
    assert body["queued_jobs"] == 6
    with catalog() as session:
        assert _jobs(session, JobKind.CULL) == 6
        assert _jobs(session, JobKind.PROXY) == 0
    project = client.get(f"/api/projects/{created['id']}").json()
    assert project["culling_enabled"] is True
    assert project["status"] == "culling"


def test_import_straight_to_editing_builds_proxies(client: TestClient, card: Path, catalog):
    created = client.post(
        "/api/projects", json={"name": "Scheda", "source_dir": str(card), "import_now": False}
    ).json()
    client.post(f"/api/projects/{created['id']}/import", json={"culling": False})
    with catalog() as session:
        assert _jobs(session, JobKind.PROXY) == 6
        assert _jobs(session, JobKind.CULL) == 0


def test_the_culling_view_carries_scores_groups_and_reasons(client, small_project):
    project_id, ids = small_project
    view = client.get(f"/api/projects/{project_id}/culling").json()
    assert view["summary"]["total"] == 7
    assert view["summary"]["analysed"] == 7
    photos = {p["id"]: p for p in view["photos"]}
    blurred = photos[ids[4]]
    assert blurred["culled"] and blurred["reasons"] == ["out_of_focus"]
    assert set(blurred["criteria"]) == {"sharpness", "motion", "exposure"}
    assert list(view["bursts"].values()) == [[ids[5], ids[6]]] or list(
        view["bursts"].values()
    ) == [[ids[6], ids[5]]]
    duplicates = [p for p in (photos[ids[5]], photos[ids[6]]) if p["culled"]]
    assert len(duplicates) == 1 and duplicates[0]["reasons"] == ["burst_duplicate"]
    assert view["summary"]["selected"] == 5
    assert view["settings"]["threshold"] == pytest.approx(0.37)
    assert view["confirmed"] is False


def test_models_that_are_not_there_are_explained_and_refused(client, small_project):
    project_id, _ = small_project
    view = client.get(f"/api/projects/{project_id}/culling").json()
    for feature in ("faces", "aesthetic"):
        assert view["availability"][feature]["available"] is False
        assert view["availability"][feature]["reason"]
    assert view["availability"]["aesthetic"]["notice"] == "ava_research_only"
    response = client.put(
        f"/api/projects/{project_id}/culling/settings", json={"criteria": {"faces": True}}
    )
    assert response.status_code == 409


def test_user_decisions_survive_recomputation_and_every_setting(client, catalog, small_project):
    """Test 10 of section 13, through the API and the catalogue."""
    project_id, ids = small_project
    client.get(f"/api/projects/{project_id}/culling")
    blurred, perfect = ids[4], ids[0]
    response = client.post(
        f"/api/projects/{project_id}/culling/decisions",
        json={"decisions": [{"photo_id": blurred, "decision": "keep"},
                            {"photo_id": perfect, "decision": "discard"}]},
    )
    assert response.status_code == 200

    def check():
        photos = {p["id"]: p for p in client.get(f"/api/projects/{project_id}/culling").json()[
            "photos"]}
        assert photos[blurred]["culled"] is False and photos[blurred]["decided_by"] == "user"
        assert photos[perfect]["culled"] is True and photos[perfect]["reasons"] == ["user"]

    check()
    for change in (
        {"aggressiveness": 1.0},
        {"aggressiveness": 0.0},
        {"mode": "target_percent", "target": 10},
        {"mode": "target_count", "target": 1},
        {"mode": "conservative"},
        {"criteria": {"sharpness": False, "burst": False}},
        {"criteria": {"sharpness": True, "burst": True}},
        {"weights": {"sharpness": 0.0, "exposure": 1.0}},
    ):
        assert client.put(
            f"/api/projects/{project_id}/culling/settings", json=change
        ).status_code == 200, change
        check()

    # A recomputation of the scores: what the culling job does to a row it
    # analyses again -- new numbers, grouping mark cleared, decisions untouched.
    with catalog() as session:
        for photo in session.scalars(select(Photo).where(Photo.project_id == project_id)):
            photo.sharpness = 0.01
            photo.burst_rank = None
            photo.culling_score = None
        session.commit()
    check()

    # Handing it back is explicit, and then the automatic selection decides.
    client.post(
        f"/api/projects/{project_id}/culling/decisions",
        json={"decisions": [{"photo_id": perfect, "decision": "auto"}]},
    )
    photos = {p["id"]: p for p in client.get(f"/api/projects/{project_id}/culling").json()[
        "photos"]}
    assert photos[perfect]["decided_by"] == "auto"


def test_changing_settings_never_queues_an_analysis(client, catalog, small_project):
    project_id, _ = small_project
    for change in ({"aggressiveness": 0.9}, {"mode": "target_percent", "target": 50},
                   {"criteria": {"motion": False}}):
        client.put(f"/api/projects/{project_id}/culling/settings", json=change)
    with catalog() as session:
        kinds = session.scalars(select(Job.kind)).all()
    # The first look groups the photos, and grouping queues the one panorama
    # search of section 25.2 (on thumbnails, not an analysis); nothing else.
    assert [kind for kind in kinds if kind is not JobKind.DETECT_MERGES] == []
    assert len(kinds) <= 1


def test_a_slider_move_answers_with_only_what_changed(client, small_project):
    project_id, ids = small_project
    client.get(f"/api/projects/{project_id}/culling")
    same = client.put(f"/api/projects/{project_id}/culling/settings",
                      json={"aggressiveness": 0.51}).json()
    assert same["decisions"] == []
    assert same["summary"]["selected"] == 5
    everything = client.put(f"/api/projects/{project_id}/culling/settings",
                            json={"weights": {"motion": 0.9}}).json()
    assert len(everything["decisions"]) == 7


def test_a_target_mode_needs_a_target(client, small_project):
    project_id, _ = small_project
    response = client.put(f"/api/projects/{project_id}/culling/settings",
                          json={"mode": "target_count"})
    assert response.status_code == 400


def test_confirming_develops_the_selection_and_only_the_selection(client, catalog, small_project):
    """Section 7.6: that button is the only way into the editing."""
    project_id, ids = small_project
    client.get(f"/api/projects/{project_id}/culling")
    body = client.post(f"/api/projects/{project_id}/culling/confirm").json()
    assert body["selected"] == 5 and body["queued"] == 5
    with catalog() as session:
        queued = {
            job.payload["photo_id"]
            for job in session.scalars(select(Job).where(Job.kind == JobKind.PROXY))
        }
        assert ids[4] not in queued
        assert session.get(Project, project_id).status.value == "analyzing"

    # A discarded photo brought back later is developed at once (section 7.5).
    client.post(f"/api/projects/{project_id}/culling/decisions",
                json={"decisions": [{"photo_id": ids[4], "decision": "keep"}]})
    with catalog() as session:
        payloads = [j.payload["photo_id"] for j in session.scalars(
            select(Job).where(Job.kind == JobKind.PROXY))]
        assert ids[4] in payloads
    assert client.get(f"/api/projects/{project_id}/culling").json()["confirmed"] is True


def test_the_culled_photos_are_hidden_from_the_viewer_on_request(client, small_project):
    project_id, ids = small_project
    client.get(f"/api/projects/{project_id}/culling")
    page = client.get(f"/api/projects/{project_id}/photos",
                      params={"include_culled": False}).json()
    assert ids[4] not in {p["id"] for p in page["items"]}
    assert all(p["has_thumb"] for p in page["items"])


@pytest.fixture
def big_project(catalog, tmp_path):
    """Two thousand analysed photos, written in bulk.

    Analysing two thousand previews would take the test a minute and prove
    nothing more: what is measured is the selection over stored features, so
    the rows reuse the features of a few real analyses, with scores spread
    across the whole range.
    """
    templates = [
        insert_analysed_features(natural_scene(9000 + i)) for i in range(4)
    ]
    with catalog() as session:
        project = make_project(session, tmp_path / "grande")
        rows = []
        for i in range(2000):
            features = templates[i % len(templates)]
            rows.append(
                {
                    "project_id": project.id,
                    "path": f"{project.source_dir}/DSC{i:05d}.ARW",
                    "filename": f"DSC{i:05d}.ARW",
                    "hash": f"q:big:{i}",
                    "kind": PhotoKind.RAW,
                    "status": PhotoStatus.CULLED,
                    "shot_at": T0 + timedelta(seconds=i * (1 if i % 7 else 30)),
                    "focal_length": 35.0,
                    "sharpness": (i * 37 % 101) / 100,
                    "motion_blur": (i * 13 % 67) / 200,
                    "exposure_score": 1.0 - (i * 7 % 53) / 120,
                    "culling_features": features,
                }
            )
        session.execute(insert(Photo), rows)
        session.commit()
        return project.id


def insert_analysed_features(image) -> dict:
    from ape.culling.features import analyse_preview
    from ape.raw.embedded import EmbeddedPreview

    features = analyse_preview(EmbeddedPreview(image, "embedded")).features
    features["camera"] = {"exposure_bias": 0.0, "release_mode": None, "shot_number": None,
                          "focus_point": None}
    return features


def test_a_slider_on_two_thousand_photos_answers_in_under_100_ms(client, big_project):
    """Phase 4's acceptance: no perceptible reanalysis, under 100 ms."""
    project_id = big_project
    view = client.get(f"/api/projects/{project_id}/culling").json()
    assert view["summary"]["total"] == 2000

    timings = {}
    for label, changes in (
        ("slider", [{"aggressiveness": a / 20} for a in range(21)]),
        ("mode", [{"mode": "target_percent", "target": 30}, {"mode": "conservative"}] * 5),
    ):
        durations = []
        for change in changes:
            started = time.perf_counter()
            response = client.put(f"/api/projects/{project_id}/culling/settings", json=change)
            durations.append(time.perf_counter() - started)
            assert response.status_code == 200
        timings[label] = statistics.median(durations) * 1000
    assert timings["slider"] < 100, f"{timings}"
    assert timings["mode"] < 100, f"{timings}"


# --- real files ------------------------------------------------------------


@pytest.mark.fixtures
def test_a_real_card_is_culled_end_to_end(client, catalog, raw_fixtures, tmp_path):
    """Import, analyse with a real worker, look, confirm -- on real ARW files."""
    from ape.config import get_settings
    from ape.jobs.worker import worker_loop

    card = tmp_path / "scheda"
    card.mkdir()
    for raw in raw_fixtures[:6]:
        shutil.copy2(raw, card / raw.name)

    created = client.post(
        "/api/projects", json={"name": "Vera", "source_dir": str(card), "import_now": False}
    ).json()
    client.post(f"/api/projects/{created['id']}/import", json={"culling": True})

    class _Never:
        @staticmethod
        def is_set() -> bool:
            return False

    worker_loop(get_settings().db_path, _Never(), idle_exit=True)

    view = client.get(f"/api/projects/{created['id']}/culling").json()
    assert view["summary"]["analysed"] == 6
    assert view["summary"]["selected"] == 6, [p["reasons"] for p in view["photos"]]
    for photo in view["photos"]:
        assert photo["status"] == "culled"
        assert photo["focus_point"] is not None
        grid = client.get(f"/api/photos/{photo['id']}/thumb")
        full = client.get(f"/api/photos/{photo['id']}/thumb", params={"size": "full"})
        assert grid.status_code == full.status_code == 200
        assert grid.headers["content-type"] == "image/jpeg"
        assert len(full.content) > len(grid.content)

    with catalog() as session:
        assert _jobs(session, JobKind.PROXY) == 0, "nessuna anteprima prima della conferma"
    assert client.post(f"/api/projects/{created['id']}/culling/confirm").json()["queued"] == 6


def test_the_thumbnail_of_an_unreadable_file_is_a_clear_error(client, catalog, tmp_path):
    card = tmp_path / "rotta"
    write_raw(card, "DSC00001.ARW", content=b"not a raw file" * 100)
    created = client.post(
        "/api/projects", json={"name": "Rotta", "source_dir": str(card)}
    ).json()
    photo = client.get(f"/api/projects/{created['id']}/photos").json()["items"][0]
    response = client.get(f"/api/photos/{photo['id']}/thumb")
    assert response.status_code == 422
    assert "illeggibile" in response.json()["detail"]


def test_an_ungrouped_photo_stays_marked_ungrouped(client, catalog, small_project):
    """Regression: the selection wrote a rank on a photo re-analysed since the
    grouping, which marked it grouped without grouping it."""
    project_id, ids = small_project
    client.get(f"/api/projects/{project_id}/culling")
    with catalog() as session:
        photo = session.get(Photo, ids[0])
        photo.burst_rank = None
        session.commit()
        from ape.culling import service

        project = session.get(Project, project_id)
        service.apply_selection(session, project)
        session.commit()
        assert session.get(Photo, ids[0]).burst_rank is None


def test_after_confirming_newly_selected_photos_are_developed(client, catalog, small_project):
    project_id, ids = small_project
    client.put(f"/api/projects/{project_id}/culling/settings",
               json={"mode": "target_count", "target": 2})
    client.post(f"/api/projects/{project_id}/culling/confirm")
    with catalog() as session:
        before = _jobs(session, JobKind.PROXY)
    client.put(f"/api/projects/{project_id}/culling/settings", json={"mode": "conservative"})
    with catalog() as session:
        assert _jobs(session, JobKind.PROXY) > before
        project = session.get(Project, project_id)
        from ape.db.models import ProjectStatus

        project.status = ProjectStatus.REVIEWING
        session.commit()
    client.post(f"/api/projects/{project_id}/culling/confirm")
    with catalog() as session:
        assert session.get(Project, project_id).status.value == "reviewing"

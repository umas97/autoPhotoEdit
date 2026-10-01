# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Section 20.3: the cache quota, "Svuota cache", and a cache that loses a file.

Least recently used first, the merges' intermediates last of all, the painted
masks and the models never. A proxy the cache lost is noticed where it is read
and queued again as a restore, which does not analyse the photo a second time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from ape.api.app import create_app
from conftest import available_raws
from conftest_catalog import write_raw


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _file(path: Path, size: int, used: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    os.utime(path, (used, used))
    return path


def _tree(xdg_home) -> dict[str, Path]:
    from ape.config import get_settings

    settings = get_settings()
    now = time.time()
    kb = 1024
    return {
        "old_proxy": _file(settings.proxy_dir / "ab" / "old.jpg", 100 * kb, now - 5000),
        "new_proxy": _file(settings.proxy_dir / "cd" / "new.jpg", 100 * kb, now - 10),
        "developed": _file(settings.cache_dir / "developed" / "1-2.jpg", 100 * kb, now - 3000),
        "preview": _file(settings.cache_dir / "previews" / "ef" / "p.jpg", 100 * kb, now - 1000),
        # The oldest of all, and still the last to go.
        "merge": _file(settings.intermediate_dir / "m.tif", 100 * kb, now - 99_999),
        "mask": _file(settings.masks_dir / "hand.png", 100 * kb, now - 99_999),
        "fill": _file(settings.retouch_dir / "f.patch", 100 * kb, now - 99_999),
        "model": _file(settings.models_dir / "x.onnx", 100 * kb, now - 99_999),
    }


def test_the_quota_evicts_the_least_recently_used_and_the_merges_last(xdg_home):
    from ape import cache

    files = _tree(xdg_home)
    kb = 1024
    usage = cache.usage()
    assert usage["cache_bytes"] == 500 * kb  # masks, fills and models are not cache
    assert usage["categories"]["masks"] == {"bytes": 100 * kb, "files": 1, "evictable": False}
    assert usage["categories"]["retouch"] == {"bytes": 100 * kb, "files": 1, "evictable": False}

    assert cache.enforce(limit=600 * kb) == {"removed": 0, "freed": 0}
    # Down to 90% of 400 KB = 360 KB: two files go, the two used longest ago
    # among the ordinary cache, although the intermediate is older still.
    result = cache.enforce(limit=400 * kb)
    assert result == {"removed": 2, "freed": 200 * kb}
    assert not files["old_proxy"].exists() and not files["developed"].exists()
    assert files["new_proxy"].exists() and files["preview"].exists() and files["merge"].exists()

    # Only when nothing else is left does a merge go; masks, fills and models never.
    cache.enforce(limit=1)
    assert not files["merge"].exists()
    assert files["mask"].exists() and files["model"].exists() and files["fill"].exists()


def test_emptying_the_cache_says_first_what_it_frees_and_what_it_keeps(
    client: TestClient, xdg_home
):
    files = _tree(xdg_home)
    storage = client.get("/api/storage").json()
    assert storage["limit_gb"] == pytest.approx(20.0)
    plan = storage["clear"]
    assert plan["frees_bytes"] == 500 * 1024 and plan["merges_to_rebuild"] == 1
    assert set(plan["keeps"]) == {"masks", "retouch", "models"}
    assert set(plan["categories"]) == {
        "proxies",
        "previews",
        "developed",
        "stages",
        "merges",
        "intermediates",
    }

    done = client.post("/api/storage/clear").json()
    assert done == {"removed": 5, "freed": 500 * 1024}
    assert files["mask"].exists() and files["model"].exists()
    assert client.get("/api/storage").json()["cache_bytes"] == 0


def test_the_quota_is_a_validated_setting(client: TestClient, catalog):
    from ape import cache

    assert client.put("/api/settings/cache_max_gb", json={"value": 0.5}).status_code == 400
    assert client.put("/api/settings/cache_max_gb", json={"value": "tanti"}).status_code == 400
    assert client.put("/api/settings/cache_max_gb", json={"value": 5}).status_code == 200
    with catalog() as session:
        assert cache.limit_bytes(session) == 5 * cache.GB
    assert client.get("/api/storage").json()["limit_gb"] == pytest.approx(5.0)
    # The program's own keys are not the interface's to write.
    assert client.put("/api/settings/schema_version", json={"value": 1}).status_code == 400
    assert client.put("/api/settings/artist", json={"value": "  Ada  "}).json() == {"artist": "Ada"}


def test_a_proxy_the_cache_lost_is_queued_again_as_a_restore(client: TestClient, catalog, tmp_path):
    from ape.db.enums import JobKind
    from ape.db.models import Job, Photo, Project

    folder = tmp_path / "scheda"
    raw = write_raw(folder, "DSC00001.ARW")
    with catalog() as session:
        project = Project(name="p", source_dir=str(folder))
        session.add(project)
        session.flush()
        photo = Photo(
            project_id=project.id,
            path=str(raw),
            filename=raw.name,
            proxy_path=str(tmp_path / "cache" / "gone.jpg"),
        )
        session.add(photo)
        session.commit()
        project_id, photo_id = project.id, photo.id

    listed = client.get(f"/api/projects/{project_id}/photos").json()["items"]
    assert listed[0]["has_proxy"] is False  # the grid shows the embedded preview meanwhile
    assert client.get(f"/api/photos/{photo_id}/proxy").status_code == 404
    with catalog() as session:
        (job,) = session.scalars(select(Job)).all()  # asked twice, queued once
        assert job.kind is JobKind.PROXY and job.payload == {"photo_id": photo_id, "restore": True}


@pytest.mark.fixtures
def test_a_restored_proxy_does_not_analyse_the_photo_again(catalog, tmp_path):
    from ape.db.enums import JobKind, JobState
    from ape.db.models import Job, Photo, Project
    from ape.jobs import queue as jobq
    from ape.jobs.worker import run_job

    raws = available_raws()
    if not raws:
        pytest.skip("servono i file ARW di tests/fixtures/")
    with catalog() as session:
        project = Project(name="p", source_dir=str(raws[0].parent))
        session.add(project)
        session.flush()
        photo = Photo(
            project_id=project.id,
            path=str(raws[0]),
            filename=raws[0].name,
            analysis={"version": 1, "straighten": {"rotation_deg": 0.0}},
        )
        session.add(photo)
        session.commit()
        photo_id = photo.id

    for payload in ({"photo_id": photo_id, "restore": True}, {"photo_id": photo_id}):
        record = jobq.JobRecord(
            id=0, kind=JobKind.PROXY, payload=payload, project_id=None, attempts=1
        )
        run_job(record, lambda _fraction: None)
        with catalog() as session:
            analyses = session.scalars(
                select(Job).where(Job.kind == JobKind.ANALYZE, Job.state == JobState.QUEUED)
            ).all()
            assert Path(session.get(Photo, photo_id).proxy_path).is_file()
            if payload.get("restore"):
                assert analyses == [], "un proxy ripristinato ha rifatto l'analisi"
            else:
                assert len(analyses) == 1

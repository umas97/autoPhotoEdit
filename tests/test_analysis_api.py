# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 5 end to end: import, proxy, analysis, scenes, lenses, crop proposals.

On the user's real files, in a temporary catalogue, with the jobs run by hand
one at a time -- the same arrangement as ``test_api.py``. The lensfun data and
the CLIP model are the ones this machine has, copied into the temporary home
when present, so that what is checked is what the user would get.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from ape.api.app import create_app
from conftest import available_raws
from conftest_catalog import file_state

#: The user's real data home, read before ``xdg_home`` replaces it.
_REAL_DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "autophotoedit"


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _borrow(xdg_home: Path, relative: str) -> bool:
    """Link a file or folder of the real data home into the temporary one."""
    source = _REAL_DATA / relative
    if not source.exists():
        return False
    target = xdg_home / "data" / "autophotoedit" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        os.symlink(source, target)
    return True


def _run_all(catalog) -> int:
    from ape.jobs.queue import claim_job, complete_job, fail_job
    from ape.jobs.worker import run_job

    ran = 0
    while True:
        with catalog() as session:
            record = claim_job(session)
            session.commit()
        if record is None:
            return ran
        try:
            run_job(record, lambda _fraction: None)
        except Exception as exc:  # noqa: BLE001 - recorded like the worker does
            with catalog() as session:
                fail_job(session, record.id, str(exc))
                session.commit()
            raise
        with catalog() as session:
            complete_job(session, record.id)
            session.commit()
        ran += 1


def _project(client: TestClient, folder: Path) -> dict:
    response = client.post("/api/projects", json={"name": "fase5", "source_dir": str(folder)})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def card(tmp_path: Path) -> Path:
    """Six of the user's frames: the lake twice, the stumps twice, two portraits."""
    raws = {p.stem: p for p in available_raws()}
    wanted = ["DSC05634", "DSC05635", "DSC05637", "DSC05638", "DSC05630", "DSC05631"]
    if not all(name in raws for name in wanted):
        pytest.skip("servono i file ARW di tests/fixtures/")
    folder = tmp_path / "scheda"
    folder.mkdir()
    for name in wanted:
        shutil.copy2(raws[name], folder)
    return folder


@pytest.mark.fixtures
def test_import_to_scenes_end_to_end(client: TestClient, catalog, card: Path, xdg_home: Path):
    has_lensfun = _borrow(xdg_home, "lensfun/version_1")
    has_model = _borrow(xdg_home, "models/clip-vit-b32-visual-int8.onnx")
    before = file_state(card)

    project = _project(client, card)
    ran = _run_all(catalog)
    assert ran == 12  # six proxies, and the six analyses they chained

    photos = client.get(f"/api/projects/{project['id']}/photos").json()["items"]
    assert all(photo["status"] == "analyzed" for photo in photos)

    detail = client.get(f"/api/photos/{photos[0]['id']}").json()
    analysis = detail["analysis"]
    assert analysis["straighten"]["outcome"] in {"rotated", "level", "no_lines", "contradictory"}
    assert analysis["as_shot"]["temperature_k"] > 2000
    if has_lensfun:
        assert analysis["lens"]["maker"] == "Tamron"
    # The first version is the neutral development, straightened.
    assert detail["params"]["geometry"]["rotation_deg"] == pytest.approx(
        analysis["straighten"]["rotation_deg"], abs=1e-3
    )
    assert abs(detail["params"]["geometry"]["rotation_deg"]) <= 0.5

    scenes = client.get(f"/api/projects/{project['id']}/scenes").json()
    assert scenes["clustered"] == 6 and scenes["pending"] == 0
    assert scenes["basis"] == ("embedding" if has_model else "features")
    by_photo = {p: c["cluster"] for c in scenes["clusters"] for p in c["photos"]}
    names = {photo["id"]: photo["filename"][:8] for photo in photos}
    cluster_of = {names[p]: c for p, c in by_photo.items()}
    # The lake twice and the stumps twice are the same scene; the two are not.
    assert cluster_of["DSC05634"] == cluster_of["DSC05635"]
    assert cluster_of["DSC05637"] == cluster_of["DSC05638"]
    assert cluster_of["DSC05634"] != cluster_of["DSC05637"]
    for group in scenes["clusters"]:
        assert group["representative"] == group["photos"][0]

    lenses = client.get(f"/api/projects/{project['id']}/lenses").json()
    assert lenses["lenses"][0]["lens"] == "E 28-75mm F2.8 A063"
    assert lenses["lenses"][0]["photos"] == 6

    # Section 2: nothing of the above touched the card.
    assert file_state(card) == before


@pytest.mark.fixtures
def test_a_lens_association_redoes_the_proxies_and_the_analysis(
    client: TestClient, catalog, card: Path
):
    """Without updated lensfun data the A063 has no profile: the user picks one."""
    project = _project(client, card)
    _run_all(catalog)
    photos = client.get(f"/api/projects/{project['id']}/photos").json()["items"]
    detail = client.get(f"/api/photos/{photos[0]['id']}").json()
    assert detail["analysis"]["lens"] is None

    candidates = client.get("/api/lenses/search", params={"q": "tamron 28-75"}).json()
    chosen = next(c for c in candidates if "28-75" in c["model"])
    body = {"lens": "E 28-75mm F2.8 A063", "maker": chosen["maker"], "model": chosen["model"]}
    response = client.put("/api/lenses/override", json=body)
    assert response.status_code == 200
    assert response.json()["queued"] == 6
    assert _run_all(catalog) == 12

    detail = client.get(f"/api/photos/{photos[0]['id']}").json()
    assert detail["analysis"]["lens"] == {
        "maker": chosen["maker"], "model": chosen["model"], "source": "override"
    }
    lenses = client.get(f"/api/projects/{project['id']}/lenses").json()["lenses"]
    assert lenses[0]["override"]["model"] == chosen["model"]


def _seed_proposals(catalog, project_id: int, count: int) -> list[int]:
    """Pending proposals on the first ``count`` photos, whatever the analysis found."""
    from ape.db.models import CropDecision, CropProposal, Photo

    with catalog() as session:
        photos = session.query(Photo).filter(Photo.project_id == project_id).all()[:count]
        ids = []
        for photo in photos:
            session.query(CropProposal).filter(CropProposal.photo_id == photo.id).delete()
            proposal = CropProposal(
                photo_id=photo.id,
                rect={"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
                aspect="original",
                score=1.0,
                decision=CropDecision.PENDING,
            )
            session.add(proposal)
            session.flush()
            ids.append(proposal.id)
        session.commit()
    return ids


@pytest.mark.fixtures
def test_crop_proposals_are_applied_only_on_request_and_pause_after_two_rejections(
    client: TestClient, catalog, card: Path
):
    project = _project(client, card)
    _run_all(catalog)
    first, second, third = _seed_proposals(catalog, project["id"], 3)

    photos = client.get(f"/api/projects/{project['id']}/photos").json()["items"]
    detail = client.get(f"/api/photos/{photos[0]['id']}").json()
    # Proposed, not applied (section 6.4).
    assert detail["crop_proposal"]["id"] == first
    assert detail["params"]["geometry"]["crop"] is None

    applied = client.post(f"/api/crop-proposals/{first}/apply").json()
    assert applied["params"]["geometry"]["crop"]["width"] == pytest.approx(0.8)
    assert applied["versions"][0]["source"] == "user_edited"
    assert len(applied["versions"]) == 2  # the analysis' version is still there
    assert client.post(f"/api/crop-proposals/{first}/apply").status_code == 409

    client.post(f"/api/crop-proposals/{second}/reject")
    after = client.post(f"/api/crop-proposals/{third}/reject").json()
    assert after["crop_proposals_paused"] is True

    _seed_proposals(catalog, project["id"], 1)
    detail = client.get(f"/api/photos/{photos[0]['id']}").json()
    assert detail["crop_proposal"] is None  # paused: not shown
    client.post(f"/api/projects/{project['id']}/crop-proposals/resume")
    assert client.get(f"/api/photos/{photos[0]['id']}").json()["crop_proposal"] is not None


def test_models_and_lensfun_state_are_reported(client: TestClient):
    models = {m["feature"]: m for m in client.get("/api/models").json()}
    assert models["embedding"]["downloadable"] is True
    assert models["embedding"]["available"] is False  # nothing in a fresh home
    assert models["aesthetic"]["notice"] == "ava_research_only"
    assert models["faces"]["downloadable"] is False
    assert client.post("/api/models/faces/download").status_code == 409
    lensfun = client.get("/api/lensfun").json()
    assert lensfun["source"] == "bundled" and lensfun["lenses"] > 1000


def test_a_catalogue_of_schema_2_is_migrated(xdg_home):
    from sqlalchemy import inspect

    from ape.db.models import Project
    from ape.db.session import SCHEMA_VERSION, get_engine, init_db, read_setting, session_scope

    engine = get_engine()
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE photo DROP COLUMN analysis"))
        connection.execute(text("ALTER TABLE photo DROP COLUMN cluster_rank"))
        connection.execute(text("ALTER TABLE crop_proposal DROP COLUMN decided_at"))
        connection.execute(text("ALTER TABLE project DROP COLUMN crop_proposals_paused"))
        connection.execute(text("UPDATE setting SET value = '2' WHERE key = 'schema_version'"))
    init_db(engine)
    inspector = inspect(engine)
    assert {"analysis", "cluster_rank"} <= {c["name"] for c in inspector.get_columns("photo")}
    assert "decided_at" in {c["name"] for c in inspector.get_columns("crop_proposal")}
    assert "crop_proposals_paused" in {c["name"] for c in inspector.get_columns("project")}
    with session_scope() as session:
        assert int(read_setting(session, "schema_version")) == SCHEMA_VERSION
        session.add(Project(name="nuovo", source_dir="/x"))
        session.flush()
        assert session.query(Project).one().crop_proposals_paused is False


def test_pending_proposals_in_another_ratio_are_withdrawn_and_proposed_again(catalog, project):
    """Older builds proposed 4:5, 16:9 and 1:1; only the frame's own ratio is proposed now.

    A pending one in another ratio goes, and the photo's crop alone is queued
    again. A decided one is the user's and stays, and so does a panorama's
    clean rectangle, whose ratio is the stitching's.
    """
    from sqlalchemy import select

    from ape.analysis.service import retire_stale_proposals
    from ape.db.models import CropDecision, CropProposal, Job, JobKind, Photo

    project_id, _ = project
    rect = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
    with catalog() as session:
        photos = session.scalars(
            select(Photo).where(Photo.project_id == project_id).order_by(Photo.id)
        ).all()
        for photo in photos:
            photo.proxy_path = f"/tmp/proxy-{photo.id}.jpg"
        stale, decided, borders = photos[0], photos[1], photos[2]
        session.add_all([
            CropProposal(photo_id=stale.id, rect=rect, aspect="16:9",
                         decision=CropDecision.PENDING),
            CropProposal(photo_id=decided.id, rect=rect, aspect="4:5",
                         decision=CropDecision.APPLIED),
            CropProposal(photo_id=borders.id, rect=rect, aspect="borders",
                         decision=CropDecision.PENDING),
        ])
        session.commit()

        assert retire_stale_proposals(session) == 1
        session.commit()
        left = session.scalars(select(CropProposal).order_by(CropProposal.photo_id)).all()
        assert [(p.photo_id, p.aspect) for p in left] == [(decided.id, "4:5"), (borders.id, "borders")]
        jobs = session.scalars(select(Job).where(Job.kind == JobKind.ANALYZE)).all()
        assert [job.payload for job in jobs] == [{"photo_id": stale.id, "only": "crop"}]

        # Idempotent: the second start finds nothing.
        assert retire_stale_proposals(session) == 0

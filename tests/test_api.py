# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The HTTP surface of phase 2, driven in process.

These tests run the real application against a real SQLite catalogue in a
temporary XDG home, with the worker pool switched off: a test that raced sixteen
background processes would fail for reasons that have nothing to do with what it
is checking. The jobs are run by hand, one at a time, where a test needs them
run at all.

What is being checked here is the contract the frontend of phase 3 will be
written against, plus the two invariants the API is in a position to break:
exactly one current version per photo (section 23), and a source folder that is
never written to (section 2).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ape.api.app import create_app
from conftest_catalog import file_state, write_raw


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _create(client: TestClient, folder: Path, name: str = "Vacanza") -> dict:
    response = client.post(
        "/api/projects", json={"name": name, "source_dir": str(folder)}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_reports_the_catalogue_in_use(client: TestClient):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"].endswith("catalog.db")
    assert body["workers"] == 0


def test_creating_a_project_imports_the_folder(client: TestClient, card: Path):
    body = _create(client, card)
    assert body["photo_count"] == 6
    assert body["source_missing"] is False
    # One proxy job per photo, queued and waiting for a worker.
    summary = client.get("/api/jobs/summary", params={"project_id": body["id"]}).json()
    assert summary["counts"]["queued"] == 6


def test_a_project_on_a_folder_that_does_not_exist_is_refused(client: TestClient, tmp_path: Path):
    response = client.post(
        "/api/projects", json={"name": "fantasma", "source_dir": str(tmp_path / "nulla")}
    )
    assert response.status_code == 400
    assert "non esiste" in response.json()["detail"]


def test_importing_twice_over_http_adds_nothing(client: TestClient, card: Path):
    project = _create(client, card)
    body = client.post(f"/api/projects/{project['id']}/import", json={}).json()
    assert body["imported"] == 0
    assert body["already_present"] == 6
    assert "già presenti" in body["summary"]
    assert client.get(f"/api/projects/{project['id']}").json()["photo_count"] == 6


def test_the_import_never_writes_to_the_source(client: TestClient, card: Path):
    before = file_state(card)
    project = _create(client, card)
    client.post(f"/api/projects/{project['id']}/import", json={})
    assert file_state(card) == before


def test_scanning_declares_what_it_will_not_import(client: TestClient, card: Path):
    project = _create(client, card)
    body = client.get(f"/api/projects/{project['id']}/scan").json()
    assert body["raws"] == 6
    assert body["subdirectories"] == ["sottocartella"]
    assert body["rejected_formats"] == {".cr2": 1}


def test_photos_are_listed_with_paging(client: TestClient, card: Path):
    project = _create(client, card)
    page = client.get(
        f"/api/projects/{project['id']}/photos", params={"limit": 4}
    ).json()
    assert page["total"] == 6
    assert len(page["items"]) == 4
    assert page["items"][0]["has_proxy"] is False

    rest = client.get(
        f"/api/projects/{project['id']}/photos", params={"limit": 4, "offset": 4}
    ).json()
    assert len(rest["items"]) == 2


def test_saving_parameters_builds_a_history_with_one_current_version(
    client: TestClient, card: Path
):
    project = _create(client, card)
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]

    first = client.put(
        f"/api/photos/{photo['id']}/params",
        json={"params": {"exposure": {"ev": 0.5}}, "source": "user_edited"},
    )
    assert first.status_code == 200, first.text
    assert first.json()["params"]["exposure"]["ev"] == 0.5

    second = client.put(
        f"/api/photos/{photo['id']}/params",
        json={"params": {"exposure": {"ev": -0.25}}, "source": "user_edited"},
    ).json()
    assert second["params"]["exposure"]["ev"] == -0.25
    assert len(second["versions"]) == 2
    assert sum(1 for v in second["versions"] if v["is_current"]) == 1
    # The history is a tree: the new version knows where it came from.
    current = next(v for v in second["versions"] if v["is_current"])
    assert current["parent_version_id"] is not None


def test_restoring_an_old_version_adds_one_rather_than_rewinding(
    client: TestClient, card: Path
):
    project = _create(client, card)
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]
    first = client.put(
        f"/api/photos/{photo['id']}/params", json={"params": {"exposure": {"ev": 1.0}}}
    ).json()
    original_version = first["current_version_id"]
    client.put(f"/api/photos/{photo['id']}/params", json={"params": {"exposure": {"ev": -1.0}}})

    restored = client.post(
        f"/api/photos/{photo['id']}/versions/{original_version}/restore"
    ).json()
    assert restored["params"]["exposure"]["ev"] == 1.0
    assert len(restored["versions"]) == 3, "il ripristino deve aggiungere, non cancellare"
    assert sum(1 for v in restored["versions"] if v["is_current"]) == 1


def test_invalid_parameters_are_refused_with_the_field_named(client: TestClient, card: Path):
    project = _create(client, card)
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]
    response = client.put(
        f"/api/photos/{photo['id']}/params", json={"params": {"exposure": {"ev": 99.0}}}
    )
    assert response.status_code == 400
    assert "ev" in response.json()["detail"]


def test_a_photo_without_a_proxy_says_so(client: TestClient, card: Path):
    project = _create(client, card)
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]
    response = client.get(f"/api/photos/{photo['id']}/proxy")
    assert response.status_code == 404
    assert "non ancora generata" in response.json()["detail"]


def test_a_missing_file_is_reported_and_the_row_stays(client: TestClient, card: Path):
    project = _create(client, card)
    photos = client.get(f"/api/projects/{project['id']}/photos").json()["items"]
    (card / photos[0]["filename"]).unlink()

    client.post(f"/api/projects/{project['id']}/import", json={})
    after = client.get(f"/api/projects/{project['id']}/photos").json()
    assert after["total"] == 6
    assert sum(1 for p in after["items"] if p["missing"]) == 1
    assert client.get(f"/api/projects/{project['id']}").json()["missing_count"] == 1


def test_remapping_a_moved_folder_over_http(client: TestClient, card: Path, tmp_path: Path):
    project = _create(client, card)
    moved = tmp_path / "altro-disco"
    card.rename(moved)

    body = client.post(f"/api/projects/{project['id']}/remap", json={"folder": str(moved)}).json()
    assert body["reattached"] == 6
    assert body["still_missing"] == 0
    assert client.get(f"/api/projects/{project['id']}").json()["source_dir"] == str(moved)


def test_a_queued_job_can_be_cancelled_and_a_running_one_cannot(
    client: TestClient, card: Path, catalog
):
    from ape.jobs.queue import claim_job

    project = _create(client, card)
    jobs = client.get("/api/jobs", params={"project_id": project["id"]}).json()
    assert len(jobs) == 6

    cancelled = client.post(f"/api/jobs/{jobs[0]['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"

    with catalog() as session:
        record = claim_job(session)
    assert record is not None
    refused = client.post(f"/api/jobs/{record.id}/cancel")
    assert refused.status_code == 409


def test_the_project_patch_only_changes_what_it_names(client: TestClient, card: Path):
    project = _create(client, card)
    body = client.patch(
        f"/api/projects/{project['id']}", json={"export_strip_gps": True}
    ).json()
    assert body["export_strip_gps"] is True
    assert body["name"] == project["name"]
    assert body["export_template"] == project["export_template"]


def test_deleting_a_project_leaves_the_photographs_alone(client: TestClient, card: Path):
    before = file_state(card)
    project = _create(client, card)
    assert client.delete(f"/api/projects/{project['id']}").status_code == 204
    assert client.get(f"/api/projects/{project['id']}").status_code == 404
    assert file_state(card) == before
    assert len(list(card.glob("*.ARW"))) == 6


def test_the_progress_socket_sends_the_state_at_once(client: TestClient, card: Path):
    project = _create(client, card)
    with client.websocket_connect(f"ws://127.0.0.1/ws?project_id={project['id']}") as socket:
        message = socket.receive_json()
    assert message["type"] == "progress"
    assert message["counts"]["queued"] == 6
    assert message["photos"] == 6
    assert message["photos_with_proxy"] == 0


def test_two_projects_on_the_same_folder_keep_separate_catalogues(
    client: TestClient, card: Path
):
    """The uniqueness of a content hash is per project, not global."""
    first = _create(client, card, name="primo")
    second = _create(client, card, name="secondo")
    assert first["photo_count"] == 6
    assert second["photo_count"] == 6


def test_an_empty_folder_imports_nothing_without_complaining(
    client: TestClient, tmp_path: Path
):
    empty = tmp_path / "vuota"
    empty.mkdir()
    body = _create(client, empty)
    assert body["photo_count"] == 0
    assert client.get(f"/api/projects/{body['id']}/scan").json()["raws"] == 0


@pytest.mark.fixtures
def test_a_real_raw_becomes_a_proxy_and_a_preview(client: TestClient, tmp_path: Path, catalog):
    """The whole phase, end to end, on a file from the user's camera."""
    import shutil

    from ape.jobs.queue import claim_job
    from ape.jobs.worker import run_job
    from conftest import available_raws

    folder = tmp_path / "scheda"
    folder.mkdir()
    shutil.copy2(available_raws()[0], folder)

    project = _create(client, folder, name="reale")
    with catalog() as session:
        record = claim_job(session)
    assert record is not None
    run_job(record, lambda _fraction: None)
    with catalog() as session:
        from ape.jobs.queue import complete_job

        complete_job(session, record.id)
        session.commit()

    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]
    assert photo["has_proxy"] is True
    assert photo["camera"] is not None
    assert photo["width"] and photo["width"] > 2000

    proxy = client.get(f"/api/photos/{photo['id']}/proxy")
    assert proxy.status_code == 200
    assert proxy.headers["content-type"] == "image/jpeg"
    assert "immutable" in proxy.headers["cache-control"]

    preview = client.post(
        f"/api/photos/{photo['id']}/preview",
        json={"exposure": {"ev": 0.3}, "color": {"saturation": 0.1}},
    )
    assert preview.status_code == 200
    assert len(preview.content) > 10_000


def test_a_photo_with_no_file_cannot_be_previewed(client: TestClient, card: Path, catalog):
    """A merged photo (phase 11) has no RAW; the endpoint must say so, not crash."""
    from ape.db.models import Photo, PhotoKind

    project = _create(client, card)
    with catalog() as session:
        session.add(
            Photo(
                project_id=project["id"],
                path=None,
                filename="fusione.exr",
                kind=PhotoKind.MERGED,
            )
        )
        session.commit()
        merged = session.query(Photo).filter(Photo.kind == PhotoKind.MERGED).one()
        merged_id = merged.id

    response = client.post(f"/api/photos/{merged_id}/preview", json={})
    assert response.status_code == 409
    assert "sorgente" in response.json()["detail"]


def test_an_unknown_photo_is_a_clean_404(client: TestClient):
    assert client.get("/api/photos/9999").status_code == 404
    assert client.post("/api/photos/9999/preview", json={}).status_code == 404


def test_a_second_folder_can_be_added_without_marking_the_first_missing(
    client: TestClient, card: Path, tmp_path: Path
):
    project = _create(client, card)
    other = tmp_path / "seconda-scheda"
    write_raw(other, "DSC01000.ARW")

    body = client.post(
        f"/api/projects/{project['id']}/import",
        json={"folder": str(other), "mark_missing": False},
    ).json()
    assert body["imported"] == 1
    assert body["marked_missing"] == 0
    assert client.get(f"/api/projects/{project['id']}").json()["photo_count"] == 7


@pytest.mark.fixtures
def test_the_preview_the_viewer_shows_is_the_image_an_export_writes(
    client: TestClient, tmp_path: Path
):
    """Phase 3's second acceptance criterion, measured (section 14).

    The viewer never draws a pixel of its own: it asks the server to develop the
    photograph and shows the answer. So "the preview matches the export" is the
    question of whether the interactive path -- which decodes the RAW at the size
    being displayed, to make section 10's 150 ms -- lands where the export path
    does, which decodes the whole sensor and resizes inside the pipeline.

    The tolerance is the one test 1 uses for the same property, and the measured
    distance is far below it: the shortcut that makes the slider fast does not
    change what the photograph looks like.
    """
    import shutil

    import cv2
    import numpy as np

    from ape.export.image import ExportFormat, encode_image
    from ape.pipeline.colorspace import OutputSpace
    from ape.pipeline.params import EditParams
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.decode import decode_linear
    from conftest import available_raws

    source = available_raws()[0]
    folder = tmp_path / "scheda"
    folder.mkdir()
    shutil.copy2(source, folder)
    project = _create(client, folder, name="anteprima")
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]

    params = {
        "exposure": {"ev": 0.4},
        "tone": {"contrast": 1.35},
        "color": {"saturation": 0.15, "vibrance": 0.2},
        "local_contrast": {"clarity": 0.3},
    }
    edge = 1024

    response = client.post(
        f"/api/photos/{photo['id']}/preview", params={"long_edge": edge}, json=params
    )
    assert response.status_code == 200

    # The export path, from the full-resolution sensor, at the same output size.
    exported = encode_image(
        render(
            decode_linear(folder / source.name),
            EditParams.model_validate(params),
            RenderOptions(output_space=OutputSpace.SRGB, long_edge=edge),
        ),
        ExportFormat.JPEG,
        quality=95,
    )

    def as_lab(data: bytes) -> np.ndarray:
        import colour

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))

    preview_lab = as_lab(response.content)
    export_lab = as_lab(exported)
    assert preview_lab.shape == export_lab.shape

    from ape.pipeline.colorspace import delta_e_2000

    # The border is skipped: an unsharp mask has nothing to work with there, and
    # the two paths pad it differently.
    inner = (slice(8, -8), slice(8, -8))
    difference = delta_e_2000(preview_lab[inner], export_lab[inner])
    mean = float(np.mean(difference))
    print(f"anteprima vs export: dE2000 medio {mean:.3f}, massimo {float(np.max(difference)):.3f}")
    assert mean < 1.5, f"dE2000 medio {mean:.3f}"

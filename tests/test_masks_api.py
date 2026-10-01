# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The masks over HTTP: a painted raster goes up, comes back, and renders.

Plus the one guarantee a new writer of files owes section 2: the rasters land
in the masks folder, and the source folder of the photo they are painted on is
byte-for-byte what it was.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ape.api.app import create_app
from conftest_catalog import file_state


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _png(width: int = 64, height: int = 40) -> bytes:
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[:, : width // 2, 3] = 255
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def test_a_raster_goes_up_and_comes_back(client: TestClient):
    response = client.post(
        "/api/masks/rasters", content=_png(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 201, response.text
    name = response.json()["name"]
    back = client.get(f"/api/masks/rasters/{name}")
    assert back.status_code == 200
    assert back.headers["content-type"] == "image/png"
    assert "immutable" in back.headers["cache-control"]
    stored = cv2.imdecode(np.frombuffer(back.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert stored.ndim == 2, "un canale solo, 8 bit"


def test_what_is_not_a_mask_is_refused(client: TestClient):
    assert client.post("/api/masks/rasters", content=b"ciao").status_code == 400
    assert client.get("/api/masks/rasters/" + "a" * 64).status_code == 404
    assert client.get("/api/masks/rasters/..%2F..%2Fcatalog").status_code == 404


def test_a_painted_mask_previews_on_a_real_raw_and_leaves_the_source_alone(
    client: TestClient, tmp_path: Path, catalog
):
    from ape.jobs.queue import claim_job, complete_job
    from ape.jobs.worker import run_job
    from conftest import available_raws

    folder = tmp_path / "scheda"
    folder.mkdir()
    shutil.copy2(available_raws()[0], folder)
    before = file_state(folder)

    project = client.post("/api/projects", json={"name": "m", "source_dir": str(folder)}).json()
    with catalog() as session:
        record = claim_job(session)
    run_job(record, lambda _fraction: None)
    with catalog() as session:
        complete_job(session, record.id)
        session.commit()
    photo = client.get(f"/api/projects/{project['id']}/photos").json()["items"][0]

    name = client.post("/api/masks/rasters", content=_png()).json()["name"]
    params = {
        "masks": [
            {"kind": "brush", "name": "Pennello", "definition": {"raster": name},
             "exposure": {"ev": 1.0}},
            {"kind": "radial", "definition": {"cx": 0.7, "cy": 0.5, "rx": 0.2, "ry": 0.2},
             "color": {"saturation": -1.0}},
        ]
    }
    preview = client.post(f"/api/photos/{photo['id']}/preview", json=params)
    assert preview.status_code == 200, preview.text
    saved = client.put(f"/api/photos/{photo['id']}/params", json={"params": params})
    assert saved.status_code == 200, saved.text
    assert saved.json()["params"]["masks"][0]["definition"]["raster"] == name

    # What mask 1 selects, framed as the preview: the ellipse right of centre.
    selection = client.post(f"/api/photos/{photo['id']}/masks/1/selection", json=params)
    assert selection.status_code == 200, selection.text
    overlay = cv2.imdecode(np.frombuffer(selection.content, np.uint8), cv2.IMREAD_UNCHANGED)
    shown = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_COLOR)
    assert overlay.shape[:2] == shown.shape[:2] and overlay.shape[2] == 4
    height, width = overlay.shape[:2]
    assert overlay[height // 2, int(0.7 * width), 3] == 255
    assert overlay[height // 2, int(0.2 * width), 3] == 0
    # Even with every adjustment at zero, and never past the last mask.
    bare = {"masks": [{"kind": "radial", "definition": {"cx": 0.7}}]}
    assert client.post(f"/api/photos/{photo['id']}/masks/0/selection", json=bare).status_code == 200
    assert client.post(f"/api/photos/{photo['id']}/masks/1/selection", json=bare).status_code == 400

    missing = {"masks": [{"kind": "brush", "definition": {"raster": "b" * 64},
                          "exposure": {"ev": 1.0}}]}
    refused = client.post(f"/api/photos/{photo['id']}/preview", json=missing)
    assert refused.status_code == 400
    assert "maschera disegnata" in refused.json()["detail"]
    assert file_state(folder) == before

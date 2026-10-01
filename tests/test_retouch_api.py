# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removals over HTTP, on real RAWs (docs/SPEC_rimozione.md section 8, 11 and 12).

A spot gets its source from the server; an area comes from a mask; an eraser
released queues its fill, the worker makes it, and the photo's history does
not grow for it; names and rasters that are not ours are refused; the spots
copied from a landscape photo land on the same photosites of a portrait one;
a restored version finds its fills; and the source folder stays as it was.
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
from conftest import FIXTURES
from conftest_catalog import file_state

pytestmark = pytest.mark.fixtures

_LANDSCAPE, _PORTRAIT = "DSC05618.ARW", "DSC05620.ARW"


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def _run_jobs(catalog, kinds=None) -> int:
    from ape.jobs.queue import claim_job, complete_job
    from ape.jobs.worker import run_job

    count = 0
    while True:
        with catalog() as session:
            record = claim_job(session, kinds)
            session.commit()
        if record is None:
            return count
        run_job(record, lambda _fraction: None)
        with catalog() as session:
            complete_job(session, record.id)
            session.commit()
        count += 1


@pytest.fixture
def project(client, catalog, tmp_path: Path):
    for name in (_LANDSCAPE, _PORTRAIT):
        if not (FIXTURES / name).is_file():
            pytest.skip(f"{name} non è fra le fixture")
    folder = tmp_path / "scheda"
    folder.mkdir()
    for name in (_LANDSCAPE, _PORTRAIT):
        shutil.copy2(FIXTURES / name, folder)
    before = file_state(folder)
    created = client.post("/api/projects", json={"name": "r", "source_dir": str(folder)}).json()
    _run_jobs(catalog)
    photos = client.get(f"/api/projects/{created['id']}/photos").json()["items"]
    by_name = {photo["filename"]: photo for photo in photos}
    yield {"id": created["id"], "folder": folder, "before": before,
           "landscape": by_name[_LANDSCAPE]["id"], "portrait": by_name[_PORTRAIT]["id"]}
    assert file_state(folder) == before, "la cartella sorgente è cambiata"


def _raster(width: int = 2048, height: int = 1366) -> bytes:
    canvas = np.zeros((height, width), np.uint8)
    cv2.circle(canvas, (int(0.5 * width), int(0.6 * height)), 40, 255, -1)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def _versions(catalog, photo_id: int) -> int:
    from ape.db.models import EditVersion

    with catalog() as session:
        return session.query(EditVersion).filter_by(photo_id=photo_id).count()


def test_a_spot_gets_its_source_from_the_server(client, project):
    spot = {"cx": 0.4, "cy": 0.5, "radius": 0.01}
    answer = client.post(
        f"/api/photos/{project['landscape']}/retouch/source",
        json={"params": {}, "index": 0, **spot},
    )
    assert answer.status_code == 200, answer.text
    source = answer.json()
    assert 0.0 <= source["sx"] <= 1.0 and 0.0 <= source["sy"] <= 1.0
    assert (source["sx"], source["sy"]) != (spot["cx"], spot["cy"])
    again = client.post(
        f"/api/photos/{project['landscape']}/retouch/source",
        json={"params": {}, "index": 0, **spot, "avoid": [[source["sx"], source["sy"]]]},
    ).json()
    assert (again["sx"], again["sy"]) != (source["sx"], source["sy"])
    heal = {"kind": "heal", "id": "h1", **spot, "sx": source["sx"], "sy": source["sy"]}
    shown = client.post(f"/api/photos/{project['landscape']}/preview", json={"retouch": [heal]})
    assert shown.status_code == 200


def test_an_area_is_taken_from_a_mask_once(client, project):
    photo = project["landscape"]
    params = {"masks": [{"kind": "radial", "definition": {"cx": 0.5, "cy": 0.5, "rx": 0.1,
                                                          "ry": 0.1, "feather": 0.0}}]}
    made = client.post(f"/api/photos/{photo}/retouch/area", json={"params": params,
                                                                   "mask_index": 0})
    assert made.status_code == 201, made.text
    name = made.json()["name"]
    raster = cv2.imdecode(
        np.frombuffer(client.get(f"/api/masks/rasters/{name}").content, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    h, w = raster.shape
    assert raster[h // 2, w // 2] == 255 and raster[5, 5] == 0
    assert client.post(f"/api/photos/{photo}/retouch/area",
                       json={"params": params, "mask_index": 1}).status_code == 400
    nothing = {"masks": [{"kind": "radial", "definition": {"cx": 2.5, "cy": 2.5}}]}
    assert client.post(f"/api/photos/{photo}/retouch/area",
                       json={"params": nothing, "mask_index": 0}).status_code == 400


def test_an_eraser_is_filled_by_the_worker_without_a_new_version(client, project, catalog):
    photo = project["landscape"]
    area = client.post("/api/masks/rasters", content=_raster()).json()["name"]
    erase = {"kind": "erase", "id": "e1", "area": area}
    saved = client.put(f"/api/photos/{photo}/params", json={"params": {"retouch": [erase]}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["params"]["retouch"][0]["fill"] is None
    versions = _versions(catalog, photo)
    state = client.post(f"/api/photos/{photo}/retouch/states", json={"retouch": [erase]}).json()
    assert state["items"][0]["state"] == "computing"
    assert state["ml"]["available"] is False

    from ape.db.models import JobKind

    assert _run_jobs(catalog, [JobKind.RETOUCH_FILL]) == 1
    assert _versions(catalog, photo) == versions, "l'arrivo della patch non è un gesto"
    state = client.post(f"/api/photos/{photo}/retouch/states", json={"retouch": [erase]}).json()
    assert state["items"][0]["state"] == "ready"
    shown = client.post(f"/api/photos/{photo}/preview", json={"retouch": [erase]})
    hidden = client.post(f"/api/photos/{photo}/preview?retouch=false", json={"retouch": [erase]})
    assert shown.status_code == hidden.status_code == 200
    assert shown.content != hidden.content
    # The next save carries the fill the worker made.
    again = client.put(f"/api/photos/{photo}/params",
                       json={"params": {"retouch": [erase], "exposure": {"ev": 0.3}}}).json()
    assert again["params"]["retouch"][0]["fill"] is not None


def test_a_missing_model_is_an_error_until_retried(client, project, catalog):
    from ape.db.models import JobKind

    photo = project["landscape"]
    area = client.post("/api/masks/rasters", content=_raster()).json()["name"]
    erase = {"kind": "erase", "id": "e1", "area": area, "engine": "ml"}
    client.put(f"/api/photos/{photo}/params", json={"params": {"retouch": [erase]}})
    _run_jobs(catalog, [JobKind.RETOUCH_FILL])
    state = client.post(f"/api/photos/{photo}/retouch/states", json={"retouch": [erase]}).json()
    assert state["items"][0]["state"] == "error"
    assert "IA" in state["items"][0]["error"]
    assert _run_jobs(catalog, [JobKind.RETOUCH_FILL]) == 0, "non riaccodato da solo"
    assert client.post(f"/api/photos/{photo}/retouch/retry").status_code == 200
    assert _run_jobs(catalog, [JobKind.RETOUCH_FILL]) == 1


@pytest.mark.parametrize(
    "item",
    [
        {"kind": "erase", "id": "e1", "area": "../../catalog"},
        {"kind": "erase", "id": "e1", "area": "a" * 64, "fill": "zz"},
        {"kind": "erase", "id": "../x", "area": "a" * 64},
        {"kind": "heal", "id": "h1", "cx": 1.5, "cy": 0.5, "sx": 0.1, "sy": 0.1},
        {"kind": "clone", "id": "c1"},
    ],
)
def test_removals_that_are_not_ours_are_refused(client, project, item):
    photo = project["landscape"]
    assert client.put(f"/api/photos/{photo}/params",
                      json={"params": {"retouch": [item]}}).status_code == 400
    assert client.post(f"/api/photos/{photo}/preview",
                       json={"retouch": [item]}).status_code == 400


def test_two_removals_with_the_same_id_are_refused(client, project):
    heal = {"kind": "heal", "id": "h1", "cx": 0.5, "cy": 0.5, "sx": 0.6, "sy": 0.5}
    answer = client.put(f"/api/photos/{project['landscape']}/params",
                        json={"params": {"retouch": [heal, heal]}})
    assert answer.status_code == 400


def test_spots_copied_to_a_portrait_land_on_the_same_photosites(client, project, catalog):
    """Test 11: a spot at (0.2, 0.3) of a landscape frame is the sensor's
    (0.2, 0.3); on a portrait turned counter-clockwise (LibRaw 5) that is
    (0.3, 0.8) of its upright frame."""
    from ape.db.models import JobKind

    heal = {"kind": "heal", "id": "h1", "cx": 0.2, "cy": 0.3, "radius": 0.006,
            "sx": 0.24, "sy": 0.3}
    client.put(f"/api/photos/{project['landscape']}/params", json={"params": {"retouch": [heal]}})
    answer = client.post(
        f"/api/projects/{project['id']}/retouch/copy",
        json={"photo_id": project["landscape"], "targets": [project["portrait"]]},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["queued"] == 1
    assert _run_jobs(catalog, [JobKind.RETOUCH_COPY]) == 1
    copied = client.get(f"/api/photos/{project['portrait']}").json()["params"]["retouch"]
    assert len(copied) == 1
    spot = copied[0]
    assert spot["cx"] == pytest.approx(0.3) and spot["cy"] == pytest.approx(0.8)
    assert spot["radius"] == pytest.approx(0.006)
    assert (spot["sx"], spot["sy"]) != (spot["cx"], spot["cy"]), "sorgente cercata di nuovo"
    # A second copy adds, never replaces.
    client.post(f"/api/projects/{project['id']}/retouch/copy",
                json={"photo_id": project["landscape"], "targets": [project["portrait"]]})
    _run_jobs(catalog, [JobKind.RETOUCH_COPY])
    assert len(client.get(f"/api/photos/{project['portrait']}").json()["params"]["retouch"]) == 2


def test_the_portrait_frame_is_the_sensor_turned_counter_clockwise(project):
    """What test 11 rests on: LibRaw's 5 is a quarter turn counter-clockwise,
    measured on the RAW itself rather than taken from the documentation."""
    import rawpy

    from ape.retouch.copy import turns

    path = project["folder"] / _PORTRAIT
    with rawpy.imread(str(path)) as raw:
        flip = raw.sizes.flip
        upright = raw.postprocess(half_size=True, output_bps=8, no_auto_bright=True)
    with rawpy.imread(str(path)) as raw:
        sensor = raw.postprocess(half_size=True, output_bps=8, no_auto_bright=True, user_flip=0)
    turned = np.rot90(sensor, turns(flip))
    assert turned.shape == upright.shape
    small = [cv2.resize(image, (96, 64)).astype(np.float32) for image in (turned, upright)]
    assert np.corrcoef(small[0].ravel(), small[1].ravel())[0, 1] > 0.99


def test_a_restored_version_finds_its_fills(client, project, catalog):
    from ape.db.models import JobKind

    photo = project["landscape"]
    area = client.post("/api/masks/rasters", content=_raster()).json()["name"]
    erase = {"kind": "erase", "id": "e1", "area": area}
    first = client.put(f"/api/photos/{photo}/params", json={"params": {"retouch": [erase]}}).json()
    _run_jobs(catalog, [JobKind.RETOUCH_FILL])
    client.put(f"/api/photos/{photo}/params", json={"params": {}})
    restored = client.post(f"/api/photos/{photo}/versions/{first['current_version_id']}/restore")
    assert restored.status_code == 200
    params = restored.json()["params"]
    state = client.post(f"/api/photos/{photo}/retouch/states", json=params).json()
    assert state["items"][0]["state"] == "ready"

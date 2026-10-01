# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Segmentation on request (section 6.3, phase 9): the subjects, and the job.

The rules that clean up the models' mistakes are checked on synthetic maps;
what the models find is checked on the fixtures, when the pinned models are on
this machine (they are downloaded on request, never with the tests): a sky
where there is one, none on a frozen pond, nobody in a close-up of bark.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from ape.analysis import segment

FIXTURES = Path(__file__).parent / "fixtures"


def test_only_the_regions_the_rule_names_are_kept():
    probability = np.zeros((200, 300), dtype=np.float32)
    probability[0:40, 50:250] = 1.0  # touches the top
    probability[120:180, 50:250] = 1.0  # does not
    top = segment._regions(probability, lambda labels: np.unique(labels[:4]))
    assert top[20, 150] > 0.99
    assert top[150, 150] < 0.01
    # Soft at its edge, so the feathering of the kept region survives.
    assert 0.0 < top[44, 150] < 1.0
    assert not segment._regions(np.zeros_like(probability), lambda labels: [1]).any()


def _preview(name: str) -> np.ndarray:
    from ape.raw.embedded import read_embedded_preview

    image = read_embedded_preview(FIXTURES / f"{name}.ARW").image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


needs_models = pytest.mark.skipif(
    not (segment.status("sky")[0] and segment.status("person")[0]),
    reason="modelli di segmentazione non scaricati su questa macchina",
)


@needs_models
@pytest.mark.slow
def test_the_sky_is_found_above_the_hills_and_not_on_a_frozen_pond():
    sky = segment.segment(_preview("DSC05636"), "sky")
    height, width = sky.shape
    # Between the bare branches on the left and the wooded slope on the right.
    assert sky[: height // 10, int(0.3 * width) : int(0.5 * width)].mean() > 0.8
    assert sky[:, int(0.8 * width) :].mean() < 0.05
    assert sky[height // 2 :].mean() < 0.02
    # The pond is flat and pale, and the model alone calls it sky.
    assert segment.segment(_preview("DSC05634"), "sky").mean() < 0.01


@needs_models
@pytest.mark.slow
def test_bark_is_not_a_person():
    image = _preview("DSC05618")
    assert segment.segment(image, "person").mean() < 0.01
    assert segment.segment(image, "skin").mean() < 0.01


@needs_models
@pytest.mark.slow
def test_the_job_stores_a_raster_the_api_hands_out(catalog, tmp_path, monkeypatch):
    """Through the queue and the API, the way the interface asks for a sky."""
    from fastapi.testclient import TestClient

    from ape.api.app import create_app
    from ape.config import get_settings
    from ape.jobs.queue import claim_job, complete_job
    from ape.jobs.worker import run_job

    # The test's XDG directories have no models: borrow this machine's.
    real = Path.home() / ".local" / "share" / "autophotoedit" / "models"
    monkeypatch.setattr(type(get_settings()), "models_dir", property(lambda _self: real))

    folder = tmp_path / "scheda"
    folder.mkdir()
    shutil.copy2(FIXTURES / "DSC05636.ARW", folder)

    def drain() -> None:
        while True:
            with catalog() as session:
                record = claim_job(session)
            if record is None:
                return
            run_job(record, lambda _fraction: None)
            with catalog() as session:
                complete_job(session, record.id)
                session.commit()

    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as client:
        project = client.post("/api/projects", json={"name": "s", "source_dir": str(folder)})
        drain()
        photo = client.get(f"/api/projects/{project.json()['id']}/photos").json()["items"][0]
        url = f"/api/photos/{photo['id']}/segments"
        states = {entry["subject"]: entry for entry in client.get(url).json()}
        assert states["sky"]["state"] == "none" and states["sky"]["available"]

        assert client.post(f"{url}/sky").json()["state"] == "queued"
        drain()
        ready = client.post(f"{url}/sky").json()
        assert ready["state"] == "ready" and ready["raster"]

        definition = {"subject": "sky", "raster": ready["raster"]}
        params = {"masks": [{"kind": "segment", "definition": definition, "exposure": {"ev": -1}}]}
        assert client.post(f"/api/photos/{photo['id']}/preview", json=params).status_code == 200
        assert client.post(f"{url}/sea").status_code == 404

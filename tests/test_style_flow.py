# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 6 end to end, over the API, on the user's files.

A profile is created from two folders -- the RAWs and "edits" of them made by
this very pipeline with a known look, one of them renamed so that it needs the
manual pairing screen. The jobs invert the pairs and train the profile. A
project then chooses it, the predictions land as versions, a photo the user
edits by hand keeps its edit through a re-application, and nothing of either
folder is touched (section 2).

The whole flow runs the real jobs one at a time, as ``test_analysis_api.py``.
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

pytestmark = [pytest.mark.fixtures, pytest.mark.slow]


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


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


def _look():
    from ape.style import vector as sv

    v = sv.neutral_vector()
    for name, value in {
        "exposure_offset": 0.3,
        "contrast": 1.5,
        "vibrance": 0.35,
        "wb_mired_shift": 15.0,
        "local_highlights": -0.3,
        "hsl_green_saturation": 0.25,
    }.items():
        v[sv.index(name)] = value
    return v


def _edit(raw: Path, destination: Path) -> None:
    """The "user's edit": the known look, rendered and saved as a JPEG."""
    from PIL import Image

    from ape.pipeline.filters import resize_long_edge
    from ape.pipeline.render import render
    from ape.raw.decode import decode_linear
    from ape.style import auto
    from ape.style import vector as sv

    decoded = decode_linear(raw, half_size=True)
    decoded.rgb = np.ascontiguousarray(resize_long_edge(decoded.rgb, 1200))
    from ape.pipeline.params import EditParams

    neutral = render(decoded, EditParams())
    camera = decoded.camera
    anchor = auto.measure(
        neutral,
        as_shot_temperature_k=camera.as_shot_temperature_k,
        as_shot_tint=camera.as_shot_tint,
        white_balance=False,
    ).exposure_anchor_ev
    context = sv.StyleContext(camera.as_shot_temperature_k, camera.as_shot_tint, anchor)
    image = render(decoded, sv.to_params(_look(), context))
    Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        destination, quality=92
    )


def _distance_to_look(raw: Path, params: dict) -> tuple[float, float]:
    """Mean ΔE between these parameters and the true look, and neutral's."""
    from ape.pipeline.filters import resize_long_edge
    from ape.pipeline.params import EditParams
    from ape.pipeline.render import render
    from ape.raw.decode import decode_linear
    from ape.style import auto
    from ape.style import vector as sv
    from ape.style.colordiff import delta_e_2000, srgb_to_lab

    decoded = decode_linear(raw, half_size=True)
    decoded.rgb = np.ascontiguousarray(resize_long_edge(decoded.rgb, 512))
    camera = decoded.camera
    neutral = render(decoded, EditParams())
    anchor = auto.measure(
        neutral,
        as_shot_temperature_k=camera.as_shot_temperature_k,
        as_shot_tint=camera.as_shot_tint,
        white_balance=False,
    ).exposure_anchor_ev
    truth = render(
        decoded,
        sv.to_params(
            _look(), sv.StyleContext(camera.as_shot_temperature_k, camera.as_shot_tint, anchor)
        ),
    )
    predicted = EditParams.from_dict(params)
    predicted.geometry = EditParams().geometry
    ours = render(decoded, predicted)

    def distance(a, b):
        return float(delta_e_2000(srgb_to_lab(a), srgb_to_lab(b)).mean())

    return distance(ours, truth), distance(neutral, truth)


@pytest.fixture
def folders(tmp_path: Path) -> tuple[Path, Path, Path]:
    raws = {p.stem: p for p in available_raws()}
    training = ["DSC05618", "DSC05620", "DSC05623", "DSC05625", "DSC05628", "DSC05637"]
    project = ["DSC05634", "DSC05635", "DSC05638"]
    if not all(name in raws for name in training + project):
        pytest.skip("servono i file ARW di tests/fixtures/")
    raw_dir, ref_dir, card = tmp_path / "raw", tmp_path / "editate", tmp_path / "scheda"
    for folder in (raw_dir, ref_dir, card):
        folder.mkdir()
    for name in training:
        shutil.copy2(raws[name], raw_dir)
        # One renamed and without metadata: only the manual screen can pair it.
        target = "consegna_finale.jpg" if name == training[-1] else f"{name}.jpg"
        _edit(raws[name], ref_dir / target)
    for name in project:
        shutil.copy2(raws[name], card)
    return raw_dir, ref_dir, card


def test_learn_choose_predict_keep_user_edits(client: TestClient, catalog, folders):
    raw_dir, ref_dir, card = folders
    before = {folder: file_state(folder) for folder in folders}

    listing = client.get("/api/styles").json()
    assert [p["name"] for p in listing if p["builtin"]] == [
        "Naturale",
        "Neutro automatico",
        "Paesaggio",
        "Ritratto",
    ]

    created = client.post(
        "/api/styles",
        json={
            "name": "Look di prova",
            "raw_dir": str(raw_dir),
            "reference_dir": str(ref_dir),
        },
    )
    assert created.status_code == 201, created.text
    profile = created.json()
    assert profile["pairing"]["pairs"] == 5
    assert profile["pairing"]["methods"]["name"] == 5
    assert [Path(p).name for p in profile["pairing"]["unpaired_references"]] == [
        "consegna_finale.jpg"
    ]

    unpaired = client.get(f"/api/styles/{profile['id']}/unpaired").json()
    assert len(unpaired["raws"]) == 1 and len(unpaired["references"]) == 1
    manual = client.post(
        f"/api/styles/{profile['id']}/pairs",
        json={
            "raw": unpaired["raws"][0],
            "reference": unpaired["references"][0],
        },
    )
    assert manual.status_code == 201, manual.text

    _run_all(catalog)
    detail = client.get(f"/api/styles/{profile['id']}").json()
    assert detail["trained"] and detail["samples"]["ready"] == 6
    assert detail["few_pairs"]  # fewer than 8: the interface warns (section 8.1)
    for sample in detail["sample_list"]:
        # The pipeline made these edits itself, so it reproduces them closely.
        assert sample["delta_e"] < 2.5, sample
        assert client.get(f"/api/styles/samples/{sample['id']}/thumbnail").status_code == 200

    project = client.post(
        "/api/projects", json={"name": "passeggiata", "source_dir": str(card)}
    ).json()
    _run_all(catalog)
    state = client.get(f"/api/projects/{project['id']}/style").json()
    assert state["proposed"] is not None and state["current"] is None

    chosen = client.put(f"/api/projects/{project['id']}/style", json={"profile_id": profile["id"]})
    assert chosen.json()["queued"] == 3
    _run_all(catalog)
    state = client.get(f"/api/projects/{project['id']}/style").json()
    assert state["applied"]["written"] == 3 and state["pending"] == 0

    photos = client.get(f"/api/projects/{project['id']}/photos").json()["items"]
    assert all(p["status"] == "predicted" for p in photos)
    first = client.get(f"/api/photos/{photos[0]['id']}").json()
    analysis_rotation = first["analysis"]["straighten"]["rotation_deg"]
    # The photo's own geometry stayed, and the look arrived -- judged on the
    # picture, not on the sliders: the inversion may reach the same look with
    # a different split between contrast, curve and local tone.
    assert first["params"]["geometry"]["rotation_deg"] == pytest.approx(analysis_rotation, abs=1e-3)
    for photo in photos:
        params = client.get(f"/api/photos/{photo['id']}").json()["params"]
        error, neutral_error = _distance_to_look(card / photo["filename"], params)
        assert error < 4.0 and error < 0.6 * neutral_error, (photo["filename"], error)
    style = client.get(f"/api/photos/{photos[0]['id']}/style").json()
    assert style["profile"]["name"] == "Look di prova" and len(style["neighbours"]) == 5

    # The user takes one photo over; a new coherence value re-applies the rest.
    edited = dict(first["params"])
    edited["exposure"] = {"ev": -1.0}
    assert (
        client.put(f"/api/photos/{photos[0]['id']}/params", json={"params": edited}).status_code
        == 200
    )
    client.put(
        f"/api/projects/{project['id']}/style",
        json={"profile_id": profile["id"], "coherence_lambda": 0.0},
    )
    state = client.get(f"/api/projects/{project['id']}/style").json()
    assert state["applied"]["kept_user_edit"] == 1
    kept = client.get(f"/api/photos/{photos[0]['id']}").json()
    assert kept["params"]["exposure"]["ev"] == -1.0

    # "This profile vs neutral" (section 22): the parameters of another profile.
    neutral_id = next(p["id"] for p in listing if p["name"] == "Neutro automatico")
    compare = client.get(
        f"/api/photos/{photos[1]['id']}/style/params", params={"profile_id": neutral_id}
    )
    assert compare.status_code == 200 and "tone" in compare.json()["params"]

    exported = client.get(f"/api/styles/{profile['id']}/export")
    assert exported.status_code == 200 and exported.content[:2] == b"PK"
    imported = client.post("/api/styles/import", content=exported.content)
    assert imported.status_code == 201 and imported.json()["trained"]

    for folder, state_before in before.items():
        assert file_state(folder) == state_before, f"§2 violato in {folder}"

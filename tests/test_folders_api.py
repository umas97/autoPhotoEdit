# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Browsing folders to choose a project's source (``api/routes_folders.py``).

The list the "Sfoglia" panel walks: sub-folders sorted as a person reads them,
hidden ones left out, the RAWs of the folder counted the way the import will
count them -- and nothing written anywhere, which section 2 asks of every
folder the program looks at.
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
    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as test_client:
        yield test_client


@pytest.fixture
def card(tmp_path: Path) -> Path:
    root = tmp_path / "SCHEDA"
    for name in ("DCIM", "private", "avchd", ".Trash-1000"):
        (root / name).mkdir(parents=True)
    photos = root / "DCIM" / "10000930"
    for index in range(3):
        write_raw(photos, f"DSC0{index:04d}.ARW")
    (photos / "DSC00000.JPG").write_bytes(b"jpeg")
    (photos / "sub").mkdir()
    return root


def test_a_folder_lists_its_folders_and_counts_its_raws(client: TestClient, card: Path):
    answer = client.get("/api/folders", params={"path": str(card)}).json()
    assert answer["path"] == str(card.resolve())
    assert answer["parent"] == str(card.resolve().parent)
    assert [f["name"] for f in answer["folders"]] == ["avchd", "DCIM", "private"]
    assert answer["raw_count"] == 0 and not answer["truncated"]

    photos = card / "DCIM" / "10000930"
    before = file_state(photos)
    answer = client.get("/api/folders", params={"path": str(photos)}).json()
    assert answer["raw_count"] == 3  # the JPEG is not a RAW, the sub-folder is not counted
    assert answer["reference_count"] == 1  # but it is an edited photo, for a style profile
    assert [f["name"] for f in answer["folders"]] == ["sub"]
    assert file_state(photos) == before  # section 2: looked at, never touched

    shown = client.get("/api/folders", params={"path": str(card), "hidden": True}).json()
    assert ".Trash-1000" in [f["name"] for f in shown["folders"]]


def test_without_a_path_the_browser_starts_at_home(client: TestClient):
    answer = client.get("/api/folders").json()
    assert answer["path"] == str(Path.home().resolve())
    kinds = [p["kind"] for p in answer["places"]]
    assert kinds[0] == "home" and kinds[-1] == "root"


def test_what_is_not_a_folder_is_said_plainly(client: TestClient, card: Path):
    missing = client.get("/api/folders", params={"path": str(card / "nope")})
    assert missing.status_code == 404 and "non esiste" in missing.json()["detail"]
    relative = client.get("/api/folders", params={"path": "Foto"})
    assert relative.status_code == 400
    raw = card / "DCIM" / "10000930" / "DSC00000.ARW"
    assert client.get("/api/folders", params={"path": str(raw)}).status_code == 404

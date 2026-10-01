# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Only a local page may talk to the server (``api/local_only.py``).

A rebinding page arrives with a foreign ``Host``; a cross-site one with the
right ``Host`` and a foreign ``Origin``. Both are refused, on HTTP and on the
WebSocket alike, and a refused write writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ape.api.app import create_app


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    with TestClient(create_app(start_workers=False), base_url="http://127.0.0.1") as test_client:
        yield test_client


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8787", "localhost:5173", "LOCALHOST"])
def test_loopback_hosts_pass(client: TestClient, host: str) -> None:
    assert client.get("/api/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8787", "127.0.0.1.evil.example", "testserver", ""]
)
def test_foreign_host_is_refused(client: TestClient, host: str) -> None:
    response = client.get("/api/health", headers={"host": host})
    assert response.status_code == 400
    assert "host non locale" in response.json()["detail"]


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:8787", "http://localhost:5173", "http://127.0.0.1"]
)
def test_local_origin_passes(client: TestClient, origin: str) -> None:
    assert client.get("/api/projects", headers={"origin": origin}).status_code == 200


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "http://127.0.0.1.evil.example", "null", "file://"]
)
def test_foreign_origin_cannot_write(client: TestClient, tmp_path: Path, origin: str) -> None:
    payload = {"name": "intruso", "source_dir": str(tmp_path), "import_now": False}
    response = client.post("/api/projects", json=payload, headers={"origin": origin})
    assert response.status_code == 403
    assert client.get("/api/projects").json() == []


def test_websocket_from_local_page(client: TestClient) -> None:
    headers = {"origin": "http://127.0.0.1:8787"}
    with client.websocket_connect("ws://127.0.0.1/ws", headers=headers) as socket:
        assert socket.receive_json()["type"] == "progress"


@pytest.mark.parametrize(
    ("url", "headers"),
    [
        ("ws://127.0.0.1/ws", {"origin": "https://evil.example"}),
        ("ws://evil.example/ws", {}),
    ],
)
def test_websocket_from_foreign_page_is_refused(
    client: TestClient, url: str, headers: dict[str, str]
) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect(url, headers=headers),
    ):
        pass
    assert refused.value.code == 1008

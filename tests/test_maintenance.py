# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sections 19 and 20.2: the log file, the Problems panel, the diagnostics
bundle, backup and restore.

The bundle is checked for what it must *not* contain as much as for what it
does: no byte of a RAW, no project name, no author, no home directory.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from ape.api.app import create_app
from conftest_catalog import file_state, write_raw

#: Bytes that stand for the pixels of a RAW: they must never leave the file.
PIXELS = b"PIXEL-DATA-OF-A-PRIVATE-PHOTO"


@pytest.fixture
def client(catalog) -> Iterator[TestClient]:
    app = create_app(start_workers=False)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


@pytest.fixture
def failing(catalog, tmp_path: Path):
    """A project named like a secret, one failed photo with a traceback naming home."""
    from ape.db.enums import JobKind, JobState, PhotoStatus
    from ape.db.models import Job, Photo, Project, Setting

    folder = tmp_path / "Segreto"
    raw = write_raw(folder, "DSC00001.ARW", content=PIXELS * 64)
    write_raw(folder, "DSC00002.ARW")
    home = str(Path.home())
    with catalog() as session:
        project = Project(name="Progetto Segreto", source_dir=str(folder))
        session.add(project)
        session.flush()
        broken = Photo(
            project_id=project.id,
            path=str(raw),
            filename=raw.name,
            camera="ILCE-7M3",
            lens="E 28-75mm F2.8-2.8",
            iso=400,
            status=PhotoStatus.FAILED,
            error=f"ValueError: file RAW illeggibile: {raw.name} (data corrupted)",
        )
        fine = Photo(
            project_id=project.id, path=str(folder / "DSC00002.ARW"), filename="DSC00002.ARW"
        )
        session.add_all([broken, fine])
        session.flush()
        session.add(
            Job(
                project_id=project.id,
                kind=JobKind.PROXY,
                payload={"photo_id": broken.id},
                state=JobState.FAILED,
                attempts=3,
                error=f"ValueError: file RAW illeggibile: {raw.name}",
                traceback=(
                    f'Traceback (most recent call last):\n  File "{home}/x.py", line 1\nValueError'
                ),
            )
        )
        # A job that failed on a photo that is fine: a problem of its own.
        session.add(
            Job(
                project_id=project.id,
                kind=JobKind.SEGMENT,
                payload={"photo_id": fine.id},
                state=JobState.FAILED,
                error="RuntimeError: modello non scaricato",
            )
        )
        session.add(Setting(key="artist", value="Mario Rossi"))
        session.commit()
        return {"project": project.id, "broken": broken.id, "fine": fine.id, "folder": folder}


def test_the_log_file_rotates_at_five_files_of_five_megabytes(xdg_home):
    from ape import logs
    from ape.config import get_settings

    root = logging.getLogger()
    terminal = logging.StreamHandler(io.StringIO())
    root.addHandler(terminal)
    root.setLevel(logging.WARNING)
    try:
        handler = logs.configure_server()
        assert handler.maxBytes == 5 * 1024 * 1024 and handler.backupCount == 4
        logging.getLogger("ape.test").info("una riga nel file")
        handler.flush()
        text = (get_settings().log_dir / logs.LOG_NAME).read_text(encoding="utf-8")
        assert "una riga nel file" in text
        # The file wants INFO; the terminal keeps the level it had.
        assert terminal.level == logging.WARNING and "una riga" not in terminal.stream.getvalue()
        assert logs.log_files() == [get_settings().log_dir / logs.LOG_NAME]
        # A second configuration replaces the first, it does not duplicate lines.
        logs.configure_worker()
        assert sum(getattr(h, "_ape_file_handler", False) for h in root.handlers) == 1
    finally:
        for h in [h for h in root.handlers if getattr(h, "_ape_file_handler", False)]:
            root.removeHandler(h)
            h.close()
        root.removeHandler(terminal)


def test_a_failed_job_keeps_its_traceback_for_the_details(catalog, monkeypatch):
    from ape.db.enums import JobKind, JobState
    from ape.db.models import Job
    from ape.jobs import queue as jobq
    from ape.jobs import worker

    def explode(record, progress):
        raise ValueError("file RAW illeggibile: prova.ARW")

    worker._load_handlers()  # the production handlers first, or they win over ours
    monkeypatch.setitem(worker.HANDLERS, JobKind.SEGMENT, explode)
    with catalog() as session:
        jobq.enqueue(session, JobKind.SEGMENT, {"photo_id": 1})
        session.commit()
        record = jobq.claim_job(session, [JobKind.SEGMENT], worker_pid=os.getpid())
        session.commit()
    for _ in range(jobq.MAX_ATTEMPTS):
        worker._run_one(catalog, record)
        with catalog() as session:
            record = jobq.claim_job(session, [JobKind.SEGMENT], worker_pid=os.getpid()) or record
            session.commit()
    with catalog() as session:
        job = session.scalars(select(Job)).one()
        assert job.state is JobState.FAILED
        assert job.error == "ValueError: file RAW illeggibile: prova.ARW"
        assert "Traceback" in job.traceback and "explode" in job.traceback


def test_the_problems_panel_says_what_failed_and_retries_it(client: TestClient, catalog, failing):
    from ape.db.enums import JobState
    from ape.db.models import Job

    overview = client.get("/api/problems").json()
    assert overview["count"] == 2
    (photo,) = overview["photos"]
    assert photo["filename"] == "DSC00001.ARW"
    # The class name is for the log; the sentence is for the person.
    assert photo["reason"] == "file RAW illeggibile: DSC00001.ARW (data corrupted)"
    assert photo["stage"] == "anteprima"
    assert str(Path.home()) not in photo["details"] and "~/x.py" in photo["details"]
    (other,) = overview["jobs"]
    assert other["stage"] == "segmentazione" and other["reason"] == "modello non scaricato"
    assert client.get("/api/problems", params={"project_id": 999}).json()["count"] == 0
    from ape.problems import plain

    assert plain("ValueError: file RAW illeggibile: x.ARW (b'Unsupported file format')") == (
        "file RAW illeggibile: x.ARW (Unsupported file format)"
    )
    assert plain("MemoryError") == "memoria esaurita durante l'elaborazione"
    assert plain("ZeroDivisionError: division by zero") == "ZeroDivisionError: division by zero"

    # The count is in the progress message: always visible, no polling of its own.
    with client.websocket_connect("ws://127.0.0.1/ws") as socket:
        assert socket.receive_json()["problems"] == 2

    retried = client.post("/api/problems/retry", json={}).json()
    assert retried["retried"] == 2
    with catalog() as session:
        assert all(j.state is JobState.QUEUED for j in session.scalars(select(Job)).all())


def test_a_failed_photo_without_a_failed_job_is_retried_from_its_proxy(client, catalog, failing):
    from ape.db.enums import JobKind, JobState
    from ape.db.models import Job

    with catalog() as session:
        for job in session.scalars(select(Job)).all():
            session.delete(job)
        session.commit()
    answer = client.post("/api/problems/retry", json={"photo_ids": [failing["broken"]]}).json()
    assert answer["retried"] == 1
    with catalog() as session:
        (job,) = session.scalars(select(Job)).all()
        assert job.kind is JobKind.PROXY and job.state is JobState.QUEUED
        assert job.payload == {"photo_id": failing["broken"], "force": True}


def test_the_bundle_is_what_the_plan_says_and_nothing_private(client: TestClient, failing):
    from ape.config import get_settings
    from ape.logs import LOG_NAME

    log = get_settings().log_dir / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"INFO avvio da {Path.home()}/Immagini\n", encoding="utf-8")

    plan = client.get("/api/diagnostics/plan").json()
    names = [entry["name"] for entry in plan]
    assert names[:4] == ["versioni.json", "catalogo.json", "foto_fallite.json", "impostazioni.json"]
    assert f"log/{LOG_NAME}" in names
    assert all(entry["description"] and entry["size"] > 0 for entry in plan)

    response = client.get("/api/diagnostics/bundle.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    payload = response.content
    archive = zipfile.ZipFile(io.BytesIO(payload))
    assert archive.namelist() == names
    everything = b"".join(archive.read(name) for name in archive.namelist())
    assert PIXELS not in everything and PIXELS not in payload
    assert str(Path.home()).encode() not in everything
    assert b"Segreto" not in everything and b"Mario Rossi" not in everything

    failed = json.loads(archive.read("foto_fallite.json"))
    assert failed[0]["file"] == "DSC00001.ARW" and failed[0]["exif"]["camera"] == "ILCE-7M3"
    assert failed[0]["errore"].startswith("file RAW illeggibile")
    versions = json.loads(archive.read("versioni.json"))
    assert versions["python"] and versions["librerie"]["libraw"]
    catalogue = json.loads(archive.read("catalogo.json"))
    assert (
        catalogue["foto_per_stato"]["failed"] == 1 and catalogue["righe_per_tabella"]["photo"] == 2
    )
    assert "~/Immagini" in archive.read(f"log/{LOG_NAME}").decode()


def test_diagnose_on_the_command_line_writes_the_zip_and_never_into_a_source(
    catalog, failing, tmp_path: Path, capsys
):
    from ape.__main__ import main

    before = file_state(failing["folder"])
    out = tmp_path / "fuori"
    out.mkdir()
    assert main(["diagnose", "-o", str(out)]) == 0
    (written,) = list(out.glob("autophotoedit-diagnostica-*.zip"))
    assert "foto_fallite.json" in zipfile.ZipFile(written).namelist()
    assert "mai RAW" in capsys.readouterr().out
    # Refused by the guard of section 2: exit code 3, and nothing written.
    assert main(["diagnose", "-o", str(failing["folder"] / "diag.zip")]) == 3
    assert "cartella sorgente" in capsys.readouterr().err
    assert file_state(failing["folder"]) == before, "§2 violato nella cartella sorgente"


def test_backup_and_restore_keep_the_catalogue_and_the_one_replaced(
    catalog, failing, tmp_path: Path, xdg_home
):
    from ape.backup import BackupError, backup_catalogue, restore_catalogue
    from ape.config import get_settings
    from ape.db.models import Project
    from ape.safety import SourceWriteError
    from ape.single_instance import LockFile  # noqa: F401 - the lock format is its business

    # Consistent with a session open on the catalogue, as with the server on.
    session = catalog()
    session.get(Project, failing["project"])
    backup = backup_catalogue(tmp_path / "copia.db", [str(failing["folder"])])
    session.close()
    assert backup["bytes"] > 0
    with sqlite3.connect(backup["path"]) as copy:
        assert copy.execute("SELECT name FROM project").fetchall() == [("Progetto Segreto",)]
    with pytest.raises(BackupError, match="esiste già"):
        backup_catalogue(tmp_path / "copia.db")
    with pytest.raises(SourceWriteError):
        backup_catalogue(failing["folder"] / "copia.db", [str(failing["folder"])])

    with catalog() as session:
        session.get(Project, failing["project"]).name = "cambiato dopo il backup"
        session.commit()

    # While the program runs, no restore.
    lock = get_settings().lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "port": 8787}), encoding="utf-8")
    with pytest.raises(BackupError, match="in esecuzione"):
        restore_catalogue(backup["path"])
    lock.unlink()

    stranger = tmp_path / "non-un-catalogo.db"
    stranger.write_bytes(b"questo non e' sqlite")
    with pytest.raises(BackupError, match="non è un catalogo"):
        restore_catalogue(stranger)
    newer = tmp_path / "futuro.db"
    with sqlite3.connect(newer) as future:
        future.execute("CREATE TABLE setting (key TEXT, value TEXT)")
        future.execute("INSERT INTO setting VALUES ('schema_version', '999')")
    with pytest.raises(BackupError, match="più recente"):
        restore_catalogue(newer)

    result = restore_catalogue(backup["path"])
    assert result["aside"] is not None and result["aside"].exists()
    with sqlite3.connect(result["aside"]) as aside:
        assert aside.execute("SELECT name FROM project").fetchall() == [
            ("cambiato dopo il backup",)
        ]
    from ape.db.session import get_sessionmaker, init_db

    init_db()
    with get_sessionmaker()() as session:
        assert session.get(Project, failing["project"]).name == "Progetto Segreto"

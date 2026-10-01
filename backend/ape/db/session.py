# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine, sessions and the pragmas that make SQLite behave under a worker pool.

docs/SPEC.md section 4 settles the choice of SQLite and section 11 fixes two of the
pragmas. The rest of what is here exists because the catalogue is written by
sixteen processes at once, which is the one workload plain SQLite handles badly
by default:

``journal_mode = WAL``
    readers do not block the writer and the writer does not block readers. Not
    an optimisation: with the default rollback journal, a worker committing a
    finished job would freeze every reader for the length of the commit, and
    the UI polls constantly.

``synchronous = NORMAL``
    one fsync per checkpoint instead of one per commit. With WAL this is still
    crash-safe for the application: a power cut can lose the last transactions,
    never the file. The catalogue is rebuildable from the RAW files; the RAW
    files are what must not be risked, and those are never written.

``busy_timeout``
    without it, a worker that meets a locked database raises immediately.
    Fifteen seconds is far longer than any transaction here, so a timeout means
    a real problem rather than contention.

``foreign_keys = ON``
    SQLite does not enforce them otherwise, and every ``ondelete`` in
    ``models.py`` would be a comment.

Sessions come from :func:`session_scope`, which commits on success and rolls
back on failure. Nothing in the codebase should hold a session open across a
long computation: decode a RAW, render it, *then* open a session to record it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base, Setting
from .tint_migration import flip_tint

__all__ = [
    "SCHEMA_VERSION",
    "engine_for",
    "get_engine",
    "get_sessionmaker",
    "init_db",
    "read_setting",
    "reset_engine",
    "session_scope",
    "write_setting",
]

_log = logging.getLogger(__name__)

#: Bumped whenever ``models.py`` changes shape. Stored in ``Setting`` so that an
#: older build meeting a newer catalogue can say so instead of failing oddly.
SCHEMA_VERSION = 10

#: What turns a catalogue of version ``n - 1`` into version ``n``. ``create_all``
#: creates missing *tables* but never alters an existing one, so every column
#: added to a table that already shipped needs its statement here. A step is
#: SQL, or a function of the session for a rewrite SQL cannot express; what a
#: user wrote is only ever re-expressed, never changed in meaning.
_MIGRATIONS: dict[int, tuple[str | Callable[[Session], None], ...]] = {
    2: ("ALTER TABLE photo ADD COLUMN culling_features JSON",),
    3: (
        "ALTER TABLE photo ADD COLUMN analysis JSON",
        "ALTER TABLE photo ADD COLUMN cluster_rank INTEGER",
        "ALTER TABLE crop_proposal ADD COLUMN decided_at DATETIME",
        "ALTER TABLE project ADD COLUMN crop_proposals_paused BOOLEAN NOT NULL DEFAULT 0",
    ),
    4: (
        "ALTER TABLE photo ADD COLUMN prediction JSON",
        "ALTER TABLE style_profile ADD COLUMN trained_at DATETIME",
        "ALTER TABLE style_profile ADD COLUMN raw_dir TEXT",
        "ALTER TABLE style_profile ADD COLUMN reference_dir TEXT",
        "ALTER TABLE style_sample ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'pending'",
        "ALTER TABLE style_sample ADD COLUMN error TEXT",
        "ALTER TABLE style_sample ADD COLUMN pairing VARCHAR(16)",
        "ALTER TABLE style_sample ADD COLUMN vector JSON",
        "ALTER TABLE style_sample ADD COLUMN context JSON",
        "ALTER TABLE style_sample ADD COLUMN delta_e FLOAT",
        "ALTER TABLE style_sample ADD COLUMN thumbnail BLOB",
        "ALTER TABLE style_sample ADD COLUMN created_at DATETIME",
    ),
    5: (
        "ALTER TABLE project ADD COLUMN confidence_weights JSON NOT NULL DEFAULT '{}'",
        # The one rewrite of a value: 0.6 was the default of every earlier
        # build and no screen could change it, so it is nobody's choice.
        "UPDATE project SET confidence_threshold = 0.55 WHERE confidence_threshold = 0.6",
        "ALTER TABLE photo ADD COLUMN review JSON",
        "ALTER TABLE style_sample ADD COLUMN project_id INTEGER "
        "REFERENCES project(id) ON DELETE SET NULL",
        "ALTER TABLE style_sample ADD COLUMN photo_id INTEGER "
        "REFERENCES photo(id) ON DELETE SET NULL",
        "CREATE INDEX IF NOT EXISTS ix_style_sample_project_id ON style_sample (project_id)",
    ),
    # The export tables are new and ``create_all`` makes them; the project
    # only gains the options of the Export screen (``export/settings.py``).
    6: ("ALTER TABLE project ADD COLUMN export_settings JSON NOT NULL DEFAULT '{}'",),
    # The tint turned round to Adobe's sign: values rewritten, nothing added.
    7: (flip_tint,),
    # Snapshots also remember the review decision and who decided the culling
    # (section 23.2): a rollback to "fine predizione" takes the approvals back.
    8: ("ALTER TABLE snapshot_entry ADD COLUMN decisions JSON",),
    # The traceback of a failed job, behind "Dettagli tecnici" (section 19).
    9: ("ALTER TABLE job ADD COLUMN traceback TEXT",),
    # Phase 11: the frames an accepted merge replaces leave the working set,
    # and a group keeps what its preview and its merge measured (section 25).
    10: (
        "ALTER TABLE photo ADD COLUMN superseded BOOLEAN NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS ix_photo_superseded ON photo (superseded)",
        "ALTER TABLE merge_group ADD COLUMN report JSON",
    ),
}

_SCHEMA_KEY = "schema_version"

_engine: Engine | None = None
_sessions: sessionmaker[Session] | None = None


def _configure_connection(dbapi_connection: Any, _record: Any) -> None:
    """Apply the pragmas to every new connection, pooled or not."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=15000")
        # Keeps the temporary tables of a sort or a large join in memory rather
        # than in a file next to the database.
        cursor.execute("PRAGMA temp_store=MEMORY")
    finally:
        cursor.close()


def engine_for(path: str | Path, *, echo: bool = False) -> Engine:
    """Build an engine for one database file. ``":memory:"`` works, for tests."""
    if str(path) == ":memory:":
        url = "sqlite+pysqlite:///:memory:"
    else:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+pysqlite:///{target}"

    engine = create_engine(
        url,
        echo=echo,
        future=True,
        # Sessions are short-lived and never shared between threads, but the
        # API server does hand a connection to whichever thread FastAPI runs the
        # request on, and SQLite's own check is per-thread rather than
        # per-session. The pool, not the flag, is what keeps this safe.
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    event.listen(engine, "connect", _configure_connection)
    return engine


def get_engine() -> Engine:
    """The process-wide engine on the catalogue of ``config.py``."""
    global _engine
    if _engine is None:
        _engine = engine_for(get_settings().db_path)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _sessions
    if _sessions is None:
        _sessions = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _sessions


def reset_engine() -> None:
    """Drop the cached engine. For tests, and after a fork.

    A forked child must not inherit its parent's SQLite connections: two
    processes writing through the same file descriptor is exactly the situation
    SQLite's locking cannot see.
    """
    global _engine, _sessions
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _sessions = None


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """A session that commits on the way out and rolls back on an exception."""
    maker = factory or get_sessionmaker()
    session = maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> Engine:
    """Create the schema if it is not there, and check the version if it is.

    Raises:
        RuntimeError: if the catalogue was written by a newer build. Opening it
            anyway would mean writing rows an older schema cannot express, and
            the damage would be silent.
    """
    target = engine or get_engine()
    Base.metadata.create_all(target)

    maker = sessionmaker(bind=target, expire_on_commit=False, future=True)
    with session_scope(maker) as session:
        row = session.get(Setting, _SCHEMA_KEY)
        if row is None:
            session.add(Setting(key=_SCHEMA_KEY, value=SCHEMA_VERSION))
        elif int(row.value) > SCHEMA_VERSION:  # type: ignore[arg-type]
            raise RuntimeError(
                f"il catalogo è stato scritto da una versione più recente "
                f"(schema {row.value}, questa build arriva a {SCHEMA_VERSION}). "
                "Aggiorna autoPhotoEdit."
            )
        elif int(row.value) < SCHEMA_VERSION:  # type: ignore[arg-type]
            _migrate(session, int(row.value))  # type: ignore[arg-type]
            row.value = SCHEMA_VERSION
        # A jobs table left behind by a crash: section 11 says running jobs go
        # back into the queue at startup. Doing it here, before any worker can
        # start, is what makes the resume of test 6 deterministic.
        requeued = _requeue_orphans(session)
    if requeued:
        _log.info("ripresi %d job interrotti", requeued)
    return target


def _migrate(session: Session, current: int) -> None:
    """Bring a catalogue from ``current`` up to :data:`SCHEMA_VERSION`, in steps."""
    from sqlalchemy import inspect, text

    inspector = inspect(session.connection())
    columns: dict[str, set[str]] = {}
    for version in range(current + 1, SCHEMA_VERSION + 1):
        for statement in _MIGRATIONS.get(version, ()):
            if callable(statement):
                statement(session)
                continue
            # A column that is already there -- a catalogue created by
            # ``create_all`` of a newer build and then stamped with an old
            # version by hand -- is not an error worth refusing to open over.
            words = statement.split()
            if "ADD COLUMN" in statement:
                table = words[2]
                if table not in columns:
                    columns[table] = {c["name"] for c in inspector.get_columns(table)}
                if words[5] in columns[table]:
                    continue
            session.execute(text(statement))
        _log.info("catalogo aggiornato allo schema %d", version)


def _requeue_orphans(session: Session) -> int:
    from .models import Job, JobState

    orphans = session.scalars(select(Job).where(Job.state == JobState.RUNNING)).all()
    for job in orphans:
        job.state = JobState.QUEUED
        job.claimed_by = None
        job.started_at = None
        job.progress = 0.0
    return len(orphans)


def read_setting(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(Setting, key)
    return default if row is None else row.value


def write_setting(session: Session, key: str, value: Any) -> None:
    row = session.get(Setting, key)
    if row is None:
        session.add(Setting(key=key, value=value))
    else:
        row.value = value


def register_fork_hook() -> None:
    """Make every forked child drop the parent's connections. Idempotent."""
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=reset_engine)

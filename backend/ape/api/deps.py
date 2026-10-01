# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared request plumbing: sessions, lookups, and the open-photo cache.

The one thing here that is more than plumbing is :class:`PreviewCache`. Section
10 asks for a slider to redraw in under 150 ms and says how: render at 1024 px
while the slider is being dragged, at 2048 px when it is released, never from
the full-resolution RAW. Neither size is reachable if every request decodes the
file again -- that alone is a second.

So the cache holds, per photo, one decoded frame *per size*, each behind a
``StageRenderer`` that resumes from the last stage whose parameters did not
change. The 1024 px source is derived from the 2048 px one by halving it, which
is exactly the relationship test 1 of section 13 measures: the two agree to
within a twentieth of a dE2000, so what the user sees while dragging is what
they get when they let go.

Rendering the drag at 1024 *throughout* rather than resizing at the end is the
whole point. A quarter of the pixels through every stage is four times less
arithmetic, and a quarter of the memory in the stage cache -- measured below the
eighty milliseconds of a comfortable slider instead of above the hundred and
fifty of an uncomfortable one.

The cache is small on purpose, and it empties itself. Section 26 gives the
server 400 MB at rest and forbids background timers, so entries are dropped by
age on the next access rather than by a reaper thread: a photo nobody has looked
at for five minutes costs nothing but the second it takes to decode again. Three
sources -- one photo at both sizes, plus the drag size of the one before it --
is what a person actually keeps in play.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from ..db.models import JobKind, Photo, Project
from ..db.session import get_sessionmaker
from ..pipeline.render import StageRenderer

__all__ = [
    "PreviewCache",
    "get_photo",
    "get_project",
    "get_session",
    "preview_cache",
    "session_factory",
]

#: How many decoded sources stay open at once, counting every size of each.
_MAX_OPEN_SOURCES = 3

#: An entry untouched for this long is dropped at the next access. The same five
#: minutes section 26 gives the ONNX models, for the same reason: memory held
#: for a photo nobody is looking at is memory taken from the workers.
_MAX_IDLE_SECONDS = 300.0


def session_factory() -> sessionmaker[Session]:
    return get_sessionmaker()


def get_session() -> Iterator[Session]:
    """A session for one request, committed on the way out.

    FastAPI closes the generator after the response has been sent, so a handler
    may return ORM objects as long as the schema reads them eagerly -- which is
    why the sessions are made with ``expire_on_commit=False``.
    """
    maker = session_factory()
    session = maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_project(project_id: int, session: Session = Depends(get_session)) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"progetto {project_id} inesistente")
    return project


def get_photo(photo_id: int, session: Session = Depends(get_session)) -> Photo:
    photo = session.get(Photo, photo_id)
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"foto {photo_id} inesistente")
    return photo


def pixels_for_editing(session: Session, photo: Photo) -> Path:
    """The file the editor decodes for this photo.

    A merged photo whose intermediate the cache lost is not rebuilt here: a
    merge takes seconds to minutes and the server must stay light (section
    26). A worker rebuilds it -- the proxy job does, as a side effect -- and
    the editor asks again.

    Raises:
        HTTPException 409: the photo has no pixels at all.
        HTTPException 503: the merge is being rebuilt.
    """
    from ..jobs.queue import enqueue
    from ..raw.intermediate import is_intermediate
    from ..raw.source import pixels_of

    source = pixels_of(photo)
    if source is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "questa foto non ha un file sorgente da sviluppare"
        )
    if is_intermediate(source) and not source.is_file():
        enqueue(
            session, JobKind.PROXY, {"photo_id": photo.id, "force": True},
            project_id=photo.project_id, dedupe_key=f"proxy:{photo.id}",
        )
        # Committed here: the exception below would roll the request back.
        session.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "la fusione si sta rigenerando: la cache l'aveva rimossa. Riprova fra poco.",
        )
    return source


@dataclass
class _Entry:
    renderer: StageRenderer
    source: Path
    mtime: float
    touched: float
    lens_override: tuple[str, str] | None = None


class PreviewCache:
    """The photos currently open in the editor, keyed by ``(photo, size)``.

    Thread-safe because uvicorn runs the synchronous handlers on a thread pool:
    two requests for the same photo can arrive at once, and both would otherwise
    decode it.
    """

    def __init__(self, capacity: int = _MAX_OPEN_SOURCES) -> None:
        self._entries: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def renderer_for(
        self,
        photo_id: int,
        source: str | Path,
        long_edge: int | None = None,
        lens_override: tuple[str, str] | None = None,
    ) -> StageRenderer:
        """The renderer for one photo at one size, decoding it if it is not open.

        A source file that changed on disk since it was decoded invalidates every
        size of it: the user may have replaced the file, and showing them the old
        pixels would be worse than the wait. So does a change of the lens profile
        the user associated by hand, which the decoded frame carries.
        """
        path = Path(source)
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise HTTPException(
                status.HTTP_410_GONE, f"{path.name} non è più leggibile: {exc.strerror or exc}"
            ) from exc

        # Above the proxy size there is nothing more to show: the editing
        # source *is* the proxy, and asking for 4000 px would decode a frame the
        # interface has no way to display (section 26).
        proxy_edge = get_settings().proxy_long_edge
        edge = min(int(long_edge or proxy_edge), proxy_edge)
        key = (photo_id, edge)

        now = time.monotonic()
        with self._lock:
            self._evict_idle(now)
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.source == path
                and entry.mtime == mtime
                and entry.lens_override == lens_override
            ):
                entry.touched = now
                self._entries.move_to_end(key)
                return entry.renderer
            if entry is not None:
                # The file changed: nothing decoded from it is worth keeping.
                self._drop_photo(photo_id)

        # Decoding takes about a second; doing it outside the lock lets a
        # request for a different photo through in the meantime.
        decoded = self._decode(photo_id, path, mtime, edge, lens_override)
        renderer = StageRenderer(decoded=decoded)

        with self._lock:
            self._entries[key] = _Entry(
                renderer=renderer,
                source=path,
                mtime=mtime,
                touched=time.monotonic(),
                lens_override=lens_override,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return renderer

    def _decode(
        self,
        photo_id: int,
        path: Path,
        mtime: float,
        edge: int,
        lens_override: tuple[str, str] | None,
    ):
        """The frame for one size, reusing a larger one already in the cache.

        Deriving the drag-sized source from the release-sized one, rather than
        decoding twice, is both cheaper and *more correct*: the two renders then
        stand in exactly the proxy-to-export relationship that test 1 measures.
        """
        from ..pipeline.filters import resize_long_edge
        from ..raw.decode import DecodedRaw
        from ..raw.proxy import editing_proxy

        with self._lock:
            larger = [
                entry
                for (other_id, other_edge), entry in self._entries.items()
                if other_id == photo_id
                and other_edge > edge
                and entry.source == path
                and entry.mtime == mtime
                and entry.lens_override == lens_override
            ]
        if larger:
            base = larger[0].renderer.decoded
            reduced = resize_long_edge(base.rgb, edge)
            return DecodedRaw(
                rgb=np.ascontiguousarray(reduced),
                camera=base.camera,
                baseline_exposure_ev=base.baseline_exposure_ev,
                source_path=base.source_path,
                half_size=base.half_size,
                raw_metadata=base.raw_metadata,
                lens=base.lens,
            )
        return editing_proxy(path, long_edge=edge, lens_override=lens_override)

    def _evict_idle(self, now: float) -> None:
        """Drop what nobody has looked at in a while. The caller holds the lock."""
        for key, entry in list(self._entries.items()):
            if now - entry.touched > _MAX_IDLE_SECONDS:
                del self._entries[key]

    def _drop_photo(self, photo_id: int) -> None:
        """Remove every size of one photo. The caller holds the lock."""
        for key in [key for key in self._entries if key[0] == photo_id]:
            del self._entries[key]

    def forget(self, photo_id: int) -> None:
        with self._lock:
            self._drop_photo(photo_id)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def cached_bytes(self) -> int:
        """Total memory held, for the diagnostics bundle of section 19."""
        with self._lock:
            return sum(
                entry.renderer.decoded.rgb.nbytes + entry.renderer.cached_bytes()
                for entry in self._entries.values()
            )

    def __len__(self) -> int:
        return len(self._entries)


#: One per server process.
preview_cache = PreviewCache()

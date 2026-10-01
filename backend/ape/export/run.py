# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exporting one photo: develop at full resolution, tag, write, add its sidecars.

Pure work, no catalogue: ``jobs/handlers_export.py`` reads what is needed into
an :class:`ExportWork`, calls :func:`export_one` with no session open, and
records the :class:`ExportResult`. That is the rule of every handler (a
session held across a two-second render is a pool serialised on SQLite), and it
is also what lets the command line and the tests run an export without a
catalogue at all.

Failures come in two kinds and are reported differently, because they are
about different things:

* the **photo** cannot be developed -- the RAW is unreadable or gone. The photo
  itself is marked ``failed`` (test 11), since every later attempt will fail the
  same way;
* the **destination** refuses the file -- disk full, no permission, a folder
  that vanished. The photo is fine; the item fails, and "Riprova" after freeing
  space is the whole remedy.

Either way the error is a sentence in Italian, and the batch goes on.
"""

from __future__ import annotations

import errno
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..db.enums import ExportConflict
from ..pipeline.params import EditParams
from ..safety import SourceWriteError, create_beside_source
from .image import SOFTWARE_TAG, encode_image
from .naming import WriteOutcome, write_exclusive
from .settings import SHARPENING_AMOUNT, ExportSettings
from .sidecar import SidecarContext, build_sidecar, context_from_decoded, sidecar_name

__all__ = ["ExportFailure", "ExportResult", "ExportWork", "export_one"]

_log = logging.getLogger(__name__)


class ExportFailure(Exception):
    """An export that did not happen, with a message for the user.

    ``photo_fault`` separates "this RAW cannot be developed" from "this folder
    cannot be written to" (module docstring).
    """

    def __init__(self, message: str, *, photo_fault: bool) -> None:
        super().__init__(message)
        self.photo_fault = photo_fault


@dataclass(slots=True)
class ExportWork:
    """Everything one photo's export needs, read from the catalogue beforehand."""

    source: Path
    filename: str
    name: str
    params: EditParams
    settings: ExportSettings
    policy: ExportConflict
    #: The folder of section 2 that must not be written to.
    protected: Any = None
    #: Other names of the batch, which a rename must not take.
    reserved: tuple[str, ...] = ()
    lens_override: tuple[str, str] | None = None
    artist: str | None = None
    copyright: str | None = None
    shot_at: datetime | None = None
    orientation: int | None = None
    #: False for a merged photo: an XMP sidecar describes a development of a
    #: RAW, and a merge has none to describe.
    raw_file: bool = True
    #: What the photo's pixels are (``retouch.fills.photo_token``): part of
    #: the key of every eraser's fill.
    photo_token: str = ""


@dataclass(slots=True)
class ExportResult:
    image: WriteOutcome | None = None
    #: ``[{"kind", "path", "outcome"}]``, the shape of ``ExportItem.sidecars``.
    sidecars: list[dict[str, Any]] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


def _destination_error(exc: OSError, folder: Path) -> ExportFailure:
    if exc.errno == errno.ENOSPC:
        text = f"spazio esaurito in {folder}"
    elif exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        text = f"non ho il permesso di scrivere in {folder}"
    elif exc.errno == errno.ENOENT:
        text = f"la cartella di destinazione {folder} non esiste più"
    else:
        text = f"scrittura non riuscita in {folder}: {exc.strerror or exc}"
    return ExportFailure(text, photo_fault=False)


def _with_fills(work: ExportWork, decoded: Any, timings: dict[str, float]) -> EditParams:
    """The parameters with every eraser's fill in place, made here if missing.

    Never exported silently without a removal (docs/SPEC_rimozione.md 4.3): a
    fill the editor did not make, or made for a lens profile since changed,
    is computed now on the proxy -- the same function, the same seed, so the
    same fill the editor would have shown. An engine that cannot run fails
    the photo, with the reason, in the Problems panel.
    """
    from ..retouch import fills

    params = work.params
    if not any(item.kind == "erase" for item in params.retouch):
        return params
    base = fills.upstream_base(work.photo_token, fills.lens_key(params, decoded))
    keys = fills.item_keys(params, base)
    if fills.missing(keys):
        from ..raw.proxy import editing_proxy

        started = time.perf_counter()
        proxy = editing_proxy(work.source, lens_override=work.lens_override)
        try:
            fills.compute_fills(proxy, params, base)
        except fills.FillUnavailable as exc:
            raise ExportFailure(f"rimozione non calcolabile: {exc}", photo_fault=True) from exc
        del proxy
        if any(getattr(item, "engine", None) == "ml" for item in params.retouch):
            from ..retouch import ml

            # Before the full-resolution render, whose peak is the worker's:
            # the model's half a gigabyte is not needed for it.
            ml.release_if_idle(force=True)
        timings["retouch"] = time.perf_counter() - started
    return fills.resolve(params, keys)


def _develop(
    work: ExportWork, timings: dict[str, float]
) -> tuple[bytes, Any, tuple[int, int]]:
    """Decode, render, encode and tag.

    Returns the file bytes, the decoded frame's context (its pixels already
    released) and the developed frame's size before geometry, for the sidecars.
    """
    from ..pipeline.render import RenderOptions, render
    from ..raw.decode import decode_linear
    from ..raw.metadata import read_metadata
    from .metadata import ExportMetadata, embed_metadata

    settings = work.settings
    started = time.perf_counter()
    try:
        decoded = decode_linear(work.source, lens_override=work.lens_override)
    except FileNotFoundError as exc:
        raise ExportFailure(
            f"{work.filename} non è più nella cartella sorgente", photo_fault=True
        ) from exc
    except ValueError as exc:
        raise ExportFailure(str(exc), photo_fault=True) from exc
    except MemoryError as exc:
        raise ExportFailure(
            "memoria insufficiente per sviluppare la foto a piena risoluzione",
            photo_fault=False,
        ) from exc
    timings["decode"] = time.perf_counter() - started
    params = _with_fills(work, decoded, timings)

    options = RenderOptions(
        output_space=settings.output_space,
        long_edge=settings.long_edge,
        output_sharpening=SHARPENING_AMOUNT[settings.sharpening],
    )
    started = time.perf_counter()
    # The decoded frame is not needed once the render has started: the sidecars
    # need only its camera and lens context, which ``decoded`` keeps.
    frame = (decoded.rgb.shape[1], decoded.rgb.shape[0])
    image = render(decoded, params, options, consume=True)
    timings["render"] = time.perf_counter() - started

    started = time.perf_counter()
    payload = encode_image(
        image,
        settings.format,
        output_space=settings.output_space,
        quality=settings.quality,
        dither=settings.dither,
    )
    height, width = image.shape[:2]
    del image
    meta = ExportMetadata(
        source_exif=read_metadata(work.source).raw_tags,
        width=width,
        height=height,
        output_space=settings.output_space,
        software=SOFTWARE_TAG,
        artist=work.artist,
        copyright=work.copyright,
        strip_gps=settings.strip_gps,
        written_at=datetime.now(),
    )
    payload = embed_metadata(payload, meta)
    timings["encode"] = time.perf_counter() - started
    return payload, decoded, frame


def _write_sidecars(
    work: ExportWork, ctx: SidecarContext, folder: Path, result: ExportResult
) -> None:
    settings = work.settings
    wanted = (("darktable", settings.xmp_darktable), ("adobe", settings.xmp_adobe))
    kinds = [kind for kind, on in wanted if on]
    for kind in kinds:
        payload = build_sidecar(kind, work.params, ctx)
        name = sidecar_name(kind, work.filename)
        written = write_exclusive(folder, name, payload, work.policy, source=work.protected)
        entry: dict[str, Any] = {
            "kind": kind,
            "path": str(written.path) if written.path else None,
            "outcome": written.outcome,
        }
        if settings.xmp_beside_raw:
            try:
                created = create_beside_source(work.source, name, payload)
            except (SourceWriteError, OSError) as exc:
                entry["beside_raw"] = f"non scritto: {exc}"
            else:
                # An existing sidecar next to the RAW is the user's own edit
                # from another program: left as it was, and said so.
                entry["beside_raw"] = "written" if created else "exists"
        result.sidecars.append(entry)


def export_one(work: ExportWork) -> ExportResult:
    """Export one photo as ``work`` describes. The catalogue is not touched.

    Raises:
        ExportFailure: with a readable message; ``photo_fault`` tells the caller
            whether to mark the photo itself as failed.
    """
    settings = work.settings
    if not settings.output_dir:
        raise ExportFailure("nessuna cartella di destinazione scelta", photo_fault=False)
    folder = Path(settings.output_dir).expanduser()
    result = ExportResult()
    ctx_fields: dict[str, Any] = {"shot_at": work.shot_at, "software": SOFTWARE_TAG}
    if work.orientation:
        ctx_fields["orientation"] = work.orientation

    try:
        folder.mkdir(parents=True, exist_ok=True)
        decoded = None
        target = folder / work.name
        skip_image = work.policy is ExportConflict.SKIP and target.exists()
        if settings.images and not skip_image:
            payload, decoded, frame = _develop(work, result.seconds)
            started = time.perf_counter()
            result.image = write_exclusive(
                folder,
                work.name,
                payload,
                work.policy,
                source=work.protected,
                reserved=work.reserved,
            )
            result.seconds["write"] = time.perf_counter() - started
            del payload
        elif settings.images:
            result.image = WriteOutcome(None, "skipped")

        if (settings.xmp_darktable or settings.xmp_adobe) and work.raw_file:
            if decoded is not None:
                ctx = context_from_decoded(decoded, work.filename, **ctx_fields)
            else:
                from .sidecar import read_sidecar_context

                try:
                    ctx = read_sidecar_context(
                        work.source, lens_override=work.lens_override, **ctx_fields
                    )
                except FileNotFoundError as exc:
                    raise ExportFailure(
                        f"{work.filename} non è più nella cartella sorgente", photo_fault=True
                    ) from exc
                except ValueError as exc:
                    raise ExportFailure(str(exc), photo_fault=True) from exc
            if decoded is not None:
                ctx.width, ctx.height = frame
            del decoded
            _write_sidecars(work, ctx, folder, result)
    except SourceWriteError as exc:
        raise ExportFailure(str(exc), photo_fault=False) from exc
    except OSError as exc:
        raise _destination_error(exc, folder) from exc
    return result

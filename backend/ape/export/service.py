# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export batches: plan and start (phase 8). Pausing and the rest: ``batch.py``.

**Plan before queue.** Everything that can be known before a single photo is
developed is checked first, because a batch of a thousand photos is the wrong
place to discover a mistake at photo 612: the destination (set, reachable, not
the source folder -- section 2.4), the template (every name valid and distinct
-- section 16.1), and the files that already exist there (section 16.2). The
plan is what the Export screen shows before the user presses the button, and
starting a batch recomputes it rather than trusting the screen.

**Collisions are asked about up front.** With the policy "chiedi", a plan that
finds existing files refuses to start until each colliding photo has an answer,
or one answer "for all the rest" -- which is then remembered on the project, as
section 16.2 says. A file that appears *while* the batch runs, which nobody
could have asked about, is renamed: of the three choices it is the only one
that cannot lose anything.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..db.models import (
    EditVersion,
    ExportBatch,
    ExportConflict,
    ExportItem,
    Job,
    JobKind,
    JobState,
    Photo,
    PhotoStatus,
    Project,
)
from ..db.workset import editing_set
from ..jobs import queue as jobq
from ..raw.source import has_raw_file
from ..safety import is_inside
from .naming import NameContext, TemplateError, existing_collisions, render_names
from .settings import ExportSettings, store_settings
from .sidecar import sidecar_name

__all__ = ["ConflictsPending", "ExportPlan", "PlanError", "plan", "selection", "start"]

#: Photos listed by name in a plan: enough for the live preview of section 16.1
#: and the first collision dialog, not a thousand rows over the wire.
_SAMPLE_NAMES = 5


class PlanError(ValueError):
    """The batch cannot start as configured. The message is for the user."""


class ConflictsPending(PlanError):
    """Files already exist and the policy is "ask": the user has to choose."""

    def __init__(self, conflicts: list[dict[str, Any]]) -> None:
        super().__init__(f"{len(conflicts)} foto hanno già un file con lo stesso nome")
        self.conflicts = conflicts


@dataclass
class ExportPlan:
    photos: list[Photo]
    versions: dict[int, EditVersion]
    names: list[str]
    #: ``[{"photo_id", "filename", "files": [names already present]}]``
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "count": len(self.photos),
            "names": [
                {"photo_id": p.id, "filename": p.filename, "name": n}
                for p, n in list(zip(self.photos, self.names, strict=True))[:_SAMPLE_NAMES]
            ],
            "conflicts": self.conflicts,
            "warnings": self.warnings,
        }


def selection(session: Session, project: Project, settings: ExportSettings) -> list[Photo]:
    """The photos a batch exports: kept, present, readable -- in shooting order.

    "Kept" is every photo the culling did not discard (the user's choice at
    the start of phase 8); ``only_approved`` narrows it to the ones the review
    approved. The order is the one ``{counter}`` numbers.
    """
    conditions = [
        *editing_set(project.id),
        or_(Photo.path.is_not(None), Photo.intermediate_path.is_not(None)),
        Photo.status != PhotoStatus.FAILED,
    ]
    if settings.only_approved:
        conditions.append(Photo.status == PhotoStatus.APPROVED)
    return list(
        session.scalars(
            select(Photo)
            .where(*conditions)
            .order_by(Photo.shot_at.is_(None), Photo.shot_at, Photo.filename, Photo.id)
        ).all()
    )


def _versions(session: Session, photos: list[Photo]) -> dict[int, EditVersion]:
    ids = [p.id for p in photos]
    if not ids:
        return {}
    rows = session.scalars(
        select(EditVersion).where(EditVersion.photo_id.in_(ids), EditVersion.is_current.is_(True))
    ).all()
    return {row.photo_id: row for row in rows}


def _contexts(project: Project, photos: list[Photo], settings: ExportSettings) -> list[NameContext]:
    return [
        NameContext(
            filename=photo.filename,
            ext=settings.extension,
            counter=index,
            project=project.name,
            camera=photo.camera,
            lens=photo.lens,
            iso=photo.iso,
            focal_length=photo.focal_length,
            shot_at=photo.shot_at,
        )
        for index, photo in enumerate(photos, start=1)
    ]


def _files_of(photo: Photo, name: str, settings: ExportSettings) -> list[str]:
    """Every file a photo's export writes into the destination."""
    files = [name] if settings.images else []
    if not has_raw_file(photo):
        return files  # a merge has no RAW for a sidecar to describe
    if settings.xmp_darktable:
        files.append(sidecar_name("darktable", photo.filename))
    if settings.xmp_adobe:
        files.append(sidecar_name("adobe", photo.filename))
    return files


def _check_destination(project: Project, settings: ExportSettings) -> Path:
    if not settings.output_dir:
        raise PlanError("scegli la cartella di destinazione")
    folder = Path(settings.output_dir).expanduser()
    if is_inside(folder, project.source_dir):
        raise PlanError(
            "la cartella di destinazione non può essere la cartella dei RAW né stare "
            "dentro di essa: i file originali non vengono mai toccati"
        )
    if folder.exists() and not folder.is_dir():
        raise PlanError(f"{folder} esiste ma non è una cartella")
    return folder


def _pending_edits(session: Session, project: Project) -> int:
    """Predictions and analyses still queued: the edits they write are not in yet."""
    return int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project.id,
                Job.kind.in_((JobKind.PREDICT, JobKind.ANALYZE, JobKind.PROXY)),
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )


def plan(session: Session, project: Project, settings: ExportSettings) -> ExportPlan:
    """Check a batch without starting it.

    Raises:
        PlanError: no destination, a destination inside the source folder, an
            invalid template or duplicate names, nothing to export.
    """
    if not settings.writes_anything:
        raise PlanError("non c'è niente da esportare: scegli le immagini o almeno un sidecar")
    folder = _check_destination(project, settings)
    photos = selection(session, project, settings)
    if not photos:
        raise PlanError(
            "nessuna foto approvata da esportare"
            if settings.only_approved
            else "nessuna foto da esportare in questo progetto"
        )
    try:
        names = render_names(settings.template, _contexts(project, photos, settings))
    except TemplateError as exc:
        raise PlanError(str(exc)) from exc

    result = ExportPlan(photos=photos, versions=_versions(session, photos), names=names)
    everything = [_files_of(p, n, settings) for p, n in zip(photos, names, strict=True)]
    present = set(existing_collisions(folder, [f for files in everything for f in files]))
    for photo, files in zip(photos, everything, strict=True):
        clashing = [f for f in files if f in present]
        if clashing:
            result.conflicts.append(
                {"photo_id": photo.id, "filename": photo.filename, "files": clashing}
            )

    pending = _pending_edits(session, project)
    if pending:
        result.warnings.append(
            f"{pending} analisi o predizioni sono ancora in coda: alcune foto verrebbero "
            "esportate con lo sviluppo di adesso, non con quello in arrivo"
        )
    in_review = sum(p.status is PhotoStatus.NEEDS_REVIEW for p in photos)
    if in_review:
        result.warnings.append(f"{in_review} foto sono ancora in coda di revisione")
    unedited = sum(p.id not in result.versions for p in photos)
    if unedited:
        result.warnings.append(f"{unedited} foto non hanno ancora uno sviluppo: usciranno neutre")
    if settings.xmp_darktable or settings.xmp_adobe:
        merged = sum(not has_raw_file(p) for p in photos)
        if merged:
            result.warnings.append(
                f"{merged} foto sono fusioni: esportate come immagini, senza XMP, "
                "perché non hanno un RAW da descrivere"
            )
        # Neither sidecar has a mask of ours to write (xmp_darktable.py,
        # xmp_adobe.py): the photo opens there with the global edit only.
        masked = sum(
            bool((version.params or {}).get("masks")) for version in result.versions.values()
        )
        if masked:
            result.warnings.append(
                f"{masked} foto hanno maschere: le immagini esportate le contengono, "
                "gli XMP per darktable e Lightroom no"
            )
        # Nor a removal (docs/SPEC_rimozione.md 7): no translation is attempted.
        retouched = sum(
            bool((version.params or {}).get("retouch")) for version in result.versions.values()
        )
        if retouched:
            result.warnings.append(
                f"{retouched} foto hanno rimozioni: le immagini esportate le contengono, "
                "gli XMP no"
            )
    return result


def start(
    session: Session,
    project: Project,
    settings: ExportSettings,
    *,
    decisions: dict[int, ExportConflict] | None = None,
    apply_to_all: ExportConflict | None = None,
) -> ExportBatch:
    """Freeze the settings, create the items and queue one job per photo.

    Args:
        decisions: the user's answer for single colliding photos.
        apply_to_all: an answer for every collision without one, which also
            becomes the project's policy (section 16.2).

    Raises:
        PlanError: as :func:`plan`.
        ConflictsPending: the policy is "ask" and some collisions have no answer.
    """
    if apply_to_all is ExportConflict.ASK:
        apply_to_all = None
    if apply_to_all is not None:
        settings = settings.model_copy(update={"on_conflict": apply_to_all})
    result = plan(session, project, settings)
    decisions = dict(decisions or {})
    if settings.on_conflict is ExportConflict.ASK:
        unanswered = [c for c in result.conflicts if c["photo_id"] not in decisions]
        if unanswered:
            raise ConflictsPending(unanswered)
    store_settings(project, settings)

    batch = ExportBatch(
        project_id=project.id, settings=settings.model_dump(mode="json"), total=len(result.photos)
    )
    session.add(batch)
    session.flush()
    for position, (photo, name) in enumerate(zip(result.photos, result.names, strict=True), 1):
        version = result.versions.get(photo.id)
        item = ExportItem(
            batch_id=batch.id,
            photo_id=photo.id,
            version_id=version.id if version else None,
            position=position,
            name=name,
            on_conflict=decisions.get(photo.id),
        )
        session.add(item)
        session.flush()
        job = jobq.enqueue(
            session,
            JobKind.EXPORT,
            {"item_id": item.id},
            project_id=project.id,
            dedupe_key=f"export:{batch.id}:{item.id}",
        )
        item.job_id = job.id if job else None
    return batch

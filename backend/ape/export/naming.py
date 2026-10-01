# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""File names of the exports: the template of section 16.1, the collisions of 16.2.

A template is text with tokens in braces -- ``{basename}_{date:%Y%m%d}.{ext}``.
It is rendered once per photo *before* the queue starts, for every photo of the
batch, because two of its failure modes are only visible across the whole set:
an empty result, and two different photos landing on the same name. Both are
refused up front (section 16.1): finding out at photo 612 that photo 611 was
overwritten by it is exactly what "mai una sovrascrittura silenziosa" forbids.

Names are made safe for the destination rather than for this machine. The export
folder is often an SD card or a disk shared with Windows, formatted exFAT, whose
forbidden characters and case-insensitivity are stricter than ext4's; a name
that is legal here and silently mangled there is a name that collides there.
So the forbidden set is the union, and duplicates are compared case-folded.

The collision with a file *already in the folder* is decided at write time by
:func:`write_exclusive`, which never replaces a file it was not told to replace:
the final name is taken with a hard link (atomic, and it fails if the name
exists), so two workers -- or another program -- racing for ``foto_1.jpg``
cannot both get it.
"""

from __future__ import annotations

import errno
import os
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..db.enums import ExportConflict
from ..safety import assert_outside_source, guarded_open

__all__ = [
    "DEFAULT_TEMPLATE",
    "NameContext",
    "TemplateError",
    "WriteOutcome",
    "existing_collisions",
    "parse_template",
    "render_name",
    "render_names",
    "sanitise",
    "write_exclusive",
]

DEFAULT_TEMPLATE = "{basename}.{ext}"

#: Every token of section 16.1, and nothing else: an unknown token is a typo,
#: and a typo that renders as nothing produces duplicate names later.
TOKENS = (
    "basename",
    "ext",
    "counter",
    "seq",
    "date",
    "time",
    "project",
    "camera",
    "lens",
    "iso",
    "focal",
)

_TOKEN = re.compile(r"\{([a-z]+)(?::([^{}]*))?\}")

#: Forbidden on at least one of ext4, exFAT/FAT32 and NTFS -- the filesystems an
#: export folder is realistically on. Control characters are forbidden on the
#: last two and invisible everywhere.
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')

#: Device names Windows refuses as a file name, whatever the extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)
}

#: Bytes, not characters: ext4 and exFAT both cap a name at 255 bytes of UTF-8.
_MAX_NAME_BYTES = 255

#: Suffix of the file an export is written to before it takes its real name.
PART_SUFFIX = ".ape-part"


class TemplateError(ValueError):
    """A template that cannot produce valid, distinct names. Message in Italian."""


@dataclass(slots=True)
class NameContext:
    """What a template may refer to, for one photo."""

    filename: str
    ext: str
    counter: int
    project: str = ""
    camera: str | None = None
    lens: str | None = None
    iso: int | None = None
    focal_length: float | None = None
    shot_at: datetime | None = None


def parse_template(template: str) -> list[tuple[str, str | None]]:
    """The tokens of a template, validated. Literal text in between is kept as is.

    Raises:
        TemplateError: empty template, unknown token, unbalanced brace, or a
            ``{counter}`` / ``{date}`` format that does not format.
    """
    if not template or not template.strip():
        raise TemplateError("il modello del nome è vuoto")
    tokens = [(m.group(1), m.group(2)) for m in _TOKEN.finditer(template)]
    leftover = _TOKEN.sub("", template)
    if "{" in leftover or "}" in leftover:
        raise TemplateError(f"parentesi graffe non chiuse nel modello «{template}»")
    for name, spec in tokens:
        if name not in TOKENS:
            raise TemplateError(
                f"«{{{name}}}» non è un segnaposto valido; quelli disponibili sono "
                + ", ".join(f"{{{t}}}" for t in TOKENS)
            )
        if name == "counter" and spec:
            try:
                format(1, spec)
            except ValueError as exc:
                raise TemplateError(f"formato del contatore non valido: «{spec}»") from exc
        if name in ("date", "time") and spec:
            try:
                datetime(2026, 1, 2, 3, 4, 5).strftime(spec)
            except ValueError as exc:
                raise TemplateError(f"formato della data non valido: «{spec}»") from exc
    return tokens


def _sequence(filename: str) -> str:
    """The camera's own frame number: the last run of digits in the file name.

    ``DSC06312.ARW`` gives ``06312``. Sony counts to 9999 and then changes the
    prefix, so the digits alone are what identifies a frame within a card.
    """
    runs = re.findall(r"\d+", Path(filename).stem)
    return runs[-1] if runs else ""


def _focal(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.0f}mm" if abs(value - round(value)) < 0.05 else f"{value:.1f}mm"


def _value(name: str, spec: str | None, ctx: NameContext) -> str:
    if name == "basename":
        return Path(ctx.filename).stem
    if name == "ext":
        return ctx.ext
    if name == "counter":
        return format(ctx.counter, spec or "")
    if name == "seq":
        return _sequence(ctx.filename)
    if name in ("date", "time"):
        if ctx.shot_at is None:
            return ""
        default = "%Y%m%d" if name == "date" else "%H%M%S"
        return ctx.shot_at.strftime(spec or default)
    if name == "project":
        return ctx.project
    if name == "camera":
        return ctx.camera or ""
    if name == "lens":
        return ctx.lens or ""
    if name == "iso":
        return "" if ctx.iso is None else str(ctx.iso)
    if name == "focal":
        return _focal(ctx.focal_length)
    raise TemplateError(f"segnaposto sconosciuto {{{name}}}")  # parse_template stops this


def sanitise(name: str) -> str:
    """Make one file name legal on every filesystem an export may land on.

    Forbidden characters become ``_``; trailing dots and spaces (which Windows
    strips, making ``foto.`` and ``foto`` the same file) are removed; a reserved
    device name gains a leading ``_``; the result is cut to 255 bytes keeping
    the extension. Unicode is normalised to NFC, so an accented letter typed on
    two different machines is one name, not two.
    """
    text = unicodedata.normalize("NFC", name)
    text = _FORBIDDEN.sub("_", text).strip().rstrip(". ")
    stem, dot, suffix = text.rpartition(".")
    if not dot:
        stem, suffix = text, ""
    if stem.split(".")[0].upper() in _RESERVED:
        stem = "_" + stem
    ending = f".{suffix}" if suffix else ""
    budget = _MAX_NAME_BYTES - len(ending.encode("utf-8"))
    encoded = stem.encode("utf-8")
    if len(encoded) > budget:
        stem = encoded[:budget].decode("utf-8", errors="ignore").rstrip(". ")
    return f"{stem}{ending}"


def render_name(template: str, ctx: NameContext) -> str:
    """One photo's export name. The output extension is appended if the template forgets it.

    Raises:
        TemplateError: invalid template, or a name that comes out empty.
    """
    parse_template(template)
    rendered = _TOKEN.sub(lambda m: _value(m.group(1), m.group(2), ctx), template)
    # Path separators are not a way to make sub-folders: they would let a
    # template climb out of the export folder.
    rendered = rendered.replace("/", "_").replace("\\", "_")
    if not rendered.lower().endswith(f".{ctx.ext.lower()}"):
        rendered = f"{rendered}.{ctx.ext}"
    name = sanitise(rendered)
    if not Path(name).stem.strip("._ "):
        raise TemplateError(
            f"il modello «{template}» produce un nome vuoto per {ctx.filename}"
        )
    return name


def render_names(template: str, contexts: Sequence[NameContext]) -> list[str]:
    """Every name of a batch, refused if two photos would share one.

    Raises:
        TemplateError: invalid template, an empty name, or a duplicate. The
            message names the two photos, so the user sees which token to add.
    """
    parse_template(template)
    names: list[str] = []
    seen: dict[str, str] = {}
    for ctx in contexts:
        name = render_name(template, ctx)
        key = name.casefold()
        if key in seen:
            raise TemplateError(
                f"il modello «{template}» dà lo stesso nome «{name}» a {seen[key]} e a "
                f"{ctx.filename}: aggiungi un segnaposto che li distingua, per esempio "
                "{counter:03} o {basename}"
            )
        seen[key] = ctx.filename
        names.append(name)
    return names


def existing_collisions(directory: str | Path, names: Iterable[str]) -> list[str]:
    """The names of a batch that already exist in the destination folder."""
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        return []
    present = {entry.name.casefold() for entry in folder.iterdir()}
    return [name for name in names if name.casefold() in present]


@dataclass(slots=True)
class WriteOutcome:
    """Where a file went, and why there."""

    path: Path | None
    #: ``written`` | ``renamed`` | ``overwritten`` | ``skipped``
    outcome: str


def _renamed(name: str, attempt: int) -> str:
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return f"{name}_{attempt}"
    return f"{stem}_{attempt}.{suffix}"


def _link_or_rename(temporary: Path, target: Path) -> bool:
    """Give ``temporary`` the name ``target`` unless it exists. True on success.

    A hard link is the one operation that is both atomic and refuses an existing
    name. FAT and exFAT have no hard links; there the check-then-rename leaves a
    window only against *other programs* writing the same name in the same
    instant, since the names of one batch are distinct by construction.
    """
    try:
        os.link(temporary, target)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
            raise
        if target.exists():
            return False
        os.rename(temporary, target)
        return True
    temporary.unlink(missing_ok=True)
    return True


def write_exclusive(
    directory: str | Path,
    name: str,
    payload: bytes,
    policy: ExportConflict,
    *,
    source: object = None,
    reserved: Iterable[str] = (),
) -> WriteOutcome:
    """Write ``payload`` under ``name`` in ``directory``, honouring the collision policy.

    The bytes are first written in full to a hidden temporary file next to the
    destination, then given their name: a crash leaves a ``.ape-part`` file, never
    a truncated photo under a real name.

    Args:
        directory: the export folder. Refused if inside a source folder.
        name: the rendered, sanitised name.
        payload: complete file contents.
        policy: ``rename``, ``overwrite`` or ``skip``. ``ask`` must have been
            resolved by the caller; if it reaches here it behaves as ``rename``,
            which is the only choice that can never lose a file.
        source: project or folders the guard of section 2 protects.
        reserved: other names of the same batch, which a rename must not take.

    Returns:
        The final path and what happened.
    """
    folder = Path(directory).expanduser()
    target = folder / name
    assert_outside_source(target, source)
    if policy is ExportConflict.SKIP and target.exists():
        return WriteOutcome(None, "skipped")

    temporary = folder / f".{name}.{os.getpid()}{PART_SUFFIX}"
    with guarded_open(temporary, "wb", source=source) as handle:
        handle.write(payload)
    try:
        if policy is ExportConflict.OVERWRITE:
            existed = target.exists()
            os.replace(temporary, assert_outside_source(target, source))
            return WriteOutcome(target, "overwritten" if existed else "written")
        if _link_or_rename(temporary, target):
            return WriteOutcome(target, "written")
        if policy is ExportConflict.SKIP:
            return WriteOutcome(None, "skipped")
        taken = {n.casefold() for n in reserved}
        attempt = 1
        while True:
            candidate = folder / _renamed(name, attempt)
            attempt += 1
            if candidate.name.casefold() in taken:
                continue
            if _link_or_rename(temporary, assert_outside_source(candidate, source)):
                return WriteOutcome(candidate, "renamed")
    finally:
        temporary.unlink(missing_ok=True)

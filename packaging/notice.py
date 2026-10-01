# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The dependency list of NOTICE.md, generated and checked (docs/SPEC.md section 24).

Two lists, because two things are distributed: the **Python packages** the
program imports at run time (the project's dependencies and extras, followed
through their own requirements, as installed in the environment), and the
**frontend packages** that end up in the built bundle (the non-development
entries of ``frontend/package-lock.json``, plus the Workbox runtime that the
service worker inlines).

Every licence is read from the package's own metadata and normalised to SPDX.
A distributed package must be compatible with GPL-3.0; a development tool
(tests, linters, the bundler) is not distributed and only needs a known open
licence. Anything else -- an unknown licence, a non-commercial one, AGPL -- is
an error: that is what makes a new incompatible dependency break the build
(``tests/test_notice.py``) instead of passing unnoticed.

    uv run python packaging/notice.py            # check, print the offenders
    uv run python packaging/notice.py --write    # regenerate the section of NOTICE.md
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]
NOTICE = REPO / "NOTICE.md"
BEGIN = "<!-- inizio: generato da packaging/notice.py, non modificare a mano -->"
END = "<!-- fine della parte generata -->"

#: Compatible with GPL-3.0 (FSF list; "or-later" and "only" alike for LGPL/GPL-3).
COMPATIBLE = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "HPND",
    "ISC",
    "MIT",
    "MIT-CMU",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Unlicense",
    "Zlib",
    "BlueOak-1.0.0",
    "LGPL-2.1",
    "LGPL-2.1-or-later",
    "LGPL-3.0",
    "LGPL-3.0-or-later",
    "GPL-2.0-or-later",
    "GPL-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
}
#: Open, but not something to link into a GPL program: fine for a build tool.
TOOLS_ONLY = {"CC-BY-4.0", "CC-BY-3.0", "Artistic-2.0"}

#: What metadata writes instead of SPDX, and what it means.
ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "mit": "MIT",
    "mit license": "MIT",
    "bsd license": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "gplv3": "GPL-3.0",
    "gnu general public license v3 (gplv3)": "GPL-3.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "python software foundation license": "PSF-2.0",
    "isc license (iscl)": "ISC",
}

_TOKEN = re.compile(r"\(|\)|\bAND\b|\bOR\b|[^\s()]+")


def _spdx(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    if "\n" in text or len(text) > 80:  # a whole licence pasted in the field
        return None
    return ALIASES.get(text.lower(), text)


def python_licence(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    found = _spdx(meta.get("License-Expression")) or _spdx(meta.get("License"))
    if found and _known(found):
        return found
    classifiers = [
        _spdx(c.split("::")[-1])
        for c in (meta.get_all("Classifier") or [])
        if c.startswith("License ::")
    ]
    classifiers = [c for c in classifiers if c and c != "OSI Approved"]
    if classifiers:
        return " OR ".join(dict.fromkeys(classifiers))
    return found or "sconosciuta"


def _known(expression: str) -> bool:
    return all(
        t in COMPATIBLE | TOOLS_ONLY
        for t in _TOKEN.findall(expression)
        if t not in ("AND", "OR", "(", ")")
    )


def allowed(expression: str, accepted: set[str]) -> bool:
    """Evaluate an SPDX expression: OR needs one side, AND needs both."""
    tokens = _TOKEN.findall(expression)
    position = 0

    def term() -> bool:
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "(":
            value = either()
            position += 1  # the ")"
            return value
        return token.removesuffix("+") in accepted

    def both() -> bool:
        nonlocal position
        value = term()
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            value = term() and value
        return value

    def either() -> bool:
        nonlocal position
        value = both()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            value = both() or value
        return value

    try:
        return bool(tokens) and either() and position == len(tokens)
    except IndexError:
        return False


def _closure(requirements: list[Requirement], extras_of: dict[str, set[str]]) -> set[str]:
    """Every distribution the requirements pull in, as installed here."""
    seen: set[str] = set()
    queue = list(requirements)
    environment = default_environment()
    while queue:
        requirement = queue.pop()
        name = requirement.name.lower().replace("_", "-")
        wanted = extras_of.setdefault(name, set())
        new_extras = set(requirement.extras) - wanted
        if name in seen and not new_extras:
            continue
        wanted |= set(requirement.extras)
        seen.add(name)
        try:
            dist = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            continue  # not installed on this platform (a marker said so)
        for text in dist.requires or []:
            child = Requirement(text)
            if child.marker is None:
                queue.append(child)
                continue
            if any(
                child.marker.evaluate({**environment, "extra": extra}) for extra in (wanted or {""})
            ):
                queue.append(child)
    return seen


def python_runtime() -> list[tuple[str, str]]:
    """``(name, licence)`` of every Python package the program imports at run time."""
    project = metadata.distribution("autophotoedit")
    requirements = [Requirement(text) for text in project.requires or []]
    # Every extra of the project is part of the program: `install.sh` syncs them all.
    direct = [
        Requirement(str(r).split(";")[0])
        for r in requirements
        if r.marker is None or "extra" in str(r.marker)
    ]
    names = _closure(direct, {}) - {"autophotoedit"}
    rows = []
    for name in sorted(names):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        rows.append((dist.metadata["Name"], python_licence(dist)))
    return rows


def python_tools() -> list[tuple[str, str]]:
    """The development group: not distributed, only needs an open licence."""
    runtime = {name.lower() for name, _ in python_runtime()} | {"autophotoedit"}
    rows = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name.lower().replace("_", "-") in runtime or name.lower() in runtime:
            continue
        rows.append((name, python_licence(dist)))
    return sorted(rows, key=lambda row: row[0].lower())


def _lock() -> dict:
    path = REPO / "frontend" / "package-lock.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"packages": {}}


def _package_rows(dev: bool) -> list[tuple[str, str]]:
    rows = {}
    for key, entry in _lock()["packages"].items():
        if not key:
            continue
        name = key.split("node_modules/")[-1]
        # Workbox is a build dependency whose runtime the service worker inlines.
        distributed = not entry.get("dev") or name.startswith("workbox-")
        if distributed != (not dev):
            continue
        licence = entry.get("license") or "sconosciuta"
        rows[name] = licence.strip("()") if licence.count("(") == 1 else licence
    return sorted(rows.items())


def frontend_runtime() -> list[tuple[str, str]]:
    return _package_rows(dev=False)


def frontend_tools() -> list[tuple[str, str]]:
    return _package_rows(dev=True)


def offenders() -> list[str]:
    """Every package whose licence is not acceptable for what it is used for."""
    bad = []
    for kind, rows, accepted in (
        ("Python", python_runtime(), COMPATIBLE),
        ("frontend", frontend_runtime(), COMPATIBLE),
        ("Python (sviluppo)", python_tools(), COMPATIBLE | TOOLS_ONLY),
        ("frontend (sviluppo)", frontend_tools(), COMPATIBLE | TOOLS_ONLY),
    ):
        bad += [
            f"{kind}: {name} ({licence})"
            for name, licence in rows
            if not allowed(licence, accepted)
        ]
    return bad


def render() -> str:
    """The generated section of NOTICE.md."""

    def table(rows: list[tuple[str, str]]) -> str:
        return "\n".join(
            ["| Pacchetto | Licenza |", "|---|---|"] + [f"| {n} | {lic} |" for n, lic in rows]
        )

    return "\n\n".join(
        [
            BEGIN,
            "### Pacchetti Python distribuiti (dipendenze ed extra, con le loro dipendenze)",
            table(python_runtime()),
            "### Pacchetti del frontend inclusi nella build",
            table(frontend_runtime()),
            END,
        ]
    )


def current_section() -> str | None:
    text = NOTICE.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return None
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def write() -> None:
    text = NOTICE.read_text(encoding="utf-8")
    section = render()
    if BEGIN in text and END in text:
        text = text[: text.index(BEGIN)] + section + text[text.index(END) + len(END) :]
    else:
        text = text.rstrip() + "\n\n## Elenco completo\n\n" + section + "\n"
    NOTICE.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    if "--write" in argv:
        write()
    bad = offenders()
    for line in bad:
        print(f"licenza non ammessa: {line}")
    stale = current_section() != render()
    if stale:
        print("NOTICE.md non corrisponde alle dipendenze: rigeneralo con --write")
    return 1 if bad or stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

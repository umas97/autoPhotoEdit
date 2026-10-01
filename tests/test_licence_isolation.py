# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""docs/SPEC.md section 24: every exiv2 call stays behind two modules.

pyexiv2 is GPL-3.0 and is what forces this project's licence. The spec keeps the
door open to replacing it -- with exiftool over a subprocess, or with direct XMP
writing -- and notes that the replacement is only a few hours of work *if* the
dependency never leaks. This test is what keeps that true; without it the leak
happens gradually and is discovered on the day somebody tries to relicense.

The same reasoning applies to the runtime ML rule of section 26: no PyTorch, no
TensorFlow, no GPU dependency anywhere in the shipped code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent / "backend" / "ape"

#: The only modules allowed to import pyexiv2, verbatim from the spec.
EXIV2_ALLOWED = {"raw/metadata.py"}
EXIV2_ALLOWED_PACKAGES = {"export"}

#: Forbidden at runtime on a machine with no discrete GPU (section 26).
FORBIDDEN_RUNTIME = {"torch", "tensorflow", "jax", "cupy", "pycuda"}


def _modules() -> list[Path]:
    return sorted(BACKEND.rglob("*.py"))


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _relative(path: Path) -> str:
    return str(path.relative_to(BACKEND))


def test_pyexiv2_is_confined_to_metadata_and_export():
    offenders = [
        _relative(path)
        for path in _modules()
        if "pyexiv2" in _imported_names(path)
        and _relative(path) not in EXIV2_ALLOWED
        and path.parent.name not in EXIV2_ALLOWED_PACKAGES
    ]
    assert not offenders, (
        "pyexiv2 importato fuori da raw/metadata.py e export/: "
        f"{offenders}. Vedi docs/SPEC.md §24."
    )


@pytest.mark.parametrize("package", sorted(FORBIDDEN_RUNTIME))
def test_no_gpu_machine_learning_frameworks(package):
    offenders = [_relative(p) for p in _modules() if package in _imported_names(p)]
    assert not offenders, f"{package} importato in {offenders}. Vedi docs/SPEC.md §26."


def test_every_source_file_carries_the_licence_header():
    """Section 24 asks for a licence header in the main source files."""
    missing = [
        _relative(path)
        for path in _modules()
        if "SPDX-License-Identifier: GPL-3.0-or-later" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, f"intestazione di licenza mancante in {missing}"


def test_no_source_file_is_oversized():
    """Section 26: no code file above roughly 400 lines; if it grows, it splits."""
    oversized = {
        _relative(path): len(path.read_text(encoding="utf-8").splitlines())
        for path in _modules()
        if len(path.read_text(encoding="utf-8").splitlines()) > 400
    }
    assert not oversized, f"file troppo lunghi: {oversized}"


FRONTEND_SRC = BACKEND.parent.parent / "frontend" / "src"


@pytest.mark.skipif(not FRONTEND_SRC.is_dir(), reason="frontend assente")
def test_no_frontend_file_is_oversized():
    """Section 26 holds for the interface too. ``openapi.d.ts`` is generated."""
    oversized = {
        str(path.relative_to(FRONTEND_SRC)): count
        for path in sorted(FRONTEND_SRC.rglob("*.ts*"))
        if path.name != "openapi.d.ts"
        and (count := len(path.read_text(encoding="utf-8").splitlines())) > 400
    }
    assert not oversized, f"file troppo lunghi: {oversized}"

# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Section 24: NOTICE.md is generated and checked.

A new dependency with a licence incompatible with GPL-3.0 -- or with none that
can be read -- breaks this test, and so does a dependency list that no longer
matches what is installed. The fix for the second is one command, printed in
the message; the fix for the first is a decision, and it is the user's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packaging"))

import notice  # noqa: E402


def test_every_distributed_package_has_a_licence_compatible_with_gpl3():
    assert notice.offenders() == []


def test_the_list_in_notice_md_is_the_generated_one():
    assert notice.current_section() == notice.render(), (
        "NOTICE.md non corrisponde alle dipendenze installate: "
        "uv run python packaging/notice.py --write"
    )


def test_the_project_itself_is_in_the_list_of_what_it_imports():
    names = {name.lower() for name, _ in notice.python_runtime()}
    assert {"pyexiv2", "rawpy", "numpy", "fastapi", "onnxruntime", "lensfunpy"} <= names
    assert "pytest" not in names  # a development tool is not distributed
    frontend = {name for name, _ in notice.frontend_runtime()}
    assert {"react", "zustand", "workbox-precaching"} <= frontend or not frontend


@pytest.mark.parametrize(
    ("expression", "ok"),
    [
        ("MIT", True),
        ("Apache-2.0 OR BSD-2-Clause", True),
        ("BSD-3-Clause AND 0BSD AND MIT", True),
        ("MIT AND CC-BY-NC-4.0", False),  # non-commercial: never
        ("AGPL-3.0 OR Proprietary", False),
        ("(MIT OR CC0-1.0)", True),
        ("GPL-2.0-only", False),  # incompatible with GPL-3.0
        ("sconosciuta", False),
        ("", False),
    ],
)
def test_licence_expressions_are_read_as_spdx(expression: str, ok: bool):
    assert notice.allowed(expression, notice.COMPATIBLE) is ok

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Run pyright over 'pyright_cases.py' and assert its diagnostics.

This exists because the dimension half of the mypy plugin was removed (ADR 0028): client code must
now type-check under a checker that has never had gt4py support.
"""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest


CASES = pathlib.Path(__file__).parent / "pyright_cases.py"
MARKER = "# EXPECT-ERROR"


def _expected_error_lines() -> set[int]:
    return {n for n, line in enumerate(CASES.read_text().splitlines(), start=1) if MARKER in line}


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright is not installed")
def test_pyright_diagnostics_match_expectations():
    result = subprocess.run(
        [
            "pyright",
            "--outputjson",
            "--pythonversion",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            str(CASES),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    errors = {
        d["range"]["start"]["line"] + 1
        for d in report["generalDiagnostics"]
        if d["severity"] == "error"
    }
    expected = _expected_error_lines()

    assert errors == expected, (
        f"unexpected pyright errors on lines {sorted(errors - expected)}; "
        f"missing expected errors on lines {sorted(expected - errors)}\n"
        + "\n".join(
            f"  {d['range']['start']['line'] + 1}: {d['message'].splitlines()[0]}"
            for d in report["generalDiagnostics"]
        )
    )

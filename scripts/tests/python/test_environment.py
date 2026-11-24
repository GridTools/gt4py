#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Tests for the environment the dev-scripts run in, as set up by their uv shebang."""

from __future__ import annotations

import json
import pathlib
import shlex
import subprocess

import pytest
from helpers import common


_ENV_PREFIX = "#!/usr/bin/env -S "

_EXECUTABLES = [
    common.SCRIPTS_DIR / "run",
    common.SCRIPTS_DIR / "test",
    *sorted(p for p in common.PY_SCRIPTS_DIR.glob("*.py") if p.read_text().startswith(_ENV_PREFIX)),
]

_PROBE = (
    "import importlib.util, json; "
    "print(json.dumps({m: importlib.util.find_spec(m) is not None "
    "for m in ['gt4py', 'packaging', 'pytest', 'typer', 'yaml']}))"
)


def _shebang_command(script: pathlib.Path) -> list[str]:
    first_line = script.read_text().splitlines()[0]
    assert first_line.startswith(_ENV_PREFIX), first_line
    return shlex.split(first_line.removeprefix(_ENV_PREFIX))


@pytest.mark.parametrize("script", _EXECUTABLES, ids=lambda p: p.name)
def test_shebang_environment_excludes_gt4py(script: pathlib.Path):
    # Run the shebang's interpreter command on a probe instead of the script itself.
    result = subprocess.run(
        [*_shebang_command(script), "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=common.REPO_ROOT,
        check=True,
    )
    available = json.loads(result.stdout)

    assert not available["gt4py"], "the dev-scripts environment must not install gt4py"
    assert available["packaging"] and available["typer"] and available["yaml"]
    assert available["pytest"] == (script.name == "test")

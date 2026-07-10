#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for Python dev-scripts."""

from __future__ import annotations

import pathlib
from typing import Final


PY_SCRIPTS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().absolute().parent.parent
SCRIPTS_DIR: Final[pathlib.Path] = PY_SCRIPTS_DIR.parent
REPO_ROOT: Final[pathlib.Path] = SCRIPTS_DIR.parent
PYTHON_VERSIONS: Final[list[str]] = [
    v
    for line in (REPO_ROOT / ".python-versions").read_text().splitlines()
    if (v := line.strip()) and not v.startswith("#")
]

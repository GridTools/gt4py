#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Shared fixtures for dev-scripts tests."""

from __future__ import annotations

import pathlib
import sys


# Add the 'python' folder to sys.path to mimic the `sys.path` of the command
# modules run directly as scripts (so tests can `from helpers import common`,
# `import nox_sessions`, etc.).
python_modules_path = pathlib.Path(__file__).resolve().absolute().parent.parent.parent / "python"
sys.path.insert(0, str(python_modules_path))

#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Example tests for the ``helpers.common`` shared dev-scripts utilities."""

from __future__ import annotations

from helpers import common


def test_dir_constants_are_consistent():
    assert common.PY_SCRIPTS_DIR.name == "python"
    assert common.SCRIPTS_DIR.name == "scripts"
    assert common.PY_SCRIPTS_DIR.parent == common.SCRIPTS_DIR
    assert common.SCRIPTS_DIR.parent == common.REPO_ROOT


def test_repo_root_points_to_the_repository():
    assert (common.REPO_ROOT / "pyproject.toml").is_file()
    assert (common.REPO_ROOT / ".python-versions").is_file()


def test_python_versions_match_the_versions_file():
    raw = (common.REPO_ROOT / ".python-versions").read_text().splitlines()
    expected = [v for line in raw if (v := line.strip()) and not v.startswith("#")]

    assert common.PYTHON_VERSIONS == expected
    assert all(version[0].isdigit() for version in common.PYTHON_VERSIONS)

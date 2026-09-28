#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Tests for the ``update`` dev-script."""

from __future__ import annotations

import pathlib

import pytest
from helpers import common
from typer.testing import CliRunner
from update import ExitCode, cli


_CURRENT_VERSION = "1.2.2+unknown.version.details"


@pytest.fixture
def fake_repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A minimal repository layout with the two files `package-version` rewrites."""
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.versioningit]\ndefault-version = "{_CURRENT_VERSION}"\n'
    )
    about = tmp_path / "src" / "gt4py" / "__about__.py"
    about.parent.mkdir(parents=True)
    about.write_text(
        f'from typing import Final\n\non_build_version: Final = "{_CURRENT_VERSION}"\n'
    )
    monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
    return tmp_path


def _snapshot(root: pathlib.Path) -> dict[pathlib.Path, str]:
    return {p: p.read_text() for p in root.rglob("*") if p.is_file()}


def test_package_version_rejects_invalid_version(fake_repo: pathlib.Path):
    before = _snapshot(fake_repo)

    result = CliRunner().invoke(cli, ["package-version", "not-a-version"])

    assert result.exit_code == ExitCode.INVALID_NEW_VERSION_STRING, result.output
    assert not isinstance(result.exception, AttributeError)
    assert "not a valid version string" in result.output
    assert _snapshot(fake_repo) == before


def test_package_version_rewrites_default_version(fake_repo: pathlib.Path):
    result = CliRunner().invoke(cli, ["package-version", "1.2.3"])

    assert result.exit_code == 0, result.output
    new_version = "1.2.3+unknown.version.details"
    assert f'default-version = "{new_version}"' in (fake_repo / "pyproject.toml").read_text()
    assert (
        f'on_build_version: Final = "{new_version}"'
        in (fake_repo / "src" / "gt4py" / "__about__.py").read_text()
    )

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the automatic skipping of `requires_*`-marked tests.

The `requires_*` markers describe the environment a test needs, and the root
`tests/conftest.py` enforces them by skipping the marked tests when the
dependency is unavailable. The tests below pin that contract from both sides:
the `test_requires_*` canaries fail if the skipping ever stops happening, and
`test_every_requirement_marker_has_a_probe` fails if a new marker is declared
without teaching the conftest how to check for it.
"""

import importlib
import importlib.util
import pathlib
import tomllib
from typing import Any

import pytest

from gt4py._core import definitions as core_defs


_REPO_ROOT = pathlib.Path(__file__).parents[3]


def _load_root_conftest() -> Any:
    """Import `tests/conftest.py` as a standalone module, to inspect its probe table."""
    path = _REPO_ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("gt4py_tests_root_conftest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared_requirement_markers() -> set[str]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    declarations = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    names = (declaration.split(":", 1)[0].strip() for declaration in declarations)
    return {name for name in names if name.startswith("requires_")}


def test_every_requirement_marker_has_a_probe():
    """Every declared `requires_*` marker must be enforceable by the root conftest."""
    declared = _declared_requirement_markers()
    probed = set(_load_root_conftest()._REQUIREMENT_PROBES)

    assert declared, "no 'requires_*' markers found in 'pyproject.toml'"
    assert declared == probed, (
        "The 'requires_*' markers declared in 'pyproject.toml' and the probes in"
        " 'tests/conftest.py' are out of sync."
        f" Declared but not probed: {sorted(declared - probed)}."
        f" Probed but not declared: {sorted(probed - declared)}."
    )


@pytest.mark.requires_atlas
def test_requires_atlas_marker_is_enforced():
    importlib.import_module("atlas4py")


@pytest.mark.requires_jax
def test_requires_jax_marker_is_enforced():
    importlib.import_module("jax")


@pytest.mark.requires_gpu
def test_requires_gpu_marker_is_enforced():
    # An importable but unusable `cupy` must not count as a GPU.
    assert core_defs.gpu_device_count() > 0

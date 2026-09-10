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
import os
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

import pytest


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
    # Touch the device for real. Asserting `gpu_device_count() > 0` here would be
    # circular: that is the probe which decided not to skip this test.
    cp = importlib.import_module("cupy")
    assert int(cp.zeros(1).sum()) == 0


def test_auto_skip_hook_is_wired(tmp_path):
    """A `requires_*` test must be skipped when its probe reports unavailable.

    The canaries above only exercise this where the dependency is genuinely
    missing. This drives the real hook with a stubbed probe, so the contract is
    checked whatever happens to be installed.
    """
    root_conftest = _REPO_ROOT / "tests" / "conftest.py"
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    requires_gpu: stubbed requirement\n"
    )
    (tmp_path / "conftest.py").write_text(f"""
import importlib.util

_spec = importlib.util.spec_from_file_location("_root_conftest", r"{root_conftest}")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)

_root._REQUIREMENT_PROBES = {{"requires_gpu": (lambda: False, "a stubbed dependency")}}
_root._unmet_requirement_marks.cache_clear()

pytest_addoption = _root.pytest_addoption
pytest_collection_modifyitems = _root.pytest_collection_modifyitems
""")
    (tmp_path / "test_stub.py").write_text("""
import pytest

@pytest.mark.requires_gpu
def test_marked():
    raise AssertionError("should have been skipped")

def test_unmarked():
    pass
""")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no", str(tmp_path), *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            # Hermetic: no third-party plugin may skip or reorder these two tests.
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )

    result = run()
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout and "1 skipped" in result.stdout, result.stdout

    # ... and the escape hatch must let the failure through.
    result = run("--require-optional-deps")
    assert result.returncode == 1, result.stdout
    assert "1 failed" in result.stdout and "1 passed" in result.stdout, result.stdout

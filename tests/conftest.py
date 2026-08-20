# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Global configuration of pytest for collecting and running tests."""

import collections.abc
import functools
import importlib.util
import sys
import types
from typing import Final

import pytest


# `pytester` powers the wiring test in `package_tests` that runs a nested pytest
# session with a stubbed probe; it must be enabled from the root conftest.
pytest_plugins = ["pytester"]

# Ignore hidden folders and disabled tests
collect_ignore_glob = [".*", "_disabled*"]

# Custom module attribute to store package-level marks
_PKG_MARKS_ATTR_NAME: Final = "package_pytestmarks"


@functools.cache
def _get_pkg_marks(module_name: str) -> list[pytest.Mark | str]:
    """Collect markers in the `package_pytestmarks` module attribute (and recursively from its parents)."""
    module = sys.modules[module_name]
    pkg_markers = getattr(module, _PKG_MARKS_ATTR_NAME, [])
    assert isinstance(pkg_markers, collections.abc.Sequence), (
        f"'{_PKG_MARKS_ATTR_NAME}' content must be a sequence of marks"
    )

    if (parent := module_name.rsplit(".", 1)[0]) != module_name:
        pkg_markers += _get_pkg_marks(parent)

    return pkg_markers


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        # `find_spec` raises rather than returning `None` for a partially broken
        # install (and for any meta-path finder that raises). This runs inside a
        # collection hook, where an exception aborts the whole session, so treat
        # anything unresolvable as simply unavailable.
        return False


def _is_gpu_available() -> bool:
    """Check that a GPU device is actually usable, not merely that `cupy` is installed."""
    from gt4py._core import definitions as core_defs

    return core_defs.gpu_device_count() > 0


# `requires_*` markers describe the *environment* a test needs, so they can
# be enforced here instead of relying on the caller passing the matching
# `-m` filter (which only `noxfile.py` does).
_REQUIREMENT_PROBES: Final[dict[str, tuple[collections.abc.Callable[[], bool], str]]] = {
    "requires_atlas": (lambda: _is_importable("atlas4py"), "the `atlas4py` package"),
    "requires_gpu": (_is_gpu_available, "`cupy` and a reachable GPU device"),
    "requires_jax": (lambda: _is_importable("jax"), "the `jax` package"),
}


@functools.cache
def _unmet_requirement_marks() -> dict[str, pytest.MarkDecorator]:
    """Build the skip marks for the `requires_*` markers unmet in this environment."""
    return {
        marker: pytest.mark.skip(
            reason=f"Requires {description}, unavailable in this environment"
            " (use `--require-optional-deps` to run anyway and see the actual error)"
        )
        for marker, (is_available, description) in _REQUIREMENT_PROBES.items()
        if not is_available()
    }


def pytest_addoption(parser: pytest.Parser) -> None:
    """Pytest hook to register custom command line options."""
    parser.addoption(
        "--require-optional-deps",
        action="store_true",
        default=False,
        dest="require_optional_deps",
        help=(
            "Do not auto-skip `requires_*` tests when the optional dependency is"
            " missing, and let them fail instead. Meant for CI sessions that install"
            " the dependency on purpose, where a broken environment must not pass"
            " silently as a fully skipped run."
        ),
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pytest hook to modify the collected test items.

    See: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
    """
    unmet_marks = {} if config.option.require_optional_deps else _unmet_requirement_marks()

    for item in items:
        # Visit the chain of parents of the current test item in reverse order,
        # until we get to the module object where the test function (or class)
        # has been defined. At that point, process the custom package-level marks
        # attribute if present, and move to the next collected item in the list.
        for node in item.listchain()[-2::-1]:
            if not (obj := getattr(node, "obj", None)):
                break
            if not isinstance(obj, types.ModuleType):
                continue

            module_name = obj.__name__
            for marker in _get_pkg_marks(module_name):
                item.add_marker(marker)

        # Skip tests whose `requires_*` markers are not satisfied here, so the
        # markers work the same no matter how `pytest` was invoked.
        for marker, skip_mark in unmet_marks.items():
            # Not `marker in item.keywords`: keywords also hold every node *name* in
            # the chain, so a file or function merely named after a marker would be
            # skipped without carrying it.
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip_mark)

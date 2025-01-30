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
import sys
import types
from typing import Final

import pytest


# Ignore hidden folders and disabled tests
collect_ignore_glob = [".*", "_disabled*"]

# Custom module attribute to store package-level marks
_PKG_MARKS_ATTR_NAME: Final = "package_pytestmarks"


@functools.cache
def _get_pkg_marks(module_name: str) -> list[pytest.Mark | str]:
    """Collect markers in the `package_pytestmarks` module attribute (and recursively from its parents)."""
    module = sys.modules[module_name]
    pkg_markers = getattr(module, _PKG_MARKS_ATTR_NAME, [])
    assert isinstance(
        pkg_markers, collections.abc.Sequence
    ), f"'{_PKG_MARKS_ATTR_NAME}' content must be a sequence of marks"

    if (parent := module_name.rsplit(".", 1)[0]) != module_name:
        pkg_markers += _get_pkg_marks(parent)

    return pkg_markers


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pytest hook to modify the collected test items.

    See: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
    """
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

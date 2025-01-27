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


_PKG_MARKS_ATTR_NAME: Final = "package_pytestmarks"


@functools.cache
def _get_pkg_markers(module_name: str) -> list[pytest.Mark | str]:
    module = sys.modules[module_name]
    pkg_markers = getattr(module, _PKG_MARKS_ATTR_NAME, [])
    assert isinstance(
        pkg_markers, collections.abc.Sequence
    ), f"'{_PKG_MARKS_ATTR_NAME}' content must be a sequence of markers"

    if (parent := module_name.rsplit(".", 1)[0]) != module_name:
        pkg_markers += _get_pkg_markers(parent)

    return pkg_markers


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        for node in item.listchain()[-2::-1]:
            if not (obj := getattr(node, "obj", None)):
                break
            if not isinstance(obj, types.ModuleType):
                continue

            module_name = obj.__name__
            for marker in _get_pkg_markers(module_name):
                item.add_marker(marker)

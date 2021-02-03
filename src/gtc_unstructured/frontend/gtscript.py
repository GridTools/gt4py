# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from typing import Callable

import gtc_unstructured.irs.common as common

from .built_in_types import Field, Local, Location, Mesh, TemporaryField


# built-in symbols
Vertex = common.LocationType.Vertex
Edge = common.LocationType.Edge
Cell = common.LocationType.Cell
FORWARD = common.LoopOrder.FORWARD
BACKWARD = common.LoopOrder.BACKWARD

_built_in_functions = [
    "computation",
    "interval",
    "location",
    "neighbors",
    "vertices",
    "edges",
    "cells",
]

__all__ = _built_in_functions + [
    "Field",
    "Local",
    "Location",
    "Mesh",
    "TemporaryField",
    "Vertex",
    "Edge",
    "Cell",
]


# generate built-in function stubs
class GTScriptBuiltInNotCallableFromPythonException(Exception):
    message: str

    def __init__(self, name: str):
        self.message = f"GTScript built-in function `{name}` not callable from python"
        super().__init__(self.message)


def _generate_built_in_function_stub(func_name: str) -> Callable:
    def stub(*_):
        raise GTScriptBuiltInNotCallableFromPythonException(func_name)

    return stub


for built_in_function in _built_in_functions:
    setattr(
        sys.modules[__name__],
        built_in_function,
        _generate_built_in_function_stub(built_in_function),
    )

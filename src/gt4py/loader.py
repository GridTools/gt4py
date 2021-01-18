# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Implementation of stencil functions defined with symbolic Python functions.

This module contains functions to generate callable objects implementing
a high-level stencil function definition using a specific code generating backend.
"""

import types
from typing import TYPE_CHECKING, Any, Dict, Type

from gt4py import backend as gt_backend
from gt4py import frontend as gt_frontend
from gt4py.stencil_builder import StencilBuilder
from gt4py.type_hints import StencilFunc


if TYPE_CHECKING:
    from gt4py.definitions import BuildOptions
    from gt4py.stencil_object import StencilObject


def load_stencil(
    frontend_name: str,
    backend_name: str,
    definition_func: StencilFunc,
    externals: Dict[str, Any],
    build_options: "BuildOptions",
) -> Type["StencilObject"]:
    """Generate a new class object implementing the provided definition."""
    # Load components
    backend_cls = gt_backend.from_name(backend_name)
    if backend_cls is None:
        raise ValueError("Unknown backend name ({name})".format(name=backend_name))

    frontend = gt_frontend.from_name(frontend_name)
    if frontend is None:
        raise ValueError("Invalid frontend specification ({name})".format(name=frontend_name))

    builder = StencilBuilder(
        definition_func, options=build_options, backend=backend_cls, frontend=frontend
    ).with_externals(externals)

    return builder.build()


def gtscript_loader(
    definition_func: StencilFunc,
    backend: str,
    build_options: "BuildOptions",
    externals: Dict[str, Any],
) -> "StencilObject":
    if not isinstance(definition_func, types.FunctionType):
        raise ValueError("Invalid stencil definition object ({obj})".format(obj=definition_func))

    if not build_options.name:
        build_options.name = f"{definition_func.__name__}"
    stencil_class = load_stencil("gtscript", backend, definition_func, externals, build_options)

    return stencil_class()

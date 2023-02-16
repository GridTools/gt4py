# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

"""Implementation of stencil functions defined with symbolic Python functions.

This module contains functions to generate callable objects implementing
a high-level stencil function definition using a specific code generating backend.
"""

import types
from typing import TYPE_CHECKING, Any, Dict, Type

from gt4py.cartesian import backend as gt_backend, frontend as gt_frontend
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.cartesian.type_hints import StencilFunc


if TYPE_CHECKING:
    from gt4py.cartesian.definitions import BuildOptions
    from gt4py.cartesian.stencil_object import StencilObject


def load_stencil(
    frontend_name: str,
    backend_name: str,
    definition_func: StencilFunc,
    externals: Dict[str, Any],
    dtypes: Dict[Type, Type],
    build_options: "BuildOptions",
) -> Type["StencilObject"]:
    """Generate a new class object implementing the provided definition."""
    # Load components
    backend_cls = gt_backend.from_name(backend_name)
    if backend_cls is None:
        raise ValueError(f"Unknown backend name ({backend_name})")

    frontend = gt_frontend.from_name(frontend_name)
    if frontend is None:
        raise ValueError(f"Invalid frontend name ({frontend_name})")

    builder = (
        StencilBuilder(
            definition_func, options=build_options, backend=backend_name, frontend=frontend
        )
        .with_externals(externals)
        .with_dtypes(dtypes)
    )

    return builder.build()


def gtscript_loader(
    definition_func: StencilFunc,
    backend: str,
    build_options: "BuildOptions",
    externals: Dict[str, Any],
    dtypes: Dict[Type, Type],
) -> "StencilObject":
    if not isinstance(definition_func, types.FunctionType):
        raise ValueError("Invalid stencil definition object ({obj})".format(obj=definition_func))

    if not build_options.name:
        build_options.name = f"{definition_func.__name__}"
    stencil_class = load_stencil(
        "gtscript", backend, definition_func, externals, dtypes, build_options
    )

    return stencil_class()

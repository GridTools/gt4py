# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of stencil functions defined with symbolic Python functions.

This module contains functions to generate callable objects implementing
a high-level stencil function definition using a specific code generating backend.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

from gt4py.cartesian import frontend as gt_frontend
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.cartesian.type_hints import StencilFunc


if TYPE_CHECKING:
    from gt4py.cartesian.definitions import BuildOptions
    from gt4py.cartesian.stencil_object import StencilObject


def load_stencil(
    frontend_name: str,
    backend_name: str,
    definition_func: StencilFunc,
    externals: dict[str, Any],
    dtypes: dict[type, type],
    build_options: BuildOptions,
) -> type[StencilObject]:
    """Generate a new class object implementing the provided definition."""
    # Load components
    frontend = gt_frontend.from_name(frontend_name)

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
    build_options: BuildOptions,
    externals: dict[str, Any],
    dtypes: dict[type, type],
) -> StencilObject:
    if not isinstance(definition_func, types.FunctionType):
        raise ValueError("Invalid stencil definition object ({obj})".format(obj=definition_func))

    if not build_options.name:
        build_options.name = f"{definition_func.__name__}"
    stencil_class = load_stencil(
        "gtscript", backend, definition_func, externals, dtypes, build_options
    )

    return stencil_class()

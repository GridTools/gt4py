# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

from gt4py import backend as gt_backend
from gt4py import frontend as gt_frontend
from gt4py.stencil_object import StencilObject


def load_stencil(frontend_name, backend_name, definition_func, externals, build_options):
    """Generate a new class object implementing the provided definition.
    """

    # Load components
    backend = gt_backend.from_name(backend_name)
    if backend is None:
        raise ValueError("Unknown backend name ({name})".format(name=backend_name))

    frontend = gt_frontend.from_name(frontend_name)
    if frontend is None:
        raise ValueError("Invalid frontend specification ({name})".format(name=frontend_name))

    # Create ID
    options_id = backend.get_options_id(build_options)
    stencil_id = frontend.get_stencil_id(
        build_options.qualified_name, definition_func, externals, options_id
    )

    # Load or generate class
    stencil_class = (
        None if build_options.rebuild else backend.load(stencil_id, definition_func, build_options)
    )

    if stencil_class is None:
        definition_ir = frontend.generate(definition_func, externals, build_options)
        stencil_class = backend.generate(stencil_id, definition_ir, definition_func, build_options)

    return stencil_class


def gtscript_loader(definition_func, backend, build_options, externals):
    if not isinstance(definition_func, types.FunctionType):
        raise ValueError("Invalid stencil definition object ({obj})".format(obj=definition_func))

    if not build_options.name:
        build_options.name = f"{definition_func.__name__}"
    stencil_class = load_stencil("gtscript", backend, definition_func, externals, build_options)

    return stencil_class()

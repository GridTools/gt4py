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

import gt4py.definitions as gt_definitions
import gt4py.gtscript as gtscript

from ..definitions import id_version
from .stencil_definitions import EXTERNALS_REGISTRY
from .stencil_definitions import REGISTRY as stencil_registry


def generate_test_module(name, backend, *, id_version, rebuild=True):
    module_name = "_test_module." + name
    stencil_name = name
    backend_opts = {}
    if "debug_mode" in backend.options:
        backend_opts["debug_mode"] = False
    if "add_profile_info" in backend.options:
        backend_opts["add_profile_info"] = True
    if "verbose" in backend.options:
        backend_opts["verbose"] = True
    options = gt_definitions.BuildOptions(
        name=stencil_name, module=module_name, rebuild=rebuild, backend_opts=backend_opts
    )
    decorator = gtscript.stencil(backend=backend.name, externals=EXTERNALS_REGISTRY[stencil_name])
    stencil_definition = stencil_registry[name]

    return decorator(stencil_definition)

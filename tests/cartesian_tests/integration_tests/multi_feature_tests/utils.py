# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.cartesian.definitions as gt_definitions
import gt4py.cartesian.gtscript as gtscript
from tests.cartesian_tests.definitions import id_version

from .stencil_definitions import EXTERNALS_REGISTRY, REGISTRY as stencil_registry


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

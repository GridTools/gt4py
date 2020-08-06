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


import gt4py.backend as gt_back
import gt4py.definitions as gt_definitions

from ..utils import id_version
from .iir_stencil_definitions import build_iir_stencil


def generate_test_module(name, backend, *, id_version, rebuild=True):
    module_name = "_test_module." + name
    stencil_name = name + "_stencil"
    backend_opts = {}
    if issubclass(backend, gt_back.BaseGTBackend):
        backend_opts["debug_mode"] = False
        backend_opts["add_profile_info"] = True
        backend_opts["verbose"] = True
    options = gt_definitions.BuildOptions(
        name=stencil_name, module=module_name, rebuild=rebuild, backend_opts=backend_opts
    )
    return build_iir_stencil(name, options, backend, id_version=id_version)

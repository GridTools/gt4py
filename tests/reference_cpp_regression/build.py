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

import inspect
import os

import gt4py
import gt4py.utils as gt_utils
from gt4py import gt_src_manager
from gt4py.backend import pyext_builder


GT4PY_INSTALLATION_PATH = os.path.dirname(inspect.getabsfile(gt4py))

EXTERNAL_SRC_PATH = os.path.join(GT4PY_INSTALLATION_PATH, "_external_src")


assert gt_src_manager.has_gt_sources() or gt_src_manager.install_gt_sources()


def compile_reference():
    current_dir = os.path.dirname(__file__)
    build_opts = pyext_builder.get_gt_pyext_build_opts()
    build_opts["include_dirs"].append(EXTERNAL_SRC_PATH)

    build_opts.setdefault("extra_compile_args", [])
    build_opts["extra_compile_args"].append("-Wno-sign-compare")
    reference_names = pyext_builder.build_pybind_ext(
        "reference_cpp_regression",
        [os.path.join(current_dir, "reference.cpp")],
        os.path.join(current_dir, "build"),
        current_dir,
        verbose=False,
        clean=False,
        **build_opts,
    )
    return gt_utils.make_module_from_file(*reference_names)

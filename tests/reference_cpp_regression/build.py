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

import inspect
import os

import gt4py
import gt4py.utils as gt_utils


GT4PY_INSTALLATION_PATH = os.path.dirname(inspect.getabsfile(gt4py))

EXTERNAL_SRC_PATH = os.path.join(GT4PY_INSTALLATION_PATH, "_external_src")


def compile_reference():
    from gt4py.backend.pyext_builder import build_gtcpu_ext

    current_dir = os.path.dirname(__file__)
    reference_names = build_gtcpu_ext(
        "reference_cpp_regression",
        [os.path.join(current_dir, "reference.cpp")],
        os.path.join(current_dir, "build"),
        current_dir,
        verbose=False,
        clean=False,
        extra_include_dirs=[EXTERNAL_SRC_PATH],
    )
    return gt_utils.make_module_from_file(*reference_names)

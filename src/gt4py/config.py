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

import multiprocessing
import os


BOOST_ROOT = os.environ.get(
    "BOOST_ROOT", os.environ.get("BOOST_HOME", os.path.abspath("/usr/local/include"))
)

CUDA_ROOT = os.environ.get(
    "CUDA_HOME", os.environ.get("CUDA_PATH", os.path.abspath("/usr/local/cuda"))
)

GT_REPO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_external_src", "gridtools")
)

GT_INCLUDE_PATH = os.path.abspath(os.path.join(GT_REPO_PATH, "include"))

build_settings = {
    "boost_include_path": os.path.join(BOOST_ROOT, "include"),
    "cuda_bin_path": os.path.join(CUDA_ROOT, "bin"),
    "cuda_include_path": os.path.join(CUDA_ROOT, "include"),
    "cuda_library_path": os.path.join(CUDA_ROOT, "lib64"),
    "gt_include_path": os.environ.get("GT_INCLUDE_PATH", GT_INCLUDE_PATH),
    "extra_compile_args": [],
    "extra_link_args": [],
    "parallel_jobs": multiprocessing.cpu_count(),
}

cache_settings = {
    "dir_name": os.environ.get("GT_CACHE_DIR_NAME", ".gt_cache"),
    "root_path": os.environ.get("GT_CACHE_ROOT", os.path.abspath(".")),
}

code_settings = {"root_package_name": "_GT_"}

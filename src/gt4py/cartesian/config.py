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

import multiprocessing
import os
from typing import Any, Dict, Optional

import gridtools_cpp


GT4PY_INSTALLATION_PATH: str = os.path.dirname(os.path.abspath(__file__))

# Default paths (taken from user's environment vars when possible)
BOOST_ROOT: str = os.environ.get(
    "BOOST_ROOT", os.environ.get("BOOST_HOME", os.path.abspath("/usr/local"))
)

CUDA_ROOT: str = os.environ.get(
    "CUDA_HOME", os.environ.get("CUDA_PATH", os.path.abspath("/usr/local/cuda"))
)

CUDA_HOST_CXX: Optional[str] = os.environ.get("CUDA_HOST_CXX", None)


GT_INCLUDE_PATH: str = os.path.abspath(gridtools_cpp.get_include_dir())

GT_CPP_TEMPLATE_DEPTH: int = 1024

# Settings dict
build_settings: Dict[str, Any] = {
    "boost_include_path": os.path.join(BOOST_ROOT, "include"),
    "cuda_bin_path": os.path.join(CUDA_ROOT, "bin"),
    "cuda_include_path": os.path.join(CUDA_ROOT, "include"),
    "cuda_library_path": os.path.join(CUDA_ROOT, "lib64"),
    "cuda_arch": os.environ.get("CUDA_ARCH", None),
    "gt_include_path": os.environ.get("GT_INCLUDE_PATH", GT_INCLUDE_PATH),
    "openmp_cppflags": os.environ.get("OPENMP_CPPFLAGS", "-fopenmp").split(),
    "openmp_ldflags": os.environ.get("OPENMP_LDFLAGS", "-fopenmp").split(),
    "extra_compile_args": {
        "cxx": [],
        "nvcc": [],
    },
    "extra_link_args": [],
    "parallel_jobs": multiprocessing.cpu_count(),
    "cpp_template_depth": os.environ.get("GT_CPP_TEMPLATE_DEPTH", GT_CPP_TEMPLATE_DEPTH),
}

if CUDA_HOST_CXX is not None:
    build_settings["extra_compile_args"]["nvcc"].append(f"-ccbin={CUDA_HOST_CXX}")

cache_settings: Dict[str, Any] = {
    "dir_name": os.environ.get("GT_CACHE_DIR_NAME", ".gt_cache"),
    "root_path": os.environ.get("GT_CACHE_ROOT", os.path.abspath(".")),
    "load_retries": int(os.environ.get("GT_CACHE_LOAD_RETRIES", 3)),
    "load_retry_delay": int(os.environ.get("GT_CACHE_LOAD_RETRY_DELAY", 100)),  # unit miliseconds
}

code_settings: Dict[str, Any] = {"root_package_name": "_GT_"}

os.environ.setdefault("DACE_CONFIG", os.path.join(os.path.abspath("."), ".dace.conf"))

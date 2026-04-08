# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
import os
from typing import Any

import gridtools_cpp

from gt4py.cartesian.utils.compiler import cxx_compiler_defaults, gpu_configuration


GT4PY_INSTALLATION_PATH: str = os.path.dirname(os.path.abspath(__file__))

GT_INCLUDE_PATH: str = os.path.abspath(gridtools_cpp.get_include_dir())

GT_CPP_TEMPLATE_DEPTH: int = 1024

GT4PY_COMPILE_OPT_LEVEL = os.environ.get("GT4PY_COMPILE_OPT_LEVEL", "3")
_cxx_compiler_infos = cxx_compiler_defaults(GT4PY_COMPILE_OPT_LEVEL)
_gpu_compiler_configuration = gpu_configuration(GT4PY_COMPILE_OPT_LEVEL)

GT4PY_EXTRA_COMPILE_OPT_FLAGS = os.environ.get("GT4PY_EXTRA_COMPILE_OPT_FLAGS", "")

# Settings dict
GT4PY_EXTRA_COMPILE_ARGS = os.environ.get(
    "GT4PY_EXTRA_COMPILE_ARGS", _cxx_compiler_infos.cxx_compile_flags
)
_extra_compile_args = GT4PY_EXTRA_COMPILE_ARGS.split(" ") if GT4PY_EXTRA_COMPILE_ARGS else []

GT4PY_EXTRA_LINK_ARGS = os.environ.get("GT4PY_EXTRA_LINK_ARGS", "")
extra_link_args = GT4PY_EXTRA_LINK_ARGS.split(" ") if GT4PY_EXTRA_LINK_ARGS else []

# Resolve OpenMP
_enable_open_mp = os.environ.get(
    "GT4PY_CARTESIAN_ENABLE_OPENMP",
    "True" if _cxx_compiler_infos.enable_openmp else "False",
)
GT4PY_CARTESIAN_ENABLE_OPENMP = _enable_open_mp.lower() not in [
    "0",
    "false",
    "off",
]
if GT4PY_CARTESIAN_ENABLE_OPENMP:
    _openmp_cppflags = os.environ.get("OPENMP_CPPFLAGS", _cxx_compiler_infos.open_mp_flag).split()
    _openmp_ldflags = os.environ.get("OPENMP_LDFLAGS", _cxx_compiler_infos.open_mp_flag).split()
else:
    _openmp_cppflags = []
    _openmp_ldflags = []

build_settings: dict[str, Any] = {
    "cuda_bin_path": _gpu_compiler_configuration.binary_path,
    "cuda_include_path": _gpu_compiler_configuration.include_path,
    "cuda_library_path": _gpu_compiler_configuration.library_path,
    "cuda_arch": _gpu_compiler_configuration.arch,
    "gt_include_path": os.environ.get("GT_INCLUDE_PATH", GT_INCLUDE_PATH),
    "openmp": {
        "use_openmp": GT4PY_CARTESIAN_ENABLE_OPENMP,
        "cppflags": _openmp_cppflags,
        "ldflags": _openmp_ldflags,
    },
    "extra_compile_args": {
        "cxx": _extra_compile_args,
        "cuda": _gpu_compiler_configuration.gpu_compile_flags,
    },
    "extra_link_args": extra_link_args,
    "parallel_jobs": multiprocessing.cpu_count(),
    "cpp_template_depth": os.environ.get("GT_CPP_TEMPLATE_DEPTH", GT_CPP_TEMPLATE_DEPTH),
}

if _gpu_compiler_configuration.host_compiler is not None:
    build_settings["extra_compile_args"]["cuda"].append(
        f"-ccbin={_gpu_compiler_configuration.host_compiler}"
    )

cache_settings: dict[str, Any] = {
    "dir_name": os.environ.get("GT_CACHE_DIR_NAME", ".gt_cache"),
    "root_path": os.environ.get("GT_CACHE_ROOT", os.path.abspath(".")),
    "load_retries": int(os.environ.get("GT_CACHE_LOAD_RETRIES", 3)),
    "load_retry_delay": int(os.environ.get("GT_CACHE_LOAD_RETRY_DELAY", 100)),  # unit milliseconds
}

code_settings: dict[str, Any] = {"root_package_name": "_GT_"}

os.environ.setdefault("DACE_CONFIG", os.path.join(os.path.abspath("."), ".dace.conf"))

DACE_DEFAULT_BLOCK_SIZE = os.environ.get("DACE_DEFAULT_BLOCK_SIZE", "64,8,1")

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
import os
from typing import Any, Dict, List, Optional

import gridtools_cpp

from gt4py._core import definitions as core_defs


GT4PY_INSTALLATION_PATH: str = os.path.dirname(os.path.abspath(__file__))

CUDA_ROOT: str = os.environ.get(
    "CUDA_HOME", os.environ.get("CUDA_PATH", os.path.abspath("/usr/local/cuda"))
)

CUDA_HOST_CXX: Optional[str] = os.environ.get("CUDA_HOST_CXX", None)

GT_INCLUDE_PATH: str = os.path.abspath(gridtools_cpp.get_include_dir())

GT_CPP_TEMPLATE_DEPTH: int = 1024

GT4PY_COMPILE_OPT_LEVEL: str = os.environ.get("GT4PY_COMPILE_OPT_LEVEL", "3")
GT4PY_EXTRA_COMPILE_OPT_FLAGS: str = os.environ.get("GT4PY_EXTRA_COMPILE_OPT_FLAGS", "")

# Settings dict
GT4PY_EXTRA_COMPILE_ARGS: str = os.environ.get("GT4PY_EXTRA_COMPILE_ARGS", "")
extra_compile_args: List[str] = (
    list(GT4PY_EXTRA_COMPILE_ARGS.split(" ")) if GT4PY_EXTRA_COMPILE_ARGS else []
)
GT4PY_EXTRA_LINK_ARGS: str = os.environ.get("GT4PY_EXTRA_LINK_ARGS", "")
extra_link_args: List[str] = list(GT4PY_EXTRA_LINK_ARGS.split(" ")) if GT4PY_EXTRA_LINK_ARGS else []

build_settings: Dict[str, Any] = {
    "cuda_bin_path": os.path.join(CUDA_ROOT, "bin"),
    "cuda_include_path": os.path.join(CUDA_ROOT, "include"),
    "cuda_arch": os.environ.get("CUDA_ARCH", None),
    "gt_include_path": os.environ.get("GT_INCLUDE_PATH", GT_INCLUDE_PATH),
    "openmp_cppflags": os.environ.get("OPENMP_CPPFLAGS", "-fopenmp").split(),
    "openmp_ldflags": os.environ.get("OPENMP_LDFLAGS", "-fopenmp").split(),
    "extra_compile_args": {"cxx": extra_compile_args, "cuda": extra_compile_args},
    "extra_link_args": extra_link_args,
    "parallel_jobs": multiprocessing.cpu_count(),
    "cpp_template_depth": os.environ.get("GT_CPP_TEMPLATE_DEPTH", GT_CPP_TEMPLATE_DEPTH),
}
if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
    build_settings["cuda_library_path"] = os.path.join(CUDA_ROOT, "lib")
else:
    build_settings["cuda_library_path"] = os.path.join(CUDA_ROOT, "lib64")

if CUDA_HOST_CXX is not None:
    build_settings["extra_compile_args"]["cuda"].append(f"-ccbin={CUDA_HOST_CXX}")

cache_settings: Dict[str, Any] = {
    "dir_name": os.environ.get("GT_CACHE_DIR_NAME", ".gt_cache"),
    "root_path": os.environ.get("GT_CACHE_ROOT", os.path.abspath(".")),
    "load_retries": int(os.environ.get("GT_CACHE_LOAD_RETRIES", 3)),
    "load_retry_delay": int(os.environ.get("GT_CACHE_LOAD_RETRY_DELAY", 100)),  # unit milliseconds
}

code_settings: Dict[str, Any] = {"root_package_name": "_GT_"}

os.environ.setdefault("DACE_CONFIG", os.path.join(os.path.abspath("."), ".dace.conf"))

DACE_DEFAULT_BLOCK_SIZE: str = os.environ.get("DACE_DEFAULT_BLOCK_SIZE", "64,8,1")

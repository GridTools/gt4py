# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import os
import subprocess

# We rely on setuptools internal distutils - which is on
# a long term support track per the setuptools team
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

from gt4py._core import definitions as core_defs


class CxxCompilerName(enum.Enum):
    DEFAULT = "unknown"
    GNU = "gcc"
    CLANG = "clang"
    APPLE_CLANG = "apple-clang"
    INTEL = "icx"


@dataclasses.dataclass(frozen=True)
class CxxCompilerDefaults:
    name: CxxCompilerName
    """Name identifier of the compiler"""
    open_mp_flag: str
    """OpenMP flag expected on a default install of the compiler"""
    enable_openmp: bool
    """Allow OpenMP acceleration"""
    cxx_compile_flags: str
    """Cxx compile flags"""


def cxx_compiler_defaults(optimization_level: str) -> CxxCompilerDefaults:
    """Return a set of defaults for the compiler flags"""

    # Get the compiler - relying on setuptools
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    # Defaults
    name = CxxCompilerName.DEFAULT
    open_mp_flags = "-fopenmp"
    cxx_flags = ""
    enable_openmp = True

    # FMA is deactivated by default when running -O0
    if optimization_level == "0":
        cxx_flags += "-ffp-contract=off"

    # Query the compiler version string
    try:
        compiler_exe_fullpath = ccompiler.compiler_cxx[0]
        r = subprocess.run([compiler_exe_fullpath, "--version"], capture_output=True, text=True)
        version_name_on_cli = r.stdout.split("\n")[0]
    except AttributeError:
        version_name_on_cli = "default"
    if "gcc" in version_name_on_cli.lower():
        name = CxxCompilerName.GNU
    elif "icx" in version_name_on_cli.lower() or "icpx" in version_name_on_cli.lower():
        name = CxxCompilerName.INTEL
        open_mp_flags = "-qopenmp"
    elif "apple clang" in version_name_on_cli.lower():
        # By default Apple Clang doesn't have OpenMP installed,
        # so we remove the flag and disallow OpenMP altogether
        name = CxxCompilerName.APPLE_CLANG
        open_mp_flags = ""
        enable_openmp = False
    elif "clang" in version_name_on_cli.lower():
        name = CxxCompilerName.CLANG

    return CxxCompilerDefaults(
        name,
        open_mp_flags,
        enable_openmp,
        cxx_flags.strip(),  # be overly cautious for old GCC bad behavior
    )


class GPUCompilerName(enum.Enum):
    NONE = "none"
    NVCC = "nvcc"
    ROCM = "rocm"


@dataclasses.dataclass(frozen=True)
class GPUConfiguration:
    name: GPUCompilerName
    """Name identifier of the compiler"""
    gpu_compile_flags: list[str]
    """Compile flags for device code"""
    binary_path: str
    """Path to binaries for GPU compiler & tools"""
    include_path: str
    """Path to includes for GPU runtime"""
    library_path: str
    """Path to libraries for GPU runtime"""
    arch: str | None
    """Device architecture (None will auto-detect)"""
    host_compiler: str | None
    """Host compiler (None will use CXX)"""


def gpu_configuration(optimization_level: str) -> GPUConfiguration:
    """Retrieve all information to compile and execute GPU device code"""

    # We base our GPU configuration on the CUDA_HOME environment variable
    # as we have mostly supported CUDA historically
    cuda_root = os.environ.get(
        "CUDA_HOME", os.environ.get("CUDA_PATH", os.path.abspath("/usr/local/cuda"))
    )

    # Based on `cupy` we detect a ROCM capable compiler
    if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
        name = GPUCompilerName.ROCM
        library_path = os.path.join(cuda_root, "lib")
    else:
        name = GPUCompilerName.NVCC
        library_path = os.path.join(cuda_root, "lib64")

    # Default arguments for GPU source code
    gpu_compile_flags_default = ""
    if optimization_level == "0":
        # When running -O0 we deactivate FMA
        gpu_compile_flags_default = "-fmad=false"

    extra_cuda_compile_args = os.environ.get(
        "GT4PY_CARTESIAN_EXTRA_CUDA_COMPILE_ARGS", gpu_compile_flags_default
    )
    gpu_compile_flags = extra_cuda_compile_args.split(" ") if extra_cuda_compile_args else []

    return GPUConfiguration(
        name=name,
        gpu_compile_flags=gpu_compile_flags,
        binary_path=os.path.join(cuda_root, "bin"),
        include_path=os.path.join(cuda_root, "include"),
        library_path=library_path,
        arch=os.environ.get("CUDA_ARCH", None),
        host_compiler=os.environ.get("CUDA_HOST_CXX", None),
    )

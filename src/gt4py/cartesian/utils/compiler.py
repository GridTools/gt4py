# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import subprocess
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler


class CxxCompilerNames(enum.Enum):
    DEFAULT = "unknown"
    GNU = "gcc"
    CLANG = "clang"
    APPLE_CLANG = "apple-clang"
    INTEL = "icx"


@dataclasses.dataclass
class CxxCompilerDefaults:
    name: CxxCompilerNames = CxxCompilerNames.DEFAULT
    """Name identifier of the compiler"""
    open_mp_flag: str = "-fopenmp"
    """OpenMP flag expected on a default install of the compiler"""
    cxx_compile_flags: str = ""
    """Cxx compile flags"""


def get_cxx_compiler_defaults(optimization_level: str) -> CxxCompilerDefaults:
    """Return a set of defaults for the compiler flags"""

    # Get the compiler - relying on setuptools
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    name = CxxCompilerNames.DEFAULT
    open_mp_flags = "-fopenmp"
    cxx_flags = ""

    # FMA is deactivated by default when running -O0
    if optimization_level == "0":
        cxx_flags += "-ffp-contract=off "

    # Query the compiler version string
    try:
        compiler_exe_fullpath = ccompiler.compiler_cxx[0]  # type:ignore[attr-defined]
        r = subprocess.run([compiler_exe_fullpath, "--version"], capture_output=True, text=True)
        version_name_on_cli = r.stdout.split("\n")[0]
    except AttributeError:
        version_name_on_cli = "default"
    if "gcc" in version_name_on_cli.lower():
        name = CxxCompilerNames.GNU
    elif "icx" in version_name_on_cli.lower():
        name = CxxCompilerNames.INTEL
        open_mp_flags = "-qopenmp"
    elif "apple clang" in version_name_on_cli.lower():
        # By default Apple Clang doesn;t have OpenMP install,
        # so we remove the flag
        name = CxxCompilerNames.APPLE_CLANG
        open_mp_flags = ""
    elif "clang" in version_name_on_cli.lower():
        name = CxxCompilerNames.CLANG

    return CxxCompilerDefaults(name, open_mp_flags, cxx_flags)

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Description of the toolchain that turns generated code into binaries.

The persistent build cache keys a program by its generated source and the GT4Py version. The
compile steps fold the toolchain fingerprint into their build context id as well, so that a
different compiler, device architecture or code-generating library lands in another cache folder
instead of reusing binaries built by the old toolchain. Only versions are recorded: environment
variables that change code generation are not, and still need a separate cache directory.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import platform
import shutil
import subprocess
from typing import Final

from gt4py._core import definitions as core_defs
from gt4py.next import fingerprinting
from gt4py.next.otf.compilation import common


#: CMake language name -> (environment variable CMake reads the compiler from, default executable).
_COMPILERS: Final[dict[str, tuple[str, str]]] = {
    "CXX": ("CXX", "c++"),
    "CUDA": ("CUDACXX", "nvcc"),
    "HIP": ("HIPCXX", "hipcc"),
}


def _compiler_version(env_var: str, default: str) -> str | None:
    executable = shutil.which(os.environ.get(env_var) or default)
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # nvcc prints a banner first; its version is on the "release" line
    return next((line for line in lines if "release" in line), lines[0] if lines else None)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def device_language(device_type: core_defs.DeviceType) -> str:
    """CMake language name of the device code compiled for ``device_type``."""
    match device_type:
        case core_defs.DeviceType.CUDA:
            return "CUDA"
        case core_defs.DeviceType.ROCM:
            return "HIP"
        case _:
            return "CXX"


@functools.cache
def describe(
    language: str, distributions: tuple[str, ...] = ()
) -> tuple[tuple[str, str | None], ...]:
    """
    Toolchain compiling ``language`` (a CMake language name), plus the given Python distributions.

    Always the machine, Python and the C++ compiler; a device language adds its compiler and the
    device architecture. Computed once per process.
    """
    entries: dict[str, str | None] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "CXX": _compiler_version(*_COMPILERS["CXX"]),
    }
    if language != "CXX":
        entries[language] = _compiler_version(*_COMPILERS[language])
        entries["device_arch"] = common.get_device_arch()
    entries.update((name, _distribution_version(name)) for name in distributions)
    return tuple(entries.items())


def fingerprint(language: str, distributions: tuple[str, ...] = ()) -> str:
    return fingerprinting.strict_fingerprinter(describe(language, distributions))

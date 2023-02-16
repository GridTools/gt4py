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

import os
import subprocess
from pathlib import Path

import pytest


def _source_dir():
    return Path(__file__).resolve().parent


def _build_dir(backend_dir: str):
    return _source_dir() / f"build_{backend_dir}"


def _execute_cmake(backend_str: str):
    build_dir = _build_dir(backend_str)
    build_dir.mkdir(exist_ok=True)
    cmake = ["cmake", "-B", build_dir, f"-DBACKEND={backend_str}"]
    subprocess.run(cmake, cwd=_source_dir(), check=True)


def _get_available_cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _execute_build(backend_str: str):
    build = [
        "cmake",
        "--build",
        _build_dir(backend_str),
        "--parallel",
        str(_get_available_cpu_count()),
    ]
    subprocess.run(build, check=True)


def _execute_ctest(backend_str: str):
    ctest = "ctest"
    subprocess.run(ctest, cwd=_build_dir(backend_str), check=True)


backends = ["naive"]
try:
    import cupy  # TODO actually cupy is not the requirement but a CUDA compiler...

    backends.append("gpu")
except ImportError:
    pass


@pytest.mark.parametrize("backend_str", backends)
def test_driver_cpp_backends(backend_str):
    _execute_cmake(backend_str)
    _execute_build(backend_str)
    _execute_ctest(backend_str)

import os
import subprocess
from pathlib import Path

import pytest


def _source_dir():
    return Path(__file__).resolve().parent


def _build_dir(backend_dir: str):
    return _source_dir().joinpath(f"build_{backend_dir}")


def _execute_cmake(backend_str: str):
    build_dir = _build_dir(backend_str)
    os.makedirs(build_dir, exist_ok=True)
    cmake = ["cmake", "-B", build_dir, f"-DBACKEND={backend_str}"]
    cmake_proc = subprocess.run(cmake, cwd=_source_dir())
    cmake_proc.check_returncode()
    return True


def _execute_build(backend_str: str):
    build = [
        "cmake",
        "--build",
        _build_dir(backend_str),
        "--parallel",
        str(len(os.sched_getaffinity(0))),
    ]
    build_proc = subprocess.run(build)
    build_proc.check_returncode()
    return True


def _execute_ctest(backend_str: str):
    ctest = "ctest"
    ctest_proc = subprocess.run(ctest, cwd=_build_dir(backend_str))
    ctest_proc.check_returncode()
    return True


backends = ["naive"]
try:
    import cupy  # TODO actually cupy is not the requirement but a CUDA compiler...

    backends.append("gpu")
except ImportError:
    pass


@pytest.mark.parametrize("backend_str", backends)
def test_driver_cpp_backends(backend_str):
    assert _execute_cmake(backend_str)
    assert _execute_build(backend_str)
    assert _execute_ctest(backend_str)

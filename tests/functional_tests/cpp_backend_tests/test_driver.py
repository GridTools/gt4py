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


def _execute_build(backend_str: str):
    build = [
        "cmake",
        "--build",
        _build_dir(backend_str),
        "--parallel",
        str(len(os.sched_getaffinity(0))),
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

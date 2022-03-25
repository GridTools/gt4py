import os
import subprocess
from pathlib import Path

import pytest


def _source_dir():
    return Path(__file__).resolve().parent


def _build_dir(backend_dir: str):
    return os.path.join(_source_dir(), f"build_{backend_dir}")


def _execute_cmake(backend_str: str):
    build_dir = _build_dir(backend_str)
    os.makedirs(build_dir, exist_ok=True)
    cmake = ["cmake", "-B", build_dir, f"-DBACKEND={backend_str}"]
    cmake_proc = subprocess.Popen(cmake, cwd=_source_dir())
    return cmake_proc.wait() == 0


def _execute_build(backend_str: str):
    build = [
        "cmake",
        "--build",
        _build_dir(backend_str),
        "--parallel",
        str(len(os.sched_getaffinity(0))),
    ]
    build_proc = subprocess.Popen(build)
    return build_proc.wait() == 0


def _execute_ctest(backend_str: str):
    ctest = "ctest"
    ctest_proc = subprocess.Popen(ctest, cwd=_build_dir(backend_str))
    return ctest_proc.wait() == 0


@pytest.mark.parametrize("backend_str", ["naive", "gpu"])
def test_driver(backend_str):
    assert _execute_cmake(backend_str)
    assert _execute_build(backend_str)
    assert _execute_ctest(backend_str)

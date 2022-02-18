import os
import subprocess
from pathlib import Path


def _source_dir():
    return Path(__file__).resolve().parent


def _build_dir():
    return os.path.join(_source_dir(), "build")


def _execute_cmake():
    build_dir = _build_dir()
    os.makedirs(build_dir, exist_ok=True)
    cmake = ["cmake", "-B", _build_dir()]
    cmake_proc = subprocess.Popen(cmake, cwd=_source_dir())
    return cmake_proc.wait() == 0


def _execute_ctest():
    ctest = "ctest"
    ctest_proc = subprocess.Popen(ctest, cwd=_build_dir())
    return ctest_proc.wait() == 0


def test_driver():
    assert _execute_cmake()
    assert _execute_ctest()

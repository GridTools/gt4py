# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import sys

import pytest

from gt4py.eve import formatting


UNFORMATTED_PYTHON = "def f( a,b ):\n  return a+b\n"
UNFORMATTED_CPP = "int f( int a,int b ){return a+b;}\n"


# -- Python tests --
def test_format_python_source():
    pytest.importorskip("black")
    assert formatting.format_python_source(UNFORMATTED_PYTHON) == (
        "def f(a, b):\n    return a + b\n"
    )


def test_format_python_source_line_length():
    pytest.importorskip("black")
    source = "result = function_name(first_argument, second_argument)\n"
    assert formatting.format_python_source(source) == source
    assert formatting.format_python_source(source, line_length=40) == (
        "result = function_name(\n    first_argument, second_argument\n)\n"
    )


def test_format_python_source_invalid_input():
    pytest.importorskip("black")
    source = "def f(:\n"
    assert formatting.format_python_source(source) == source


def test_format_python_source_black_failure(monkeypatch):
    black = pytest.importorskip("black")

    def failing_format_str(*args, **kwargs):
        raise RuntimeError("internal black error")

    monkeypatch.setattr(black, "format_str", failing_format_str)
    assert formatting.format_python_source(UNFORMATTED_PYTHON) == UNFORMATTED_PYTHON


def test_format_python_source_unsupported_interpreter(monkeypatch):
    black = pytest.importorskip("black")
    # Simulate a `black` version that does not know the running interpreter.
    monkeypatch.setattr(black, "TargetVersion", {})
    assert formatting.format_python_source(UNFORMATTED_PYTHON) == UNFORMATTED_PYTHON


def test_format_python_source_without_black(monkeypatch):
    monkeypatch.setitem(sys.modules, "black", None)
    assert formatting.format_python_source(UNFORMATTED_PYTHON) == UNFORMATTED_PYTHON


# -- C++ tests --
@pytest.fixture(autouse=True)
def clear_clang_format_cache():
    formatting._get_clang_format.cache_clear()
    yield
    formatting._get_clang_format.cache_clear()


@pytest.mark.skipif(shutil.which("clang-format") is None, reason="clang-format not available")
def test_format_cpp_source(monkeypatch):
    monkeypatch.delenv("CLANG_FORMAT_EXECUTABLE", raising=False)
    assert (
        formatting.format_cpp_source(UNFORMATTED_CPP) == "int f(int a, int b) { return a + b; }\n"
    )


def test_format_cpp_source_missing_executable(monkeypatch):
    monkeypatch.setenv("CLANG_FORMAT_EXECUTABLE", "/nonexistent")
    assert formatting.format_cpp_source(UNFORMATTED_CPP) == UNFORMATTED_CPP


@pytest.mark.skipif(shutil.which("false") is None, reason="`false` executable not available")
def test_format_cpp_source_failing_executable(monkeypatch):
    monkeypatch.setenv("CLANG_FORMAT_EXECUTABLE", "false")
    assert formatting.format_cpp_source(UNFORMATTED_CPP) == UNFORMATTED_CPP


@pytest.mark.skipif(shutil.which("false") is None, reason="`false` executable not available")
def test_format_cpp_source_formatting_failure(monkeypatch):
    # The executable passes the availability probe but fails when formatting.
    monkeypatch.setattr(formatting, "_get_clang_format", lambda: "false")
    assert formatting.format_cpp_source(UNFORMATTED_CPP) == UNFORMATTED_CPP


def test_format_cpp_source_probes_executable_once(monkeypatch):
    monkeypatch.setenv("CLANG_FORMAT_EXECUTABLE", "/nonexistent")
    formatting.format_cpp_source(UNFORMATTED_CPP)
    formatting.format_cpp_source(UNFORMATTED_CPP)
    assert formatting._get_clang_format.cache_info().misses == 1

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

import numpy as np
import pytest

import gt4py.next.otf.binding.cpp_interface as cpp
from gt4py.eve.codegen import format_source
from gt4py.next.otf.binding import interface


def test_render_types():
    rendered = cpp.render_python_type(float)
    assert rendered == "double"


@pytest.fixture
def function_scalar_example():
    return interface.Function(
        name="example",
        parameters=[
            interface.ScalarParameter("a", np.dtype(float)),
            interface.ScalarParameter("b", np.dtype(int)),
        ],
    )


def test_render_function_declaration_scalar(function_scalar_example):
    rendered = format_source(
        "cpp", cpp.render_function_declaration(function_scalar_example, "return;"), style="LLVM"
    )
    expected = format_source(
        "cpp",
        """\
    decltype(auto) example(double a, long b) {
        return;
    }\
    """,
        style="LLVM",
    )
    assert rendered == expected


def test_render_function_call_scalar(function_scalar_example):
    rendered = format_source(
        "cpp",
        cpp.render_function_call(function_scalar_example, args=["13.6", "get_arg()"]),
        style="LLVM",
    )
    expected = format_source("cpp", """example(13.6, get_arg())""", style="LLVM")
    assert rendered == expected


@pytest.fixture
def function_buffer_example():
    return interface.Function(
        name="example",
        parameters=[
            interface.BufferParameter("a_buf", 2, float),
            interface.BufferParameter("b_buf", 1, int),
        ],
    )


def test_render_function_declaration_buffer(function_buffer_example):
    rendered = format_source(
        "cpp", cpp.render_function_declaration(function_buffer_example, "return;"), style="LLVM"
    )
    expected = format_source(
        "cpp",
        """\
    template <class BufferT0, class BufferT1>
    decltype(auto) example(BufferT0&& a_buf, BufferT1&& b_buf) {
        return;
    }\
    """,
        style="LLVM",
    )
    assert rendered == expected


def test_render_function_call_buffer(function_buffer_example):
    rendered = format_source(
        "cpp",
        cpp.render_function_call(function_buffer_example, args=["get_arg_1()", "get_arg_2()"]),
        style="LLVM",
    )
    expected = format_source("cpp", """example(get_arg_1(), get_arg_2())""", style="LLVM")
    assert rendered == expected

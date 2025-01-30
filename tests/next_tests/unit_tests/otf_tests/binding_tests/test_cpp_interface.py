# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx
import gt4py.next.otf.binding.cpp_interface as cpp
import gt4py.next.type_system.type_specifications as ts
from gt4py.eve.codegen import format_source
from gt4py.next.otf.binding import interface


@pytest.fixture
def function_scalar_example():
    return interface.Function(
        name="example",
        parameters=[
            interface.Parameter(name="a", type_=ts.ScalarType(ts.ScalarKind.FLOAT64)),
            interface.Parameter(name="b", type_=ts.ScalarType(ts.ScalarKind.INT64)),
        ],
    )


def test_render_function_declaration_scalar(function_scalar_example):
    rendered = format_source(
        "cpp", cpp.render_function_declaration(function_scalar_example, "return;"), style="LLVM"
    )
    expected = format_source(
        "cpp",
        """\
decltype(auto) example(double a, std::int64_t b) {
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
            interface.Parameter(
                name="a_buf",
                type_=ts.FieldType(
                    dims=[gtx.Dimension("foo"), gtx.Dimension("bar")],
                    dtype=ts.ScalarType(ts.ScalarKind.FLOAT64),
                ),
            ),
            interface.Parameter(
                name="b_buf",
                type_=ts.FieldType(
                    dims=[gtx.Dimension("foo")], dtype=ts.ScalarType(ts.ScalarKind.INT64)
                ),
            ),
        ],
    )


def test_render_function_declaration_buffer(function_buffer_example):
    rendered = format_source(
        "cpp", cpp.render_function_declaration(function_buffer_example, "return;"), style="LLVM"
    )
    expected = format_source(
        "cpp",
        """\
template <class ArgT0, class ArgT1>
        decltype(auto) example(ArgT0&& a_buf, ArgT1&& b_buf) {
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


@pytest.fixture
def function_tuple_example():
    return interface.Function(
        name="example",
        parameters=[
            interface.Parameter(
                name="a_buf",
                type_=ts.TupleType(
                    types=[
                        ts.FieldType(
                            dims=[gtx.Dimension("foo"), gtx.Dimension("bar")],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT64),
                        ),
                        ts.FieldType(
                            dims=[gtx.Dimension("foo"), gtx.Dimension("bar")],
                            dtype=ts.ScalarType(ts.ScalarKind.FLOAT64),
                        ),
                    ]
                ),
            )
        ],
    )


def test_render_function_declaration_tuple(function_tuple_example):
    rendered = format_source(
        "cpp", cpp.render_function_declaration(function_tuple_example, "return;"), style="LLVM"
    )
    expected = format_source(
        "cpp",
        """\
template <class ArgT0>
        decltype(auto) example(ArgT0&& a_buf) {
        return;
    }\
""",
        style="LLVM",
    )
    assert rendered == expected


def test_render_function_call_tuple(function_tuple_example):
    rendered = format_source(
        "cpp", cpp.render_function_call(function_tuple_example, args=["get_arg_1()"]), style="LLVM"
    )
    expected = format_source("cpp", """example(get_arg_1())""", style="LLVM")
    assert rendered == expected

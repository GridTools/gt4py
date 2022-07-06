import pytest

import functional.fencil_processors.cpp as cpp
import functional.fencil_processors.defs as defs
from eve.codegen import format_source


def test_render_types():
    rendered = cpp.render_python_type(float)
    assert rendered == "double"


@pytest.fixture
def function_scalar_example():
    return defs.Function(
        name="example",
        parameters=[defs.ScalarParameter("a", float), defs.ScalarParameter("b", int)],
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
    return defs.Function(
        name="example",
        parameters=[defs.BufferParameter("a_buf", 2, float), defs.BufferParameter("b_buf", 1, int)],
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

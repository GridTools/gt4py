# -*- coding: utf-8 -*-
import ast
import inspect
import textwrap

import pytest

from gtc_unstructured.frontend import ast_node_matcher as anm
from gtc_unstructured.frontend.frontend import GTScriptCompilationTask

from . import stencil_definitions


class TestAstNodeMatcher:
    def test_simple(self):
        ast_node = ast.Name(id="some_id")
        pattern_node = ast.Name(id="some_id")
        assert anm.match(ast_node, pattern_node)

    def test_simple_no_match(self):
        ast_node = ast.Name(id="some_id")
        pattern_node = ast.Name(id="some_id1")
        assert not anm.match(ast_node, pattern_node)

    def test_capture_simple(self):
        ast_node = ast.Name(id="some_id")
        pattern_node = ast.Name(id=anm.Capture("id"))
        captures = {}
        matches = anm.match(ast_node, pattern_node, captures)
        assert matches
        assert captures["id"] == "some_id"

    def test_capture_simple_no_match(self):
        ast_node = ast.arg(arg="some_id")
        pattern_node = ast.Name(id=anm.Capture("id"))
        matches = anm.match(ast_node, pattern_node)
        assert not matches

    def test_capture_nested(self):
        ast_node = ast.List(
            elts=[ast.Constant(value=1), ast.Constant(value=2), ast.Constant(value=3)]
        )
        pattern_node = ast.List(
            elts=[
                ast.Constant(value=anm.Capture("first")),
                ast.Constant(value=anm.Capture("second")),
                ast.Constant(value=anm.Capture("third")),
            ]
        )
        captures = {}
        matches = anm.match(ast_node, pattern_node, captures)
        assert matches
        assert captures["first"] == 1
        assert captures["second"] == 2
        assert captures["third"] == 3
        pass

    def test_capture_from_definition(self):
        def some_func(arg1, arg2):
            return arg1

        ast_node = ast.parse(textwrap.dedent(inspect.getsource(some_func))).body[0]
        pattern_node = ast.FunctionDef(
            name=anm.Capture("function_name"),
            args=ast.arguments(
                args=[ast.arg(arg=anm.Capture("arg1_name")), ast.arg(arg=anm.Capture("arg2_name"))]
            ),
            body=[ast.Return(value=ast.Name(anm.Capture("return_name")))],
        )
        captures = {}
        matches = anm.match(ast_node, pattern_node, captures)
        assert matches
        assert captures["function_name"] == "some_func"
        assert captures["arg1_name"] == "arg1"
        assert captures["arg2_name"] == "arg2"
        assert captures["return_name"] == "arg1"

    def test_field_in_pattern_but_not_ast_no_match(self):
        ast_node = ast.Slice(lower=ast.Name(id="a"))
        pattern_node = ast.Slice(lower=ast.Name(id="a"), upper=ast.Name(id="b"))

        matches = anm.match(ast_node, pattern_node)

        assert not matches

    def test_optional(self):
        ast_node = ast.Name()
        pattern_node = ast.Name(id=anm.Capture("id", default="some_name"))

        captures = {}
        matches = anm.match(ast_node, pattern_node, captures)

        assert matches
        assert captures["id"] == "some_name"

    def test_arr(self):
        ast_node = ast.Tuple(elts=[ast.Name(id="1"), ast.Name(id="2")])
        pattern_node = ast.Tuple(elts=[ast.Name(id="1"), ast.Name(id="2")])

        assert anm.match(ast_node, pattern_node)

    def test_arr_no_match(self):
        ast_node = ast.Tuple(elts=[ast.Name(id="1")])
        pattern_node = ast.Tuple(elts=[ast.Name(id="1"), ast.Name(id="2")])

        assert not anm.match(ast_node, pattern_node)

    def test_arr_optional(self):
        ast_node = ast.Tuple(elts=[ast.Name(id="1")])
        pattern_node = ast.Tuple(
            elts=[ast.Name(id="1"), ast.Name(id=anm.Capture("id", default="some_default"))]
        )

        captures = {}
        matches = anm.match(ast_node, pattern_node, captures)

        assert matches
        assert captures["id"] == "some_default"


@pytest.fixture(params=stencil_definitions.valid_stencils)
def valid_stencil(request):
    return getattr(stencil_definitions, request.param)


def test_code_generation_for_valid_stencils(valid_stencil):
    GTScriptCompilationTask(valid_stencil).generate()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import textwrap

from functional.ffront.single_static_assign import SingleStaticAssignPass


def ssaify_string(code: str) -> ast.AST:
    return SingleStaticAssignPass().visit(ast.parse(textwrap.dedent(code)))


def test_sequence():
    """Overwriting the same variable is the simplest case"""
    ssa_ast = ssaify_string(
        """
        tmp = 1
        tmp = 2
        tmp = 3
        """
    )

    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "tmp_0 = 1"
    assert lines[1].strip() == "tmp_1 = 2"
    assert lines[2].strip() == "tmp_2 = 3"


def test_self_on_rhs():
    """Occurrences of the target name on the rhs should be handled ok."""
    ssa_ast = ssaify_string(
        """
        tmp = 1
        tmp = tmp + 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "tmp_0 = 1"
    assert lines[1].strip() == "tmp_1 = tmp_0 + 1"


def test_multi_assign():
    """Multiple assign targets get handled sequentially."""
    ssa_ast = ssaify_string(
        """
        a = a = b = a = b = 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "a_0 = a_1 = b_0 = a_2 = b_1 = 1"


def test_external_name_values():
    """A name that has not been assigned to is not incremented."""
    ssa_ast = ssaify_string(
        """
        a = inp
        a = a + inp
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "a_0 = inp"
    assert lines[1].strip() == "a_1 = a_0 + inp"


def test_overwrite_external():
    """Once an external name is assigned to it is treated normally."""
    ssa_ast = ssaify_string(
        """
        a = inp
        inp = a + inp
        b = inp
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "a_0 = inp"
    assert lines[1].strip() == "inp_0 = a_0 + inp"
    assert lines[2].strip() == "b_0 = inp_0"


def test_unpacking_swap():
    """Handling unpacking correctly allows to sequentialize unpacking assignments later."""
    ssa_ast = ssaify_string(
        """
        a = 5
        b = 1
        b, a = a, b
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0].strip() == "a_0 = 5"
    assert lines[1].strip() == "b_0 = 1"
    assert lines[2].strip() == "(b_1, a_1) = (a_0, b_0)"

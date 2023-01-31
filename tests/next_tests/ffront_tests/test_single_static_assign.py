#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

#
import ast
import textwrap

from gt4py.next.ffront.ast_passes.single_static_assign import SingleStaticAssignPass


def ssaify_string(code: str) -> ast.AST:
    return SingleStaticAssignPass.apply(ast.parse(textwrap.dedent(code)))


def test_sequence():
    """Overwriting the same variable is the simplest case."""
    ssa_ast = ssaify_string(
        """
        tmp = 1
        tmp = 2
        tmp = 3
        """
    )

    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == "tmp__0 = 1"
    assert lines[1] == "tmp__1 = 2"
    assert lines[2] == "tmp__2 = 3"


def test_self_on_rhs():
    """Occurrences of the target name on the rhs should be handled ok."""
    ssa_ast = ssaify_string(
        """
        tmp = 1
        tmp = tmp + 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == "tmp__0 = 1"
    assert lines[1] == "tmp__1 = tmp__0 + 1"


def test_multi_assign():
    """Multiple assign targets get handled sequentially."""
    ssa_ast = ssaify_string(
        """
        a = a = b = a = b = 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == "a__0 = a__1 = b__0 = a__2 = b__1 = 1"


def test_external_name_values():
    """A name that has not been assigned to is not incremented."""
    ssa_ast = ssaify_string(
        """
        a = inp
        a = a + inp
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == "a__0 = inp"
    assert lines[1] == "a__1 = a__0 + inp"


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
    assert lines[0] == "a__0 = inp"
    assert lines[1] == "inp__0 = a__0 + inp"
    assert lines[2] == "b__0 = inp__0"


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
    assert lines[0] == "a__0 = 5"
    assert lines[1] == "b__0 = 1"
    assert lines[2] == "(b__1, a__1) = (a__0, b__0)"


def test_annotated_assign():
    """The name of type annotations should not be treated as an assignment target."""
    lines = ast.unparse(
        ssaify_string(
            """
            a: int = 5
            """
        )
    ).splitlines()
    assert lines[0] == "a__0: int = 5"


def test_empty_annotated_assign():
    lines = ast.unparse(
        ssaify_string(
            """
            a = 0
            a: int
            b = a
            """
        )
    ).splitlines()
    assert lines[0] == "a__0 = 0"
    assert lines[1] == "a__1: int"
    assert lines[2] == "b__0 = a__0"

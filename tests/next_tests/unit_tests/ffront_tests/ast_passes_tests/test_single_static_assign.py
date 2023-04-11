#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


def test_if():
    lines = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 1
            else:
                a = 2
            a = 3
            """
        )
    ).splitlines()

    assert lines[1] == "    a__0 = 1"
    assert lines[3] == "    a__0 = 2"
    assert lines[4] == "a__1 = 3"


def test_if_variable_condition():
    result = ast.unparse(
        ssaify_string(
            """
            if b:
                a = 1
            """
        )
    )

    expected = textwrap.dedent(
        """
        if b:
            a__0 = 1
        """
    ).strip()

    assert result == expected


def test_nested_if():
    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 1
            else:
                a = 2
                if True:
                    a = 3
                a = 4
            a = 5
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 1
            a__2 = a__0
        else:
            a__0 = 2
            if True:
                a__1 = 3
            else:
                a__1 = a__0
            a__2 = 4
        a__3 = 5
    """
    ).strip()

    assert result == expected


def test_nested_if_chain():
    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 1
            else:
                a = 2
                if True:
                    a = a+1
                a = a+1
            a = a+1
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 1
            a__2 = a__0
        else:
            a__0 = 2
            if True:
                a__1 = a__0 + 1
            else:
                a__1 = a__0
            a__2 = a__1 + 1
        a__3 = a__2 + 1
    """
    ).strip()

    assert result == expected


def test_if_branch_local():

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 0
                a = a + 1
            else:
                b = 1
                b = b + 1
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 0
            a__1 = a__0 + 1
        else:
            b__0 = 1
            b__1 = b__0 + 1
        """
    ).strip()

    assert result == expected


def test_if_only_one_branch():

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 0
            b = a
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 0
        b__0 = a
        """
    ).strip()

    assert result == expected


def test_if_only_one_branch_other():

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 0
            else:
                c = 1
            b = c
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 0
        else:
            c__0 = 1
        b__0 = c
        """
    ).strip()

    assert result == expected


def test_if_nested_all_branches_defined():

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                if True:
                    a = 0
                    a = a + 1
                    a = a + 1
                    a = a + 1
                    a = a + 1
                else:
                    a = 0
                    a = a + 1
                    a = a + 1
            else:
                a = 0
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            if True:
                a__0 = 0
                a__1 = a__0 + 1
                a__2 = a__1 + 1
                a__3 = a__2 + 1
                a__4 = a__3 + 1
            else:
                a__0 = 0
                a__1 = a__0 + 1
                a__2 = a__1 + 1
                a__4 = a__2
        else:
            a__0 = 0
            a__4 = a__0
        """
    ).strip()

    assert result == expected


def test_elif_all_branches_defined():

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = 0
                a = a + 1
                a = a + 1
                a = a + 1
                a = a + 1
            elif True:
                a = 0
                a = a + 1
                a = a + 1
            else:
                a = 0
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = 0
            a__1 = a__0 + 1
            a__2 = a__1 + 1
            a__3 = a__2 + 1
            a__4 = a__3 + 1
        else:
            if True:
                a__0 = 0
                a__1 = a__0 + 1
                a__2 = a__1 + 1
            else:
                a__0 = 0
                a__2 = a__0
            a__4 = a__2
        """
    ).strip()

    assert result == expected


def test_nested_ifs_single_change():

    result = ast.unparse(
        ssaify_string(
            """
            a = 0
            if True:
                b = 0
            elif True:
                if True:
                    b = 1
                else:
                    if True:
                        b = 2
                        # because of this nested change, all branches need additions
                        a = a + 1
                    else:
                        b = 3
            else:
                b = 4
            """
        )
    )

    expected = textwrap.dedent(
        """
        a__0 = 0
        if True:
            b__0 = 0
            a__1 = a__0
        elif True:
            if True:
                b__0 = 1
                a__1 = a__0
            elif True:
                b__0 = 2
                a__1 = a__0 + 1
            else:
                b__0 = 3
                a__1 = a__0
        else:
            b__0 = 4
            a__1 = a__0
        """
    ).strip()

    assert result == expected


def test_if_one_sided_inside_function():

    result = ast.unparse(
        ssaify_string(
            """
            def f(a):
                if True:
                    a = a + 1
                return a
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f(a):
            if True:
                a__0 = a + 1
            else:
                a__0 = a
            return a__0
        """
    ).strip()

    assert result == expected


def test_if_preservers_definite_assignment_analysis1():

    result = ast.unparse(
        ssaify_string(
            """
            def f():
                if True:
                    a = 1
                else:
                    b = 0
                return a, b
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f():
            if True:
                a__0 = 1
            else:
                b__0 = 0
            return (a, b)
        """
    ).strip()

    assert result == expected


def test_if_preservers_definite_assignment_analysis2():

    result = ast.unparse(
        ssaify_string(
            """
            def f():
                if True:
                    a = 1
                else:
                    a = 0
                return a, b
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f():
            if True:
                a__0 = 1
            else:
                a__0 = 0
            return (a__0, b)
        """
    ).strip()

    assert result == expected


def test_if_preservers_definite_assignment_analysis3():

    result = ast.unparse(
        ssaify_string(
            """
            def f():
                a = 0
                if True:
                    a = 1
                return a, b
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f():
            a__0 = 0
            if True:
                a__1 = 1
            else:
                a__1 = a__0
            return (a__1, b)
        """
    ).strip()

    assert result == expected


def test_broken_collisions():
    # Known bug of the current SSA implementation

    result = ast.unparse(
        ssaify_string(
            """
            a = a__0 + 1
            return a, a__0
            """
        )
    )

    expected = textwrap.dedent(
        """
        a__0 = a__0 + 1
        return (a__0, a__0)
        """
    ).strip()

    assert result == expected


def test_collision_function_parameters():
    # An earlier version couldn't handle this case correctly

    result = ast.unparse(
        ssaify_string(
            """
            def f(a, a__0):
                return a__0
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f(a, a__0):
            return a__0
        """
    ).strip()

    assert result == expected


def test_broken_if():
    # Known bug of the current SSA implementation

    result = ast.unparse(
        ssaify_string(
            """
            if True:
                a = a + 1
            return a
            """
        )
    )

    expected = textwrap.dedent(
        """
        if True:
            a__0 = a + 1
        return a
        """
    ).strip()

    assert result == expected


def test_annotated_assign():

    result = ast.unparse(
        ssaify_string(
            """
            a: int # annotations always apply to the next assignment
            a = a + 1
            """
        )
    )

    expected = textwrap.dedent(
        """
        a__0: int
        a__0 = a + 1
        """
    ).strip()

    assert result == expected


def test_if_true_branch_returns():

    result = ast.unparse(
        ssaify_string(
            """
            def f(a, b):
                if True:
                    a = a + 1
                    d = 5
                    return a
                else:
                    b = b + 1
                    e = 6
                return a, b, d, e
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f(a, b):
            if True:
                a__0 = a + 1
                d__0 = 5
                return a__0
            else:
                b__0 = b + 1
                e__0 = 6
            return (a, b__0, d, e__0)
        """
    ).strip()

    assert result == expected


def test_if_false_branch_returns():

    result = ast.unparse(
        ssaify_string(
            """
            def f(a, b):
                b = a + 3
                if True:
                    b = b + 1
                    e = 6
                else:
                    a = a + 1
                    d = 5
                    return a
                return a, b, d, e
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f(a, b):
            b__0 = a + 3
            if True:
                b__1 = b__0 + 1
                e__0 = 6
            else:
                a__0 = a + 1
                d__0 = 5
                return a__0
            return (a, b__1, d, e__0)
        """
    ).strip()

    assert result == expected


def test_if_both_branches_return():

    result = ast.unparse(
        ssaify_string(
            """
            def f(a, b):
                if True:
                    b = b + 1
                    e = 6
                    return e
                else:
                    a = a + 1
                    d = 5
                    return a
                return a, b, d, e # this is dead-code
            """
        )
    )

    expected = textwrap.dedent(
        """
        def f(a, b):
            if True:
                b__0 = b + 1
                e__0 = 6
                return e__0
            else:
                a__0 = a + 1
                d__0 = 5
                return a__0
            return (a, b, d, e)
        """
    ).strip()

    assert result == expected


def test_if_nested_returns():

    result = ast.unparse(
        ssaify_string(
            """
            def f(a, b):
                if True:
                    b = b + 1
                    e = 6
                    if e == b:
                        return e
                    else:
                        return b
                else:
                    a = a + 1
                    d = 5
                return a, b, d, e
            """
        )
    )

    expected = textwrap.dedent(
        """
            def f(a, b):
                if True:
                    b__0 = b + 1
                    e__0 = 6
                    if e__0 == b__0:
                        return e__0
                    else:
                        return b__0
                else:
                    a__0 = a + 1
                    d__0 = 5
                return (a__0, b, d__0, e)
        """
    ).strip()

    assert result == expected

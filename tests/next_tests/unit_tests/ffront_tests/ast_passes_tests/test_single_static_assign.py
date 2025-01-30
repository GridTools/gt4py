#!/usr/bin/env python

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

#
import ast
import textwrap

from gt4py.next.ffront.ast_passes.single_static_assign import (
    _UNIQUE_NAME_SEPARATOR as SEP,
    SingleStaticAssignPass,
)


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
    assert lines[0] == f"tmp{SEP}0 = 1"
    assert lines[1] == f"tmp{SEP}1 = 2"
    assert lines[2] == f"tmp{SEP}2 = 3"


def test_self_on_rhs():
    """Occurrences of the target name on the rhs should be handled ok."""
    ssa_ast = ssaify_string(
        """
        tmp = 1
        tmp = tmp + 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == f"tmp{SEP}0 = 1"
    assert lines[1] == f"tmp{SEP}1 = tmp{SEP}0 + 1"


def test_multi_assign():
    """Multiple assign targets get handled sequentially."""
    ssa_ast = ssaify_string(
        """
        a = a = b = a = b = 1
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == f"a{SEP}0 = a{SEP}1 = b{SEP}0 = a{SEP}2 = b{SEP}1 = 1"


def test_external_name_values():
    """A name that has not been assigned to is not incremented."""
    ssa_ast = ssaify_string(
        """
        a = inp
        a = a + inp
        """
    )
    lines = ast.unparse(ssa_ast).split("\n")
    assert lines[0] == f"a{SEP}0 = inp"
    assert lines[1] == f"a{SEP}1 = a{SEP}0 + inp"


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
    assert lines[0] == f"a{SEP}0 = inp"
    assert lines[1] == f"inp{SEP}0 = a{SEP}0 + inp"
    assert lines[2] == f"b{SEP}0 = inp{SEP}0"


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
    assert lines[0] == f"a{SEP}0 = 5"
    assert lines[1] == f"b{SEP}0 = 1"
    assert lines[2] in [
        f"(b{SEP}1, a{SEP}1) = (a{SEP}0, b{SEP}0)",
        f"b{SEP}1, a{SEP}1 = (a{SEP}0, b{SEP}0)",
    ]  # unparse produces different parentheses in different Python versions


def test_annotated_assign():
    """The name of type annotations should not be treated as an assignment target."""
    lines = ast.unparse(
        ssaify_string(
            """
            a: int = 5
            """
        )
    ).splitlines()
    assert lines[0] == f"a{SEP}0: int = 5"


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
    assert lines[0] == f"a{SEP}0 = 0"
    assert lines[1] == f"a{SEP}1: int"
    assert lines[2] == f"b{SEP}0 = a{SEP}0"


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

    assert lines[1] == f"    a{SEP}0 = 1"
    assert lines[3] == f"    a{SEP}0 = 2"
    assert lines[4] == f"a{SEP}1 = 3"


def test_if_variable_condition():
    result = ast.unparse(
        ssaify_string(
            f"""
            if b:
                a = 1
            """
        )
    )

    expected = textwrap.dedent(
        f"""
        if b:
            a{SEP}0 = 1
        """
    ).strip()

    assert result == expected


def test_nested_if():
    result = ast.unparse(
        ssaify_string(
            f"""
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
        f"""
        if True:
            a{SEP}0 = 1
            a{SEP}2 = a{SEP}0
        else:
            a{SEP}0 = 2
            if True:
                a{SEP}1 = 3
            else:
                a{SEP}1 = a{SEP}0
            a{SEP}2 = 4
        a{SEP}3 = 5
    """
    ).strip()

    assert result == expected


def test_nested_if_chain():
    result = ast.unparse(
        ssaify_string(
            f"""
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
        f"""
        if True:
            a{SEP}0 = 1
            a{SEP}2 = a{SEP}0
        else:
            a{SEP}0 = 2
            if True:
                a{SEP}1 = a{SEP}0 + 1
            else:
                a{SEP}1 = a{SEP}0
            a{SEP}2 = a{SEP}1 + 1
        a{SEP}3 = a{SEP}2 + 1
    """
    ).strip()

    assert result == expected


def test_if_branch_local():
    result = ast.unparse(
        ssaify_string(
            f"""
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
        f"""
        if True:
            a{SEP}0 = 0
            a{SEP}1 = a{SEP}0 + 1
        else:
            b{SEP}0 = 1
            b{SEP}1 = b{SEP}0 + 1
        """
    ).strip()

    assert result == expected


def test_if_only_one_branch():
    result = ast.unparse(
        ssaify_string(
            f"""
            if True:
                a = 0
            b = a
            """
        )
    )

    expected = textwrap.dedent(
        f"""
        if True:
            a{SEP}0 = 0
        b{SEP}0 = a
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
        f"""
        if True:
            a{SEP}0 = 0
        else:
            c{SEP}0 = 1
        b{SEP}0 = c
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
        f"""
        if True:
            if True:
                a{SEP}0 = 0
                a{SEP}1 = a{SEP}0 + 1
                a{SEP}2 = a{SEP}1 + 1
                a{SEP}3 = a{SEP}2 + 1
                a{SEP}4 = a{SEP}3 + 1
            else:
                a{SEP}0 = 0
                a{SEP}1 = a{SEP}0 + 1
                a{SEP}2 = a{SEP}1 + 1
                a{SEP}4 = a{SEP}2
        else:
            a{SEP}0 = 0
            a{SEP}4 = a{SEP}0
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
        f"""
        if True:
            a{SEP}0 = 0
            a{SEP}1 = a{SEP}0 + 1
            a{SEP}2 = a{SEP}1 + 1
            a{SEP}3 = a{SEP}2 + 1
            a{SEP}4 = a{SEP}3 + 1
        else:
            if True:
                a{SEP}0 = 0
                a{SEP}1 = a{SEP}0 + 1
                a{SEP}2 = a{SEP}1 + 1
            else:
                a{SEP}0 = 0
                a{SEP}2 = a{SEP}0
            a{SEP}4 = a{SEP}2
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
        f"""
        a{SEP}0 = 0
        if True:
            b{SEP}0 = 0
            a{SEP}1 = a{SEP}0
        elif True:
            if True:
                b{SEP}0 = 1
                a{SEP}1 = a{SEP}0
            elif True:
                b{SEP}0 = 2
                a{SEP}1 = a{SEP}0 + 1
            else:
                b{SEP}0 = 3
                a{SEP}1 = a{SEP}0
        else:
            b{SEP}0 = 4
            a{SEP}1 = a{SEP}0
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
        f"""
        def f(a):
            if True:
                a{SEP}0 = a + 1
            else:
                a{SEP}0 = a
            return a{SEP}0
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
        f"""
        def f():
            if True:
                a{SEP}0 = 1
            else:
                b{SEP}0 = 0
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
        f"""
        def f():
            if True:
                a{SEP}0 = 1
            else:
                a{SEP}0 = 0
            return (a{SEP}0, b)
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
        f"""
        def f():
            a{SEP}0 = 0
            if True:
                a{SEP}1 = 1
            else:
                a{SEP}1 = a{SEP}0
            return (a{SEP}1, b)
        """
    ).strip()

    assert result == expected


def test_broken_collisions():
    # Known bug of the current SSA implementation

    result = ast.unparse(
        ssaify_string(
            f"""
            a = a{SEP}0 + 1
            return a, a{SEP}0
            """
        )
    )

    expected = textwrap.dedent(
        f"""
        a{SEP}0 = a{SEP}0 + 1
        return (a{SEP}0, a{SEP}0)
        """
    ).strip()

    assert result == expected


def test_collision_function_parameters():
    # An earlier version couldn't handle this case correctly

    result = ast.unparse(
        ssaify_string(
            f"""
            def f(a, a{SEP}0):
                return a{SEP}0
            """
        )
    )

    expected = textwrap.dedent(
        f"""
        def f(a, a{SEP}0):
            return a{SEP}0
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
        f"""
        if True:
            a{SEP}0 = a + 1
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
        f"""
        a{SEP}0: int
        a{SEP}0 = a + 1
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
        f"""
        def f(a, b):
            if True:
                a{SEP}0 = a + 1
                d{SEP}0 = 5
                return a{SEP}0
            else:
                b{SEP}0 = b + 1
                e{SEP}0 = 6
            return (a, b{SEP}0, d, e{SEP}0)
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
        f"""
        def f(a, b):
            b{SEP}0 = a + 3
            if True:
                b{SEP}1 = b{SEP}0 + 1
                e{SEP}0 = 6
            else:
                a{SEP}0 = a + 1
                d{SEP}0 = 5
                return a{SEP}0
            return (a, b{SEP}1, d, e{SEP}0)
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
        f"""
        def f(a, b):
            if True:
                b{SEP}0 = b + 1
                e{SEP}0 = 6
                return e{SEP}0
            else:
                a{SEP}0 = a + 1
                d{SEP}0 = 5
                return a{SEP}0
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
        f"""
            def f(a, b):
                if True:
                    b{SEP}0 = b + 1
                    e{SEP}0 = 6
                    if e{SEP}0 == b{SEP}0:
                        return e{SEP}0
                    else:
                        return b{SEP}0
                else:
                    a{SEP}0 = a + 1
                    d{SEP}0 = 5
                return (a{SEP}0, b, d{SEP}0, e)
        """
    ).strip()

    assert result == expected

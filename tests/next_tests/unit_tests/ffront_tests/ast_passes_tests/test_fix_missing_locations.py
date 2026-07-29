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

from gt4py.next.ffront.ast_passes.fix_missing_locations import FixMissingLocations


def get_fixed_ast(code: str) -> ast.AST:
    python_ast = ast.parse(textwrap.dedent(code))
    python_ast.lineno = 1
    python_ast.end_lineno = 999
    python_ast.col_offset = 0
    python_ast.end_col_offset = 999
    return FixMissingLocations.apply(python_ast)


def test_reused_op_nodes():
    """
    Python's `ast.parse` will reuse the same nodes multiple times in the AST sometimes
    (e.g., for operators like `ast.Add`). In this case, all those shared nodes
    will share the same location information which can be plain wrong in some cases.
    """

    astree = get_fixed_ast(
        """
        a = (
            1
            /
            2
            /
            3
            /
            4
        )
        """
    )

    match astree:
        case ast.Module(
            body=[
                ast.Assign(
                    value=ast.BinOp(
                        left=ast.BinOp(
                            left=ast.BinOp(op=ast.Div() as first), op=ast.Div() as second
                        ),
                        op=ast.Div() as third,
                    )
                )
            ]
        ):
            pass
        case _:
            assert False

    assert first is not second
    assert second is not third
    assert third is not first


def test_sibling_nesting_bug():
    """
    Test for a bug where the location information sometimes wasn't taken from the
    parent node, but from a nested sibling node.
    """
    astree = get_fixed_ast(
        """
        a = (
            1 * 2
            + 3
        )
        """
    )

    assert (
        isinstance(astree.body[0].value.left.op, ast.Mult)
        and astree.body[0].value.left.op.end_lineno == 3
    )
    assert isinstance(astree.body[0].value.op, ast.Add) and astree.body[0].value.op.end_lineno == 4


def test_consistent_end_lineno_bug():
    """
    If a parent node has attributes `end_lineno` and `end_col_offset`, then
    its child nodes have those too.
    """
    astree = get_fixed_ast(
        """
        5 + 4
        """
    )

    assert hasattr(astree.body[0].value, "end_lineno")
    assert hasattr(astree.body[0].value, "end_col_offset")
    assert hasattr(astree.body[0].value.op, "end_lineno")
    assert hasattr(astree.body[0].value.op, "end_col_offset")

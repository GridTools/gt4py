# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import List

from gtc.common import ArithmeticOperator, ComparisonOperator, DataType, ExprKind, NativeFunction
from gtc.gtir import BinaryOp, Cast, Expr, NativeFuncCall, ParAssignStmt, TernaryOp
from gtc.passes.gtir_upcaster import _GTIRUpcasting

from .gtir_utils import FieldAccessFactory, LiteralFactory


class Placeholder(Expr):
    """Used as placeholder for an expression in comparison."""

    kind = ExprKind.SCALAR  # unused
    dtype = DataType.AUTO  # unused


A_BOOL_LITERAL = LiteralFactory(value="True", dtype=DataType.BOOL)
A_INT64_LITERAL = LiteralFactory(value="42", dtype=DataType.INT64)
A_FLOAT64_LITERAL = LiteralFactory(value="42.0", dtype=DataType.FLOAT64)
AN_UNIMPORTANT_LITERAL = LiteralFactory(value="", dtype=DataType.DEFAULT)


def contains_cast_node(cast_node, expr):
    # Checks if `expr` contains `cast_node`. If `cast_node` contains `expr=Placeholder()`
    # we skip equality check of `expr` and `cast_node.expr`
    return (
        len(
            expr.walk_values()
            .if_isinstance(Cast)
            .filter(
                lambda node: node.dtype == cast_node.dtype
                and (isinstance(cast_node.expr, Placeholder) or node.expr == cast_node.expr)
            )
            .to_list()
        )
        == 1
    )


def upcast_and_validate(expr, expected_cast_nodes):

    assert isinstance(expected_cast_nodes, List)
    assert all([isinstance(cast, Cast) for cast in expected_cast_nodes])
    assert all([not contains_cast_node(cast, expr) for cast in expected_cast_nodes])

    result: Expr = _GTIRUpcasting().visit(expr)

    for cast in expected_cast_nodes:
        assert contains_cast_node(cast, result)


def test_upcast_BinaryOp_BOOL_to_FLOAT():
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_FLOAT64_LITERAL)
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_BOOL_LITERAL)])


def test_upcast_BinaryOp_INT_to_FLOAT():
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL)
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_BinaryOp_nested_casting():
    outer_expr = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_INT64_LITERAL)
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=outer_expr, right=A_FLOAT64_LITERAL)
    expected = [
        Cast(dtype=DataType.INT64, expr=A_BOOL_LITERAL),
        Cast(dtype=DataType.FLOAT64, expr=Placeholder()),
    ]
    upcast_and_validate(testee, expected)


def test_upcast_NativeFuncCall():
    testee = NativeFuncCall(func=NativeFunction.MAX, args=[A_INT64_LITERAL, A_FLOAT64_LITERAL])
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_ParAssignStmt():
    testee = ParAssignStmt(left=FieldAccessFactory(dtype=DataType.FLOAT64), right=A_INT64_LITERAL)
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_TernaryOp():
    testee = TernaryOp(
        cond=A_BOOL_LITERAL,
        true_expr=A_INT64_LITERAL,
        false_expr=A_FLOAT64_LITERAL,
    )
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_in_cond_of_TernaryOp():
    testee = TernaryOp(
        cond=BinaryOp(op=ComparisonOperator.GE, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL),
        true_expr=AN_UNIMPORTANT_LITERAL,
        false_expr=AN_UNIMPORTANT_LITERAL,
    )
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_integers_division():
    testee = BinaryOp(
        op=ArithmeticOperator.DIV,
        left=LiteralFactory(value="1", dtype=DataType.INT32),
        right=LiteralFactory(value="2", dtype=DataType.INT32),
    )

    upcast_and_validate(
        testee,
        [
            Cast(dtype=DataType.FLOAT32, expr=LiteralFactory(value="1", dtype=DataType.INT32)),
            Cast(dtype=DataType.FLOAT32, expr=LiteralFactory(value="2", dtype=DataType.INT32)),
        ],
    )

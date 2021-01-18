# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import List, Optional, Union

import pytest
from pydantic import ValidationError

import eve
from gtc import common
from gtc.common import (
    ArithmeticOperator,
    ComparisonOperator,
    DataType,
    Expr,
    ExprKind,
    IfStmt,
    Literal,
    LogicalOperator,
    NativeFunction,
    Stmt,
    UnaryOperator,
)


INT_TYPE = DataType.INT32
FLOAT_TYPE = DataType.FLOAT32
ARITHMETIC_TYPE = FLOAT_TYPE
ANOTHER_ARITHMETIC_TYPE = INT_TYPE
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD
A_ARITHMETIC_UNARY_OPERATOR = UnaryOperator.POS
A_LOGICAL_UNARY_OPERATOR = UnaryOperator.NOT

# IR testing guidelines
# - For testing leave nodes: use the node directly
#   (the builder pattern would in general hide what's being tested)
# - For testing non-leave nodes, introduce builders with defaults (for leave nodes as well)


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class UnaryOp(Expr, common.UnaryOp[Expr]):
    pass


class BinaryOp(Expr, common.BinaryOp[Expr]):
    dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(Expr, common.TernaryOp[Expr]):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class BinaryOpUpcasting(Expr, common.BinaryOp[Expr]):
    dtype_propagation = common.binary_op_dtype_propagation(strict=False)


class TernaryOpUpcasting(Expr, common.TernaryOp[Expr]):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=False)


class NativeFuncCall(Expr, common.NativeFuncCall[Expr]):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class AssignStmt(Stmt, common.AssignStmt[DummyExpr, Expr]):
    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


@pytest.mark.parametrize(
    "node,expected",
    [
        (
            TernaryOp(
                cond=DummyExpr(dtype=DataType.BOOL),
                true_expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                false_expr=DummyExpr(dtype=ARITHMETIC_TYPE),
            ),
            ARITHMETIC_TYPE,
        ),
        (
            TernaryOpUpcasting(
                cond=DummyExpr(dtype=DataType.BOOL),
                true_expr=DummyExpr(dtype=FLOAT_TYPE),
                false_expr=DummyExpr(dtype=INT_TYPE),
            ),
            FLOAT_TYPE,
        ),
        (
            BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=ArithmeticOperator.ADD,
            ),
            ARITHMETIC_TYPE,
        ),
        (
            BinaryOpUpcasting(
                left=DummyExpr(dtype=INT_TYPE),
                right=DummyExpr(dtype=FLOAT_TYPE),
                op=ArithmeticOperator.ADD,
            ),
            FLOAT_TYPE,
        ),
        (
            BinaryOp(
                left=DummyExpr(dtype=DataType.BOOL),
                right=DummyExpr(dtype=DataType.BOOL),
                op=LogicalOperator.AND,
            ),
            DataType.BOOL,
        ),
        (
            BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=ComparisonOperator.EQ,
            ),
            DataType.BOOL,
        ),
        (
            UnaryOp(
                expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=A_ARITHMETIC_UNARY_OPERATOR,
            ),
            ARITHMETIC_TYPE,
        ),
    ],
)
def test_dtype_propagation(node, expected):
    assert node.dtype == expected


@pytest.mark.parametrize(
    "invalid_node,expected_regex",
    [
        (
            lambda: TernaryOp(
                cond=DummyExpr(dtype=ARITHMETIC_TYPE),
                true_expr=DummyExpr(),
                false_expr=DummyExpr(),
            ),
            r"Condition.*must be bool.*",
        ),
        (
            lambda: TernaryOp(
                cond=DummyExpr(dtype=DataType.BOOL),
                true_expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                false_expr=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
            ),
            r"Type mismatch",
        ),
        (
            lambda: IfStmt(cond=DummyExpr(dtype=ARITHMETIC_TYPE), true_branch=[], false_branch=[]),
            r"Condition.*must be bool.*",
        ),
        (
            lambda: Literal(value="foo"),
            r".*dtype\n.*field required",
        ),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
                op=A_ARITHMETIC_OPERATOR,
            ),
            r"Type mismatch",
        ),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=DataType.BOOL),
                right=DummyExpr(dtype=DataType.BOOL),
                op=A_ARITHMETIC_OPERATOR,
            ),
            r"Bool.* expr.* not allowed with arithmetic op.*",
        ),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=LogicalOperator.AND,
            ),
            r"Arithmetic expr.* not allowed in bool.* op.*",
        ),
        (
            lambda: UnaryOp(op=A_LOGICAL_UNARY_OPERATOR, expr=DummyExpr(dtype=ARITHMETIC_TYPE)),
            r"Unary op.*only .* with bool.*",
        ),
        (
            lambda: UnaryOp(op=A_ARITHMETIC_UNARY_OPERATOR, expr=DummyExpr(dtype=DataType.BOOL)),
            r"Unary op.* not allowed with bool.*",
        ),
        (
            lambda: NativeFuncCall(func=NativeFunction.SIN, args=[DummyExpr(), DummyExpr()]),
            r"accepts 1 arg.* 2.*passed",
        ),
        (
            lambda: AssignStmt(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
            ),
            r"Type mismatch",
        ),
    ],
)
def test_invalid_nodes(invalid_node, expected_regex):
    with pytest.raises(ValidationError, match=expected_regex):
        invalid_node()


class DummyNode(eve.Node):
    dtype: Optional[common.DataType]


class DtypeRootNode(eve.Node):
    field1: DummyNode
    field2: List[DummyNode]
    _validate_dtype_is_set = common.validate_dtype_is_set()


@pytest.mark.parametrize(
    "tree_with_missing_dtype",
    [
        lambda: DtypeRootNode(field1=DummyNode(), field2=[]),
        lambda: DtypeRootNode(field1=DummyNode(dtype=FLOAT_TYPE), field2=[DummyNode()]),
        lambda: DtypeRootNode(
            field1=DummyNode(dtype=FLOAT_TYPE), field2=[DummyNode(dtype=FLOAT_TYPE), DummyNode()]
        ),
    ],
)
def test_dtype_validator_for_invalid_tree(tree_with_missing_dtype):
    with pytest.raises(ValidationError, match=r"Nodes without dtype"):
        tree_with_missing_dtype()


def test_dtype_validator_for_valid_tree():
    DtypeRootNode(
        field1=DummyNode(dtype=FLOAT_TYPE),
        field2=[DummyNode(dtype=FLOAT_TYPE), DummyNode(dtype=FLOAT_TYPE)],
    )


class SymbolRefChildNode(eve.Node):
    name: eve.SymbolRef


class SymbolChildNode(eve.Node):
    name: eve.SymbolName
    clearly_a_symbol = ""  # prevent pydantic conversion


class AnotherSymbolTable(eve.Node, eve.SymbolTableTrait):
    nodes: List[Union[SymbolRefChildNode, SymbolChildNode]]


class SymbolTableRootNode(eve.Node, eve.SymbolTableTrait):
    nodes: List[Union[SymbolRefChildNode, SymbolChildNode, AnotherSymbolTable]]

    _validate_symbol_refs = common.validate_symbol_refs()


@pytest.mark.parametrize(
    "tree_with_missing_symbol",
    [
        lambda: SymbolTableRootNode(nodes=[SymbolRefChildNode(name="foo")]),
        lambda: SymbolTableRootNode(
            nodes=[
                SymbolChildNode(name="foo"),
                SymbolRefChildNode(name="foo"),
                SymbolRefChildNode(name="foo2"),
            ],
        ),
        lambda: SymbolTableRootNode(
            nodes=[
                AnotherSymbolTable(nodes=[SymbolChildNode(name="inner_scope")]),
                SymbolRefChildNode(name="inner_scope"),
            ]
        ),
    ],
)
def test_symbolref_validation_for_invalid_tree(tree_with_missing_symbol):
    with pytest.raises(ValidationError, match=r"Symbols.*not found"):
        tree_with_missing_symbol()


def test_symbolref_validation_for_valid_tree():
    SymbolTableRootNode(
        nodes=[SymbolChildNode(name="foo"), SymbolRefChildNode(name="foo")],
    )
    SymbolTableRootNode(
        nodes=[
            SymbolChildNode(name="foo"),
            SymbolRefChildNode(name="foo"),
            SymbolRefChildNode(name="foo"),
        ],
    ),
    SymbolTableRootNode(
        nodes=[
            SymbolChildNode(name="outer_scope"),
            AnotherSymbolTable(nodes=[SymbolRefChildNode(name="outer_scope")]),
        ]
    )
    SymbolTableRootNode(
        nodes=[
            AnotherSymbolTable(
                nodes=[SymbolChildNode(name="inner_scope"), SymbolRefChildNode(name="inner_scope")]
            )
        ]
    )


# For pydantic, nodes are the same (convertible to each other) if all fields are same.
# For checking, we need to make the Expr categories clearly different.
# This behavior will most likely change in Eve in the future
class ExprA(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_a = ""


class ExprB(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_b = ""


class StmtA(Stmt):
    clearly_stmt_a = ""


class StmtB(Stmt):
    clearly_stmt_b = ""


def test_AssignSmt_category():
    Testee = common.AssignStmt[ExprA, ExprA]

    Testee(left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(left=ExprB(), right=ExprA())
        Testee(left=ExprA(), right=ExprB())


def test_IfStmt_category():
    Testee = common.IfStmt[StmtA, ExprA]

    Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())
    with pytest.raises(ValidationError):
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtB(), false_branch=StmtA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtB())
        Testee(cond=ExprB(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())


def test_UnaryOp_category():
    class Testee(ExprA, common.UnaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprB())


def test_BinaryOp_category():
    class Testee(ExprA, common.BinaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprB(), right=ExprA())
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprB())


def test_TernaryOp_category():
    class Testee(ExprA, common.TernaryOp[ExprA]):
        pass

    Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(cond=ExprB(dtype=DataType.BOOL), true_expr=ExprB(), false_expr=ExprA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprB())


def test_Cast_category():
    class Testee(ExprA, common.Cast[ExprA]):
        pass

    Testee(dtype=ARITHMETIC_TYPE, expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(dtype=ARITHMETIC_TYPE, expr=ExprB())


def test_NativeFuncCall_category():
    class Testee(ExprA, common.NativeFuncCall[ExprA]):
        pass

    Testee(func=NativeFunction.SIN, args=[ExprA()])
    with pytest.raises(ValidationError):
        Testee(func=NativeFunction.SIN, args=[ExprB()])

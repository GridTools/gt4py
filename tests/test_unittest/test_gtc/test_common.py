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

from typing import List, Optional, Tuple, Union

import pytest

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
    kind: ExprKind = ExprKind.FIELD


class BinaryOp(Expr, common.BinaryOp[Expr]):
    dtype_propagation = common.binary_op_dtype_propagation(strict=True)
    kind: ExprKind = ExprKind.FIELD


class TernaryOp(Expr, common.TernaryOp[Expr]):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)
    kind: ExprKind = ExprKind.FIELD


class BinaryOpUpcasting(Expr, common.BinaryOp[Expr]):
    dtype_propagation = common.binary_op_dtype_propagation(strict=False)
    kind: ExprKind = ExprKind.FIELD


class TernaryOpUpcasting(Expr, common.TernaryOp[Expr]):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=False)
    kind: ExprKind = ExprKind.FIELD


class NativeFuncCall(Expr, common.NativeFuncCall[Expr]):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)
    kind: ExprKind = ExprKind.FIELD


class AssignStmt(Stmt, common.AssignStmt[DummyExpr, Expr]):
    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)
    kind: ExprKind = ExprKind.FIELD


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
    "invalid_node,expected_regex,error",
    [
        (
            lambda: TernaryOp(
                cond=DummyExpr(dtype=ARITHMETIC_TYPE),
                true_expr=DummyExpr(),
                false_expr=DummyExpr(),
            ),
            r"Condition.*must be bool.*",
            ValueError,
        ),
        (
            lambda: TernaryOp(
                cond=DummyExpr(dtype=DataType.BOOL),
                true_expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                false_expr=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
            ),
            r"Type mismatch",
            ValueError,
        ),
        (
            lambda: IfStmt(cond=DummyExpr(dtype=ARITHMETIC_TYPE), true_branch=[], false_branch=[]),
            r"Condition.*must be bool.*",
            ValueError,
        ),
        (lambda: Literal(value="foo"), r"required keyword-only argument: 'dtype'", TypeError),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
                op=A_ARITHMETIC_OPERATOR,
            ),
            r"Type mismatch",
            ValueError,
        ),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=DataType.BOOL),
                right=DummyExpr(dtype=DataType.BOOL),
                op=A_ARITHMETIC_OPERATOR,
            ),
            r"Bool.* expr.* not allowed with arithmetic op.*",
            ValueError,
        ),
        (
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=LogicalOperator.AND,
            ),
            r"Arithmetic expr.* not allowed in bool.* op.*",
            ValueError,
        ),
        (
            lambda: UnaryOp(op=A_LOGICAL_UNARY_OPERATOR, expr=DummyExpr(dtype=ARITHMETIC_TYPE)),
            r"Unary op.*only .* with bool.*",
            ValueError,
        ),
        (
            lambda: UnaryOp(op=A_ARITHMETIC_UNARY_OPERATOR, expr=DummyExpr(dtype=DataType.BOOL)),
            r"Unary op.* not allowed with bool.*",
            ValueError,
        ),
        (
            lambda: NativeFuncCall(func=NativeFunction.SIN, args=[DummyExpr(), DummyExpr()]),
            r"accepts 1 arg.* 2.*passed",
            ValueError,
        ),
        (
            lambda: AssignStmt(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
            ),
            r"Type mismatch",
            ValueError,
        ),
    ],
)
def test_invalid_nodes(invalid_node, expected_regex, error):
    with pytest.raises(error, match=expected_regex):
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
    with pytest.raises(TypeError, match=r"required keyword-only argument: 'dtype'"):
        tree_with_missing_dtype()


def test_dtype_validator_for_valid_tree():
    DtypeRootNode(
        field1=DummyNode(dtype=FLOAT_TYPE),
        field2=[DummyNode(dtype=FLOAT_TYPE), DummyNode(dtype=FLOAT_TYPE)],
    )


class SymbolRefChildNode(eve.Node):
    name: eve.Coerced[eve.SymbolRef]


class SymbolChildNode(eve.Node):
    name: eve.Coerced[eve.SymbolName]


class AnotherSymbolTable(eve.Node, eve.SymbolTableTrait):
    nodes: List[Union[SymbolRefChildNode, SymbolChildNode]]


class SymbolTableRootNode(eve.Node, eve.ValidatedSymbolTableTrait):
    nodes: List[Union[SymbolRefChildNode, SymbolChildNode, AnotherSymbolTable]]


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
    with pytest.raises(ValueError, match=r"Symbols.*not found"):
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


class MultiDimDecl(SymbolChildNode):
    dimensions: Tuple[bool, bool, bool] = (True, True, True)


class MultiDimRef(DummyExpr, SymbolRefChildNode):
    kind = ExprKind.FIELD


class MultiDimLoop(eve.Node):
    loop_order: common.LoopOrder
    assigns: List[AssignStmt]


class MultiDimRoot(eve.Node, eve.SymbolTableTrait):
    decls: List[MultiDimDecl]
    loops: List[MultiDimLoop]

    _lvalue_dims_validator = common.validate_lvalue_dims(MultiDimLoop, MultiDimDecl)


def construct_dims_assignment(dimensions: Tuple[bool, bool, bool], direction: common.LoopOrder):
    in_name = "in"
    out_name = "out"
    MultiDimRoot(
        decls=[
            MultiDimDecl(name=out_name, dimensions=dimensions),
            MultiDimDecl(name=in_name, dimensions=dimensions),
        ],
        loops=[
            MultiDimLoop(
                loop_order=direction,
                assigns=[
                    AssignStmt(left=MultiDimRef(name=out_name), right=MultiDimRef(name=in_name)),
                ],
            ),
        ],
    )


def test_lvalue_dims_validation():
    # assigning to ik in forward direction not allowed
    with pytest.raises(ValueError, match=r"Not allowed to assign to ik-field"):
        construct_dims_assignment(
            dimensions=(True, False, True), direction=common.LoopOrder.FORWARD
        )

    # assigning to ij in forward direction ok
    construct_dims_assignment(dimensions=(True, True, False), direction=common.LoopOrder.FORWARD)

    # assigning to ij in parallel direction not allowed
    with pytest.raises(ValueError, match=r"Not allowed to assign to ij-field `out` in PARALLEL"):
        construct_dims_assignment(
            dimensions=(True, True, False), direction=common.LoopOrder.PARALLEL
        )


class ExprA(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_a = ""


class ExprB(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_b = ""


class ExprC(Expr):
    dtype: DataType = DataType.INT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_c = ""


class StmtA(Stmt):
    clearly_stmt_a = ""


class StmtB(Stmt):
    clearly_stmt_b = ""


def test_AssignSmt_category():
    Testee = common.AssignStmt[ExprA, ExprA]

    Testee(left=ExprA(), right=ExprA())
    with pytest.raises(TypeError):
        Testee(left=ExprB(), right=ExprA())
        Testee(left=ExprA(), right=ExprB())


def test_IfStmt_category():
    Testee = common.IfStmt[StmtA, ExprA]

    Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())
    with pytest.raises(TypeError):
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtB(), false_branch=StmtA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtB())
        Testee(cond=ExprB(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())


def test_UnaryOp_category():
    class Testee(ExprA, common.UnaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprA())
    with pytest.raises(TypeError):
        Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprB())


def test_BinaryOp_category():
    class Testee(ExprA, common.BinaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprA())
    with pytest.raises(TypeError):
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprB(), right=ExprA())
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprB())


def test_TernaryOp_category():
    class Testee(ExprA, common.TernaryOp[ExprA]):
        pass

    Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprA())
    with pytest.raises(TypeError):
        Testee(cond=ExprB(dtype=DataType.BOOL), true_expr=ExprB(), false_expr=ExprA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprB())


def test_Cast_category():
    class Testee(ExprA, common.Cast[ExprA]):
        pass

    Testee(dtype=ARITHMETIC_TYPE, expr=ExprA())
    with pytest.raises(TypeError):
        Testee(dtype=ARITHMETIC_TYPE, expr=ExprB())


def test_NativeFuncCall_category():
    class Testee(ExprA, common.NativeFuncCall[ExprA]):
        pass

    Testee(func=NativeFunction.SIN, args=[ExprA()])
    with pytest.raises(TypeError):
        Testee(func=NativeFunction.SIN, args=[ExprB()])


def test_VariableKOffset_category():
    class Testee(common.VariableKOffset[ExprC]):
        pass

    Testee(k=ExprC())
    with pytest.raises(TypeError):
        Testee(k=ExprA())


def test_HorizontalInterval():
    common.HorizontalInterval(
        start=common.AxisBound(level=common.LevelMarker.START, offset=-1),
        end=common.AxisBound(level=common.LevelMarker.START, offset=0),
    )
    with pytest.raises(ValueError):
        common.HorizontalInterval(
            start=common.AxisBound(level=common.LevelMarker.END, offset=0),
            end=common.AxisBound(level=common.LevelMarker.START, offset=-1),
        )
        common.HorizontalInterval(
            start=common.AxisBound(level=common.LevelMarker.START, offset=0),
            end=common.AxisBound(level=common.LevelMarker.START, offset=-1),
        )

from typing import List

import pytest

from gt4py.gtc.common import (
    ArithmeticOperator,
    ComparisonOperator,
    DataType,
    ExprKind,
    NativeFunction,
)
from gt4py.gtc.gtir import BinaryOp, Cast, Expr, NativeFuncCall, ParAssignStmt, TernaryOp
from gt4py.gtc.passes.gtir_upcaster import _GTIRUpcasting

from .gtir_utils import FieldAccessBuilder, make_Literal


class Placeholder(Expr):
    """Used as placeholder for an expression in comparison"""

    kind = ExprKind.SCALAR  # unused
    dtype = DataType.AUTO  # unused


A_BOOL_LITERAL = make_Literal("", dtype=DataType.BOOL)
A_INT64_LITERAL = make_Literal("", dtype=DataType.INT64)
A_FLOAT64_LITERAL = make_Literal("", dtype=DataType.FLOAT64)
AN_UNIMPORTANT_LITERAL = make_Literal("", dtype=DataType.DEFAULT)


def upcast_BinaryOp_BOOL_to_FLOAT():
    test_input = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_FLOAT64_LITERAL)
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_BOOL_LITERAL)]
    return test_input, expected_result


def upcast_BinaryOp_INT_to_FLOAT():
    test_input = BinaryOp(op=ArithmeticOperator.ADD, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL)
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)]
    return test_input, expected_result


def upcast_BinaryOp_nested_casting():
    outer_expr = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_INT64_LITERAL)
    test_input = BinaryOp(op=ArithmeticOperator.ADD, left=outer_expr, right=A_FLOAT64_LITERAL)
    expected_result = [
        Cast(dtype=DataType.INT64, expr=A_BOOL_LITERAL),
        Cast(dtype=DataType.FLOAT64, expr=Placeholder()),
    ]
    return test_input, expected_result


def upcast_NativeFuncCall():
    test_input = NativeFuncCall(func=NativeFunction.MAX, args=[A_INT64_LITERAL, A_FLOAT64_LITERAL])
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)]
    return test_input, expected_result


def upcast_ParAssignStmt():
    test_input = ParAssignStmt(
        left=FieldAccessBuilder("out").dtype(DataType.FLOAT64).build(), right=A_INT64_LITERAL
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)]
    return test_input, expected_result


def upcast_TernaryOp():
    test_input = TernaryOp(
        cond=A_BOOL_LITERAL,
        true_expr=A_INT64_LITERAL,
        false_expr=A_FLOAT64_LITERAL,
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)]
    return test_input, expected_result


def upcast_in_cond_of_TernaryOp():
    test_input = TernaryOp(
        cond=BinaryOp(op=ComparisonOperator.GE, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL),
        true_expr=AN_UNIMPORTANT_LITERAL,
        false_expr=AN_UNIMPORTANT_LITERAL,
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)]
    return test_input, expected_result


@pytest.fixture(
    params=[
        upcast_BinaryOp_BOOL_to_FLOAT,
        upcast_BinaryOp_INT_to_FLOAT,
        upcast_BinaryOp_nested_casting,
        upcast_NativeFuncCall,
        upcast_ParAssignStmt,
        upcast_TernaryOp,
        upcast_in_cond_of_TernaryOp,
    ]
)
def input_and_expected(request):
    return request.param()


def contains_cast_node(cast_node, expr):
    # Checks if `expr` contains `cast_node`. If `cast_node` contains `expr=Placeholder()`
    # we skip equality check of `expr` and `cast_node.expr`
    return (
        len(
            expr.iter_tree()
            .if_isinstance(Cast)
            .filter(
                lambda node: node.dtype == cast_node.dtype
                and (isinstance(cast_node.expr, Placeholder) or node.expr == cast_node.expr)
            )
            .to_list()
        )
        == 1
    )


def test_upcasted_nodes(input_and_expected):
    expr, expected_cast_nodes = input_and_expected

    assert isinstance(expected_cast_nodes, List)
    assert all([isinstance(cast, Cast) for cast in expected_cast_nodes])
    assert all([not contains_cast_node(cast, expr) for cast in expected_cast_nodes])

    result: Expr = _GTIRUpcasting().visit(expr)

    for cast in expected_cast_nodes:
        assert contains_cast_node(cast, result)

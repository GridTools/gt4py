from typing import List

from gt4py.gtc.common import ArithmeticOperator, DataType
from gt4py.gtc.gtir import Expr, Cast, BinaryOp
from gt4py.gtc.passes.gtir_upcaster import _GTIRUpcasting
import pytest

from .gtir_utils import make_Literal
from devtools import debug


def upcast_INT_to_FLOAT():
    expr_to_upcast = make_Literal("", dtype=DataType.INT64)
    test_input = BinaryOp(
        op=ArithmeticOperator.ADD,
        left=expr_to_upcast,
        right=make_Literal("", dtype=DataType.FLOAT64),
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=expr_to_upcast)]
    return test_input, expected_result


def upcast_BOOL_to_FLOAT():
    expr_to_upcast = make_Literal("", dtype=DataType.BOOL)
    test_input = BinaryOp(
        op=ArithmeticOperator.ADD,
        left=expr_to_upcast,
        right=make_Literal("", dtype=DataType.FLOAT64),
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=expr_to_upcast)]
    return test_input, expected_result


def upcast_INT_TO_FLOAT():
    expr_to_upcast = make_Literal("", dtype=DataType.INT64)
    test_input = BinaryOp(
        op=ArithmeticOperator.ADD,
        left=expr_to_upcast,
        right=make_Literal("", dtype=DataType.FLOAT64),
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=expr_to_upcast)]
    return test_input, expected_result


def upcast_nested_BinaryOp():
    expr_to_upcast = BinaryOp(
        op=ArithmeticOperator.ADD,
        left=make_Literal("", dtype=DataType.FLOAT32),
        right=make_Literal("", dtype=DataType.FLOAT32),
    )
    test_input = BinaryOp(
        op=ArithmeticOperator.ADD,
        left=expr_to_upcast,
        right=make_Literal("", dtype=DataType.FLOAT64),
    )
    expected_result = [Cast(dtype=DataType.FLOAT64, expr=expr_to_upcast)]
    return test_input, expected_result


# TODO tests for all node types and nesting with 2 casts


@pytest.fixture(params=[upcast_INT_to_FLOAT, upcast_BOOL_to_FLOAT, upcast_nested_BinaryOp])
def input_and_expected(request):
    return request.param()


def contains_cast_node(cast_node, expr):
    return (
        len(
            expr.iter_tree()
            .if_isinstance(Cast)
            .filter(lambda node: node.dtype == cast_node.dtype and node.expr == cast_node.expr)
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

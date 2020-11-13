# TODO tests

import ast

import pytest

from gt4py.gtc.common import ArithmeticOperator, LoopOrder
from gt4py.gtc.gtir import (
    AxisBound,
    BinaryOp,
    CartesianOffset,
    FieldAccess,
    VerticalInterval,
    ParAssignStmt,
    VerticalLoop,
)
from gt4py.gtc.python.python_naive_codegen import PythonNaiveCodegen


def ast_parse(arg):
    print(arg)
    return ast.parse(arg)


GTIROP_TO_ASTOP = {
    ArithmeticOperator.ADD: ast.Add,
    ArithmeticOperator.SUB: ast.Sub,
    ArithmeticOperator.MUL: ast.Mult,
    ArithmeticOperator.DIV: ast.Div,
}


@pytest.fixture
def naive_codegen():
    yield PythonNaiveCodegen()


@pytest.fixture(params=ArithmeticOperator.__members__.values())
def binary_operator(request):
    yield request.param


@pytest.fixture(params=LoopOrder.__members__.values())
def loop_order(request):
    yield request.param


@pytest.fixture(
    params=[
        FieldAccess.centered(name="a"),
        FieldAccess(name="b", offset=CartesianOffset(i=-1, j=3, k=-2)),
    ]
)
def field_access(request):
    yield request.param


def test_field_access(naive_codegen, field_access):
    source_tree = ast_parse(naive_codegen.apply(field_access))
    toplevel = source_tree.body[0]
    assert isinstance(toplevel, ast.Expr)
    assert isinstance(toplevel.value, ast.Subscript)


def test_assign(naive_codegen):
    assign = ParAssignStmt(
        left=FieldAccess.centered(name="a"), right=FieldAccess.centered(name="b")
    )
    ast_parse(naive_codegen.apply(assign))


def test_binary_op(naive_codegen, binary_operator):
    bin_op = BinaryOp(
        op=binary_operator,
        left=FieldAccess.centered(name="a"),
        right=FieldAccess.centered(name="b"),
    )
    source_tree = ast_parse(naive_codegen.apply(bin_op))
    toplevel = source_tree.body[0]
    assert isinstance(source_tree.body[0], ast.Expr)
    assert isinstance(toplevel.value, ast.BinOp)
    assert isinstance(toplevel.value.op, GTIROP_TO_ASTOP[binary_operator])


def test_vertical_interval(naive_codegen):
    vertical_interval = VerticalInterval(
        start=AxisBound.start(),
        end=AxisBound.end(),
        body=[
            ParAssignStmt(
                left=FieldAccess.centered(name="a"), right=FieldAccess.centered(name="b")
            ),
            ParAssignStmt(
                left=FieldAccess.centered(name="b"), right=FieldAccess.centered(name="c")
            ),
        ],
    )
    print(naive_codegen.apply(vertical_interval))
    source_tree = ast_parse(naive_codegen.apply(vertical_interval))
    for_k = source_tree.body[0]
    assert isinstance(for_k, ast.For)
    assert for_k.target.id == "K"
    assert isinstance(for_k.iter, ast.Call)
    assert isinstance(for_k.iter.args[-1], ast.Subscript)
    for_i = for_k.body[0]
    assert isinstance(for_i, ast.For)
    assert for_i.target.id == "I"
    for_i = for_k.body[1]
    assert isinstance(for_i, ast.For)
    assert for_i.target.id == "I"


def test_vertical_loop(naive_codegen, loop_order):
    vertical_loop = VerticalLoop(
        loop_order=loop_order,
        vertical_intervals=[
            VerticalInterval(
                start=AxisBound.start(),
                end=AxisBound.from_end(2),
                body=[
                    ParAssignStmt(
                        left=FieldAccess.centered(name="a"),
                        right=FieldAccess.centered(name="b"),
                    )
                ],
            ),
            VerticalInterval(
                start=AxisBound.from_end(1),
                end=AxisBound.end(),
                body=[
                    ParAssignStmt(
                        left=FieldAccess.centered(name="a"),
                        right=FieldAccess.centered(name="b"),
                    )
                ],
            ),
        ],
    )
    source_tree = ast_parse(naive_codegen.apply(vertical_loop))

    for_k = source_tree.body[0]
    assert isinstance(for_k, ast.For)
    assert for_k.target.id == "K"
    assert isinstance(for_k.iter, ast.Call)
    range_end = for_k.iter.args[-1]
    assert isinstance(range_end, ast.BinOp)
    assert isinstance(range_end.op, ast.Sub)
    assert range_end.right.n == 2

    for_k = source_tree.body[1]
    assert isinstance(for_k, ast.For)
    assert for_k.target.id == "K"
    assert isinstance(for_k.iter, ast.Call)
    range_start = for_k.iter.args[0]
    range_end = for_k.iter.args[-1]
    assert isinstance(range_start, ast.BinOp)
    assert isinstance(range_end, ast.Subscript)
    assert isinstance(range_start.op, ast.Sub)
    assert range_start.right.n == 1

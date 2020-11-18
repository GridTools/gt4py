import ast

from pydantic.error_wrappers import ValidationError

import pytest
from devtools import debug
from eve import SourceLocation
from .gtir_utils import FieldAccessBuilder, DummyExpr

from gt4py.gtc.common import (
    ArithmeticOperator,
    DataType,
    LevelMarker,
    LoopOrder,
)
from gt4py.gtc.gtir import (
    AxisBound,
    BinaryOp,
    CartesianOffset,
    Decl,
    Stencil,
    FieldAccess,
    FieldDecl,
    Stmt,
    Interval,
    VerticalLoop,
    ParAssignStmt,
    Expr,
)
from gt4py.gtc.python.python_naive_codegen import PythonNaiveCodegen

ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD


@pytest.fixture
def copy_assign():
    yield ParAssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="a", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="b", loc=SourceLocation(line=3, column=3, source="copy_gtir")
        ),
    )


@pytest.fixture
def interval(copy_assign):
    yield Interval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
    )


@pytest.fixture
def copy_v_loop(copy_assign, interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        interval=interval,
        body=[copy_assign],
    )


@pytest.fixture
def copy_computation(copy_v_loop):
    yield Stencil(
        name="copy_gtir",
        loc=SourceLocation(line=1, column=1, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[copy_v_loop],
    )


def test_copy(copy_computation):
    print(debug(copy_computation))
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


def test_naive_python_copy(copy_computation):
    assert ast.parse(PythonNaiveCodegen.apply(copy_computation))


def test_naive_python_avg():
    horizontal_avg = Stencil(
        name="horizontal_avg",
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[
            VerticalLoop(
                loop_order=LoopOrder.FORWARD,
                interval=Interval(
                    start=AxisBound(level=LevelMarker.START, offset=0),
                    end=AxisBound(level=LevelMarker.END, offset=0),
                ),
                body=[
                    ParAssignStmt(
                        left=FieldAccess.centered(name="a"),
                        right=BinaryOp(
                            left=FieldAccess(
                                name="b",
                                offset=CartesianOffset(i=-1, j=0, k=0),
                            ),
                            right=FieldAccess(name="b", offset=CartesianOffset(i=1, j=0, k=0)),
                            op=ArithmeticOperator.ADD,
                        ),
                    )
                ],
            )
        ],
    )
    assert ast.parse(PythonNaiveCodegen.apply(horizontal_avg))


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


@pytest.mark.parametrize(
    "valid_node",
    [
        pytest.param(
            lambda: ParAssignStmt(
                left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1)).build(),
                right=DummyExpr(),
            ),
            id="vertical offset is allowed in l.h.s. of assignment",
        )
    ],
)
def test_valid_nodes(valid_node):
    valid_node()


@pytest.mark.parametrize(
    "invalid_node",
    [
        pytest.param(
            lambda: ParAssignStmt(
                left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0)).build(),
                right=DummyExpr(),
            ),
            id="non-zero horizontal offset not allowed",
        ),
    ],
)
def test_invalid_nodes(invalid_node):
    with pytest.raises(ValidationError):
        invalid_node()

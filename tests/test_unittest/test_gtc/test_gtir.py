import ast
import pydantic
from pydantic.error_wrappers import ValidationError

import pytest
from devtools import debug
from eve import SourceLocation

from gt4py.gtc.common import (
    BinaryOperator,
    ComparisonOperator,
    DataType,
    LevelMarker,
    LogicalOperator,
    LoopOrder,
)
from gt4py.gtc.gtir import (
    AxisBound,
    BinaryOp,
    CartesianOffset,
    Stencil,
    FieldAccess,
    FieldDecl,
    IfStmt,
    Stmt,
    Literal,
    TernaryOp,
    VerticalInterval,
    VerticalLoop,
    ParAssignStmt,
    Expr,
)
from gt4py.gtc.python.python_naive_codegen import PythonNaiveCodegen


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
def copy_interval(copy_assign):
    yield VerticalInterval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
        body=[copy_assign],
    )


@pytest.fixture
def copy_v_loop(copy_interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        vertical_intervals=[copy_interval],
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
                vertical_intervals=[
                    VerticalInterval(
                        start=AxisBound(level=LevelMarker.START, offset=0),
                        end=AxisBound(level=LevelMarker.END, offset=0),
                        body=[
                            ParAssignStmt(
                                left=FieldAccess.centered(name="a"),
                                right=BinaryOp(
                                    left=FieldAccess(
                                        name="b",
                                        offset=CartesianOffset(i=-1, j=0, k=0),
                                    ),
                                    right=FieldAccess(
                                        name="b", offset=CartesianOffset(i=1, j=0, k=0)
                                    ),
                                    op=BinaryOperator.ADD,
                                ),
                            )
                        ],
                    )
                ],
            )
        ],
    )
    assert ast.parse(PythonNaiveCodegen.apply(horizontal_avg))


def test_ExprBaseclassIsNotInstantiatable():
    with pytest.raises(TypeError):
        Expr()


def test_StmtBaseclassIsNotInstantiatable():
    with pytest.raises(TypeError):
        Stmt()


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""


# Validation tests
def test_ParAssignStmtWithVerticalOffsetIsOk():
    ParAssignStmt(
        left=FieldAccess(name="foo", offset=CartesianOffset(i=0, j=0, k=1)), right=DummyExpr()
    )


def test_ParAssignStmtWithHorizontalOffsetIsError():
    with pytest.raises(ValidationError):
        ParAssignStmt(
            left=FieldAccess(name="foo", offset=CartesianOffset(i=1, j=0, k=0)), right=DummyExpr()
        )


def test_TernaryOpValidation():
    assert (
        TernaryOp(
            cond=DummyExpr(dtype=DataType.BOOL),
            true_expr=DummyExpr(dtype=DataType.INT32),
            false_expr=DummyExpr(dtype=DataType.INT32),
        ).dtype
        == DataType.INT32
    )

    with pytest.raises(ValidationError):
        TernaryOp(
            cond=DummyExpr(dtype=DataType.BOOL),
            true_expr=DummyExpr(dtype=DataType.BOOL),
            false_expr=DummyExpr(dtype=DataType.INT32),
        )

        TernaryOp(
            cond=DummyExpr(dtype=DataType.INT32),
            true_expr=DummyExpr(dtype=DataType.INT32),
            false_expr=DummyExpr(dtype=DataType.INT32),
        )


def test_NonBooleanIfStmtConditionIsError():
    with pytest.raises(ValidationError):
        IfStmt(cond=DummyExpr(dtype=DataType.INT32), true_branch=[], false_branch=[])


def test_LiteralRequiresDtype():
    with pytest.raises(ValidationError):
        Literal(value="foo")


def test_BinaryOpErrorsForIncompatibleTypes():
    BinaryOp(
        left=DummyExpr(dtype=DataType.INT32),
        right=DummyExpr(dtype=DataType.INT32),
        op=BinaryOperator.ADD,
    )
    with pytest.raises(ValidationError):
        BinaryOp(
            left=DummyExpr(dtype=DataType.INT32),
            right=DummyExpr(dtype=DataType.INT16),
            op=BinaryOperator.ADD,
        )


def test_BinaryOpErrorsForArithmeticOperationOnBooleanExpr():
    with pytest.raises(ValidationError):
        BinaryOp(
            left=DummyExpr(dtype=DataType.BOOL),
            right=DummyExpr(dtype=DataType.BOOL),
            op=BinaryOperator.ADD,
        )


def test_BinaryOpComparison():
    comparison = BinaryOp(
        left=DummyExpr(dtype=DataType.INT32),
        right=DummyExpr(dtype=DataType.INT32),
        op=ComparisonOperator.EQ,
    )
    assert comparison.dtype == DataType.BOOL


def test_BinaryOpLogical():
    assert (
        BinaryOp(
            left=DummyExpr(dtype=DataType.BOOL),
            right=DummyExpr(dtype=DataType.BOOL),
            op=LogicalOperator.AND,
        ).dtype
        == DataType.BOOL
    )
    with pytest.raises(ValueError):
        BinaryOp(
            left=DummyExpr(dtype=DataType.INT32),
            right=DummyExpr(dtype=DataType.INT32),
            op=LogicalOperator.AND,
        )

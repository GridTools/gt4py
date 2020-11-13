import ast

from pydantic.error_wrappers import ValidationError

import pytest
from devtools import debug
from eve import SourceLocation
from .gtir_utils import FieldAccessBuilder, DummyExpr

from gt4py.gtc.common import (
    ArithmeticOperator,
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
    Decl,
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

# IR testing guidelines
# - For testing leave nodes: use the node directly
#   (a builder/maker pattern would in general hide what's being tested)
# - For testing complex nodes, introduce buildes with defaults also for leave nodes


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
                                    op=ArithmeticOperator.ADD,
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


def test_DeclBaseclassIsNotInstantiatable():
    with pytest.raises(TypeError):
        Decl()


# Validation tests
def test_ParAssignStmtWithVerticalOffsetIsOk():
    ParAssignStmt(
        left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1))(), right=DummyExpr()
    )


def test_ParAssignStmtWithHorizontalOffsetIsError():
    with pytest.raises(ValidationError):
        ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0))(),
            right=DummyExpr(),
        )


arithmetic_type = DataType.FLOAT32
another_arithmetic_type = DataType.INT32


def test_TernaryOpValidNode():
    assert (
        TernaryOp(
            cond=DummyExpr(dtype=DataType.BOOL),
            true_expr=DummyExpr(dtype=arithmetic_type),
            false_expr=DummyExpr(dtype=arithmetic_type),
        ).dtype
        == arithmetic_type
    )


def ternaryOpExprTypesMismatch():
    return TernaryOp(
        cond=DummyExpr(dtype=DataType.BOOL),
        true_expr=DummyExpr(dtype=arithmetic_type),
        false_expr=DummyExpr(dtype=another_arithmetic_type),
    )


def ternaryOpConditionIsNotBool():
    return TernaryOp(
        cond=DummyExpr(dtype=arithmetic_type),
        true_expr=DummyExpr(),
        false_expr=DummyExpr(),
    )


@pytest.fixture(params=[ternaryOpConditionIsNotBool, ternaryOpExprTypesMismatch])
def invalidNodes(request):
    yield request.param


def test_NodeValidation(invalidNodes):
    with pytest.raises(ValidationError):
        invalidNodes()


def test_IfStmtConditionIsNotBool():
    with pytest.raises(ValidationError):
        IfStmt(cond=DummyExpr(dtype=arithmetic_type), true_branch=[], false_branch=[])


def test_LiteralRequiresDtype():
    with pytest.raises(ValidationError):
        Literal(value="foo")


def test_BinaryOpTypePropagation():
    assert (
        BinaryOp(
            left=DummyExpr(dtype=arithmetic_type),
            right=DummyExpr(dtype=arithmetic_type),
            op=ArithmeticOperator.ADD,
        ).dtype
        == arithmetic_type
    )


a_binary_operator = ArithmeticOperator.ADD


def test_BinaryOpExprTypesMismatch():
    with pytest.raises(ValidationError):
        BinaryOp(
            left=DummyExpr(dtype=arithmetic_type),
            right=DummyExpr(dtype=another_arithmetic_type),
            op=a_binary_operator,
        )


def test_BinaryOpErrorsForArithmeticOperationOnBooleanExpr():
    with pytest.raises(ValidationError):
        BinaryOp(
            left=DummyExpr(dtype=DataType.BOOL),
            right=DummyExpr(dtype=DataType.BOOL),
            op=a_binary_operator,
        )


def test_BinaryOpComparison():
    comparison = BinaryOp(
        left=DummyExpr(dtype=arithmetic_type),
        right=DummyExpr(dtype=arithmetic_type),
        op=ComparisonOperator.EQ,
    )
    assert comparison.dtype == DataType.BOOL


def test_BinaryOpLogicalTypePropagation():
    assert (
        BinaryOp(
            left=DummyExpr(dtype=DataType.BOOL),
            right=DummyExpr(dtype=DataType.BOOL),
            op=LogicalOperator.AND,
        ).dtype
        == DataType.BOOL
    )


def test_BinaryOpLogicalWithArithmeticTypes():
    with pytest.raises(ValueError):
        BinaryOp(
            left=DummyExpr(dtype=arithmetic_type),
            right=DummyExpr(dtype=arithmetic_type),
            op=LogicalOperator.AND,
        )

import pytest
from eve import SourceLocation
from pydantic.error_wrappers import ValidationError

from gt4py.gtc.common import ArithmeticOperator, DataType, LevelMarker, LoopOrder
from gt4py.gtc.gtir import (
    AxisBound,
    CartesianOffset,
    Decl,
    Expr,
    FieldAccess,
    FieldDecl,
    Interval,
    ParAssignStmt,
    Stencil,
    Stmt,
    VerticalLoop,
)

from .gtir_utils import DummyExpr, FieldAccessBuilder, ParAssignStmtBuilder, StencilBuilder


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
        temporaries=[],
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
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


def test_can_have_vertical_offset():
    ParAssignStmt(
        left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1)).build(),
        right=DummyExpr(),
    )


@pytest.mark.parametrize(
    "assign_stmt_with_offset",
    [
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0)).build(),
            right=DummyExpr(),
        ),
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=1, k=0)).build(),
            right=DummyExpr(),
        ),
    ],
)
def test_no_horizontal_offset_allowed(assign_stmt_with_offset):
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        assign_stmt_with_offset()


def test_symbolref_without_decl():
    with pytest.raises(ValidationError, match=r"Symbols.*not found"):
        StencilBuilder().add_par_assign_stmt(
            ParAssignStmtBuilder("out_field", "in_field").build()
        ).build()

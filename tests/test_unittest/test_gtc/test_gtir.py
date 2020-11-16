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

ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD

# IR testing guidelines
# - For testing leave nodes: use the node directly
#   (the builder pattern would in general hide what's being tested)
# - For testing non-leave nodes, introduce builders with defaults (for leave nodes as well)


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


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


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
            BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=ArithmeticOperator.ADD,
            ),
            ARITHMETIC_TYPE,
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
    ],
)
def test_dtype_propagation(node, expected):
    assert node.dtype == expected


@pytest.mark.parameterize(
    "valid_node",
    [
        pytest.param(
            lambda: ParAssignStmt(
                left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1))(),
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
            lambda: TernaryOp(
                cond=DummyExpr(dtype=ARITHMETIC_TYPE),
                true_expr=DummyExpr(),
                false_expr=DummyExpr(),
            ),
            id="condition is not bool",
        ),
        pytest.param(
            lambda: TernaryOp(
                cond=DummyExpr(dtype=DataType.BOOL),
                true_expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                false_expr=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
            ),
            id="expr dtype mismatch",
        ),
        pytest.param(
            lambda: IfStmt(cond=DummyExpr(dtype=ARITHMETIC_TYPE), true_branch=[], false_branch=[]),
            id="condition is not bool",
        ),
        pytest.param(
            lambda: Literal(value="foo"),
            id="missing dtype",
        ),
        pytest.param(
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ANOTHER_ARITHMETIC_TYPE),
                op=A_ARITHMETIC_OPERATOR,
            ),
            id="expr dtype mismatch",
        ),
        pytest.param(
            lambda: BinaryOp(
                left=DummyExpr(dtype=DataType.BOOL),
                right=DummyExpr(dtype=DataType.BOOL),
                op=A_ARITHMETIC_OPERATOR,
            ),
            id="arithmetic operation on boolean expr not allowed",
        ),
        pytest.param(
            lambda: BinaryOp(
                left=DummyExpr(dtype=ARITHMETIC_TYPE),
                right=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=LogicalOperator.AND,
            ),
            id="logical operation on arithmetic exprs not allowed",
        ),
        pytest.param(
            lambda: ParAssignStmt(
                left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0))(),
                right=DummyExpr(),
            ),
            id="non-zero horizontal offset not allowed",
        ),
    ],
)
def test_invalid_nodes(invalid_node):
    with pytest.raises(ValidationError):
        invalid_node()

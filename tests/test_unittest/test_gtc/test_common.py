import pytest

from pydantic import ValidationError
from devtools import debug

from gt4py.gtc.common import (
    IfStmt,
    Literal,
    Expr,
    Stmt,
    DataType,
    ExprKind,
    BinaryOp,
    TernaryOp,
    AssignStmt,
    ArithmeticOperator,
    LogicalOperator,
    ComparisonOperator,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD

# IR testing guidelines
# - For testing leave nodes: use the node directly
#   (the builder pattern would in general hide what's being tested)
# - For testing non-leave nodes, introduce builders with defaults (for leave nodes as well)


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
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
    ],
)
def test_invalid_nodes(invalid_node, expected_regex):
    with pytest.raises(ValidationError, match=expected_regex):
        invalid_node()


# For pydantic, nodes are the same (convertible to each other) if all fields are same.
# For checking, we need to make the Expr categories clearly different.
class ExprA(Expr):
    clearly_a_expr = ""


class DummyExprA(ExprA):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class ExprB(Expr):
    clearly_b_expr = ""


class DummyExprB(ExprB):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class StmtA(Stmt):
    pass


class StmtB(Stmt):
    pass


def test_AssignSmt_category():
    assign = AssignStmt[ExprA, ExprA](left=DummyExprA(), right=DummyExprA())
    debug(assign)

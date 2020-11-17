import pytest

from pydantic import ValidationError

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
# This behavior will most likely change in Eve in the future
class ExprA(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_a = ""


class ExprB(Expr):
    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD
    clearly_expr_b = ""


class StmtA(Stmt):
    clearly_stmt_a = ""


class StmtB(Stmt):
    clearly_stmt_b = ""


def test_AssignSmt_category():
    Testee = AssignStmt[ExprA, ExprA]

    Testee(left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(left=ExprB(), right=ExprA())
        Testee(left=ExprA(), right=ExprB())


def test_IfStmt_category():
    Testee = IfStmt[StmtA, ExprA]

    Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=[StmtA()], false_branch=[StmtA()])
    with pytest.raises(ValidationError):
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=[StmtB()], false_branch=[StmtA()])
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=[StmtA()], false_branch=[StmtB()])
        Testee(cond=ExprB(dtype=DataType.BOOL), true_branch=[StmtA()], false_branch=[StmtA()])


def test_BinaryOp_category():
    Testee = BinaryOp[ExprA]

    Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprB(), right=ExprA())
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprB())


def test_TernaryOp_category():
    Testee = TernaryOp[ExprA]

    Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(cond=ExprB(dtype=DataType.BOOL), true_expr=ExprB(), false_expr=ExprA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprB())

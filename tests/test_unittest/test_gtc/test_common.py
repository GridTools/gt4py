import pytest
from pydantic import ValidationError

from gt4py.gtc import common
from gt4py.gtc.common import (
    ArithmeticOperator,
    ComparisonOperator,
    DataType,
    Expr,
    ExprKind,
    IfStmt,
    Literal,
    LogicalOperator,
    NativeFuncCall,
    NativeFunction,
    Stmt,
    UnaryOperator,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD
A_ARITHMETIC_UNARY_OPERATOR = UnaryOperator.POS
A_LOGICAL_UNARY_OPERATOR = UnaryOperator.NOT

# IR testing guidelines
# - For testing leave nodes: use the node directly
#   (the builder pattern would in general hide what's being tested)
# - For testing non-leave nodes, introduce builders with defaults (for leave nodes as well)


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class UnaryOp(Expr, common.UnaryOp[Expr]):
    pass


class BinaryOp(Expr, common.BinaryOp[Expr]):
    dtype_propagation = common.binary_op_dtype_propagation(strict=True)
    pass


class TernaryOp(Expr, common.TernaryOp[Expr]):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


# TODO test dtype propagation in upcasting mode


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
        (
            UnaryOp(
                expr=DummyExpr(dtype=ARITHMETIC_TYPE),
                op=A_ARITHMETIC_UNARY_OPERATOR,
            ),
            ARITHMETIC_TYPE,
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
        (
            lambda: UnaryOp(op=A_LOGICAL_UNARY_OPERATOR, expr=DummyExpr(dtype=ARITHMETIC_TYPE)),
            r"Unary op.*only .* with bool.*",
        ),
        (
            lambda: UnaryOp(op=A_ARITHMETIC_UNARY_OPERATOR, expr=DummyExpr(dtype=DataType.BOOL)),
            r"Unary op.* not allowed with bool.*",
        ),
        (
            lambda: NativeFuncCall(func=NativeFunction.SIN, args=[DummyExpr(), DummyExpr()]),
            r"accepts 1 arg.* 2.*passed",
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
    Testee = common.AssignStmt[ExprA, ExprA]

    Testee(left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(left=ExprB(), right=ExprA())
        Testee(left=ExprA(), right=ExprB())


def test_IfStmt_category():
    Testee = common.IfStmt[StmtA, ExprA]

    Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())
    with pytest.raises(ValidationError):
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtB(), false_branch=StmtA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtB())
        Testee(cond=ExprB(dtype=DataType.BOOL), true_branch=StmtA(), false_branch=StmtA())


def test_UnaryOp_category():
    class Testee(ExprA, common.UnaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(op=A_ARITHMETIC_UNARY_OPERATOR, expr=ExprB())


def test_BinaryOp_category():
    class Testee(ExprA, common.BinaryOp[ExprA]):
        pass

    Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprA())
    with pytest.raises(ValidationError):
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprB(), right=ExprA())
        Testee(op=A_ARITHMETIC_OPERATOR, left=ExprA(), right=ExprB())


def test_TernaryOp_category():
    class Testee(ExprA, common.TernaryOp[ExprA]):
        pass

    Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(cond=ExprB(dtype=DataType.BOOL), true_expr=ExprB(), false_expr=ExprA())
        Testee(cond=ExprA(dtype=DataType.BOOL), true_expr=ExprA(), false_expr=ExprB())


def test_Cast_category():
    class Testee(ExprA, common.Cast[ExprA]):
        pass

    Testee(dtype=ARITHMETIC_TYPE, expr=ExprA())
    with pytest.raises(ValidationError):
        Testee(dtype=ARITHMETIC_TYPE, expr=ExprB())


def test_NativeFuncCall_category():
    class Testee(ExprA, common.NativeFuncCall[ExprA]):
        pass

    Testee(func=NativeFunction.SIN, args=[ExprA()])
    with pytest.raises(ValidationError):
        Testee(func=NativeFunction.SIN, args=[ExprB()])

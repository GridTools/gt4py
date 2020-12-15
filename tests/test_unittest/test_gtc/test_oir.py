import pytest
from pydantic.error_wrappers import ValidationError

from gt4py.gtc.common import CartesianOffset, DataType, ExprKind
from gt4py.gtc.oir import AssignStmt, Expr, FieldAccess, HorizontalExecution


A_ARITHMETIC_TYPE = DataType.INT32


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


# def test_dtype_required():
#     with pytest.raises(ValidationError):
#         DummyExpr(dtype=None)


@pytest.mark.parametrize(
    "invalid_node,expected_regex",
    [
        (
            lambda: HorizontalExecution(body=[], mask=DummyExpr(dtype=A_ARITHMETIC_TYPE)),
            r".*must be.* bool.*",
        ),
        (
            lambda: AssignStmt(
                left=FieldAccess(
                    name="foo", dtype=A_ARITHMETIC_TYPE, offset=CartesianOffset(i=1, j=0, k=0)
                ),
                right=DummyExpr(dtype=A_ARITHMETIC_TYPE),
            ),
            r"must not have .*horizontal offset",
        ),
    ],
)
def test_invalid_nodes(invalid_node, expected_regex):
    with pytest.raises(ValidationError, match=expected_regex):
        invalid_node()

import pytest
from pydantic.error_wrappers import ValidationError

from gtc.common import CartesianOffset, DataType, ExprKind
from gtc.oir import AssignStmt, Expr, FieldAccess, HorizontalExecution


A_ARITHMETIC_TYPE = DataType.INT32


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmt(
            left=FieldAccess(
                name="foo", dtype=A_ARITHMETIC_TYPE, offset=CartesianOffset(i=1, j=0, k=0)
            ),
            right=DummyExpr(dtype=A_ARITHMETIC_TYPE),
        ),


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        HorizontalExecution(body=[], mask=DummyExpr(dtype=A_ARITHMETIC_TYPE)),

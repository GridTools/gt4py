from gt4py.gtc.oir import Expr, HorizontalExecution
from gt4py.gtc.common import ExprKind, DataType

from pydantic.error_wrappers import ValidationError
import pytest

A_ARITHMETIC_TYPE = DataType.INT32


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


def test_dtype_required():
    with pytest.raises(ValidationError):
        DummyExpr(dtype=None)


@pytest.mark.parametrize(
    "invalid_node,expected_regex",
    [
        (
            lambda: HorizontalExecution(body=[], mask=DummyExpr(dtype=A_ARITHMETIC_TYPE)),
            r".*must be.* bool.*",
        ),
        (
            lambda: HorizontalExecution(
                body=[], mask=DummyExpr(dtype=DataType.BOOL, kind=ExprKind.SCALAR)
            ),
            r".*must be.* field.*",
        ),
    ],
)
def test_invalid_nodes(invalid_node, expected_regex):
    with pytest.raises(ValidationError, match=expected_regex):
        invalid_node()

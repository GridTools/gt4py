from gt4py.gtc.oir import Expr
from gt4py.gtc.common import ExprKind

from pydantic.error_wrappers import ValidationError
import pytest


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    kind: ExprKind = ExprKind.FIELD


def test_dtype_required():
    with pytest.raises(ValidationError):
        DummyExpr(dtype=None)

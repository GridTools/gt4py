from gt4py.gtc.gtir import CartesianOffset, FieldAccess, Expr
from gt4py.gtc.common import DataType, ExprKind


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class FieldAccessBuilder:
    def __init__(self, name) -> None:
        self.node = FieldAccess(name=name, offset=CartesianOffset.zero())

    def offset(self, offset: CartesianOffset) -> "FieldAccessBuilder":
        self.node = FieldAccess(name=self.node.name, offset=offset)
        return self

    def build(self) -> FieldAccess:
        return self.node

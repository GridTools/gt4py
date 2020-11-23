from gt4py.gtc.gtir import (
    AxisBound,
    CartesianOffset,
    FieldAccess,
    Expr,
    Interval,
    ParAssignStmt,
    Stencil,
    Decl,
    VerticalLoop,
)
from gt4py.gtc.common import DataType, ExprKind, LoopOrder


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class FieldAccessBuilder:
    def __init__(self, name) -> None:
        self.data = {}
        self.data["name"] = name
        self.data["offset"] = CartesianOffset.zero()

    def offset(self, offset: CartesianOffset) -> "FieldAccessBuilder":
        self.data["offset"] = offset
        return self

    def build(self) -> FieldAccess:
        return FieldAccess(name=self.data["name"], offset=self.data["offset"])


class StencilBuilder:
    def __init__(self, name="foo") -> None:
        self.data = {}
        self.data["name"] = name
        self.data["params"] = []
        self.data["vertical_loops"] = []

    def add_param(self, param: Decl) -> "StencilBuilder":
        self.data["params"].append(param),
        return self

    def add_vertical_loop(self, vertical_loop: VerticalLoop) -> "StencilBuilder":
        self.data["vertical_loops"].append(vertical_loop)
        return self

    def add_par_assign_stmt(self, par_assign_stmt: ParAssignStmt) -> "StencilBuilder":
        if len(self.data["vertical_loops"]) == 0:
            self.data["vertical_loops"].append(  # TODO builder
                VerticalLoop(
                    interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                    loop_order=LoopOrder.FORWARD,
                    body=[],
                )
            )

        self.data["vertical_loops"][-1].body.append(par_assign_stmt)
        return self

    def build(self) -> Stencil:
        return Stencil(
            name=self.data["name"],
            params=self.data["params"],
            vertical_loops=self.data["vertical_loops"],
        )

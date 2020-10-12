from types import MappingProxyType
from typing import ClassVar, Dict, List, Mapping

from gt4py.backend.gtc_backend import common, gtir
from gt4py.ir import IRNodeVisitor
from gt4py.ir.nodes import (
    Assign,
    AxisBound,
    AxisInterval,
    BinaryOperator,
    BinOpExpr,
    BlockStmt,
    ComputationBlock,
    FieldDecl,
    FieldRef,
    IterationOrder,
    LevelMarker,
    ScalarLiteral,
    StencilDefinition,
)


def transform_offset(offset: Dict[str, int]) -> gtir.CartesianOffset:
    i = offset["I"] if "I" in offset else 0
    j = offset["J"] if "J" in offset else 0
    k = offset["K"] if "K" in offset else 0
    return gtir.CartesianOffset(i=i, j=j, k=k)


# TODO(Rico Häuselmann): write unit tests


class DefIRToGTIR(IRNodeVisitor):
    @classmethod
    def apply(cls, root, **kwargs):
        return cls().visit(root)

    GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER: ClassVar[
        Mapping[IterationOrder, int]
    ] = MappingProxyType(
        {
            IterationOrder.BACKWARD: common.LoopOrder.BACKWARD,
            IterationOrder.PARALLEL: common.LoopOrder.PARALLEL,
            IterationOrder.FORWARD: common.LoopOrder.FORWARD,
        }
    )

    GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER: ClassVar[Mapping[LevelMarker, str]] = MappingProxyType(
        {LevelMarker.START: common.LevelMarker.START, LevelMarker.END: common.LevelMarker.END}
    )

    GT4PY_OP_TO_GTIR_OP: ClassVar[Mapping[BinaryOperator, str]] = MappingProxyType(
        {
            BinaryOperator.ADD: common.BinaryOperator.ADD,
            BinaryOperator.SUB: common.BinaryOperator.SUB,
            BinaryOperator.MUL: common.BinaryOperator.MUL,
            BinaryOperator.DIV: common.BinaryOperator.DIV,
        }
    )

    def visit_StencilDefinition(self, node: StencilDefinition):
        stencils = [self.visit(c) for c in node.computations]
        return gtir.Computation(
            name=node.name, params=[self.visit(f) for f in node.api_fields], stencils=stencils
        )

    def visit_ComputationBlock(self, node: ComputationBlock):
        horizontal_loops = [gtir.HorizontalLoop(stmt=s) for s in self.visit(node.body)]
        start, end = self.visit(node.interval)
        vertical_intervals = [
            gtir.VerticalInterval(horizontal_loops=horizontal_loops, start=start, end=end)
        ]
        return gtir.Stencil(
            vertical_loops=[
                gtir.VerticalLoop(
                    loop_order=self.GT4PY_ITERATIONORDER_TO_GTIR_LOOPORDER[node.iteration_order],
                    vertical_intervals=vertical_intervals,
                )
            ]
        )

    def visit_BlockStmt(self, node: BlockStmt) -> List[gtir.Stmt]:
        return [self.visit(s) for s in node.stmts]

    def visit_Assign(self, node: Assign):
        assert isinstance(node.target, FieldRef)
        left = self.visit(node.target)
        return gtir.AssignStmt(left=left, right=self.visit(node.value))

    def visit_ScalarLiteral(self, node: ScalarLiteral):
        return gtir.Literal(value=str(node.value), dtype=common.DataType(node.data_type.value))

    def visit_BinOpExpr(self, node: BinOpExpr):
        return gtir.BinaryOp(
            left=self.visit(node.lhs),
            right=self.visit(node.rhs),
            op=self.GT4PY_OP_TO_GTIR_OP[node.op],
        )

    def visit_FieldRef(self, node: FieldRef):
        return gtir.FieldAccess(name=node.name, offset=transform_offset(node.offset))

    def visit_AxisInterval(self, node: AxisInterval):
        return self.visit(node.start), self.visit(node.end)

    def visit_AxisBound(self, node: AxisBound):
        # TODO support VarRef
        return gtir.AxisBound(
            level=self.GT4PY_LEVELMARKER_TO_GTIR_LEVELMARKER[node.level], offset=node.offset
        )

    def visit_FieldDecl(self, node: FieldDecl):
        # datatype conversion works via same ID
        return gtir.FieldDecl(name=node.name, dtype=common.DataType(int(node.data_type.value)))

from typing import Dict, List, Optional

from dataclasses import dataclass, field
from eve.visitors import NodeTranslator

from .. import common, oir
from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    @dataclass
    class Context:
        """Context for a HorizontalExecution."""

        domain_padding: Dict[str, List] = field(
            default_factory=lambda: {"lower": [0, 0, 0], "upper": [0, 0, 0]}
        )
        parallel_k: Optional[bool] = None

        def set_parallel_k(self, value: bool) -> "OirToNpir.Context":
            return self.__class__(parallel_k=value, domain_padding=self.domain_padding)

        def add_offset(self, axis: int, value: int) -> "OirToNpir.Context":
            if value > 0:
                current_value = self.domain_padding["upper"][axis]
                self.domain_padding["upper"][axis] = max(current_value, value)
            elif value < 0:
                current_value = self.domain_padding["lower"][axis]
                self.domain_padding["lower"][axis] = max(current_value, abs(value))
            return self

        def add_offsets(self, i: int, j: int, k: int) -> "OirToNpir.Context":
            for axis_index, value in enumerate([i, j, k]):
                self.add_offset(axis_index, value)
            return self

    def visit_Stencil(
        self, node: oir.Stencil, *, ctx: Optional[Context] = None
    ) -> npir.Computation:
        ctx = ctx or self.Context()
        vertical_passes = [self.visit(vloop, ctx=ctx) for vloop in node.vertical_loops]

        return npir.Computation(
            field_params=[decl.name for decl in node.params if isinstance(decl, oir.FieldDecl)],
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
            domain_padding=npir.DomainPadding(
                lower=ctx.domain_padding["lower"],
                upper=ctx.domain_padding["upper"],
            ),
        )

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, ctx: Optional[Context] = None
    ) -> npir.VerticalPass:
        ctx = ctx or self.Context()
        parallel_k = True if node.loop_order == common.LoopOrder.PARALLEL else False
        v_assigns = [
            self.visit(h_exec, ctx=ctx.set_parallel_k(parallel_k))
            for h_exec in node.horizontal_executions
        ]
        body = [i for j in v_assigns for i in j]
        return npir.VerticalPass(
            body=body,
            lower=self.visit(node.interval.start),
            upper=self.visit(node.interval.end),
            direction=self.visit(node.loop_order),
        )

    def visit_AxisBound(self, node: oir.AxisBound) -> common.AxisBound:
        return common.AxisBound(level=node.level, offset=node.offset)

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, *, ctx: Optional[Context] = None
    ) -> List[npir.VectorAssign]:
        return self.visit(node.body, ctx=ctx)

    def visit_AssignStmt(
        self, node: oir.AssignStmt, *, ctx: Optional[Context] = None
    ) -> npir.VectorAssign:
        return npir.VectorAssign(
            left=self.visit(node.left, ctx=ctx),
            right=self.visit(node.right, ctx=ctx),
        )

    def visit_FieldAccess(self, node: oir.FieldAccess, *, ctx: Context) -> npir.FieldSlice:
        ctx.add_offsets(node.offset.i, node.offset.j, node.offset.k)
        return npir.FieldSlice(
            name=str(node.name),
            i_offset=npir.AxisOffset.i(node.offset.i),
            j_offset=npir.AxisOffset.j(node.offset.j),
            k_offset=npir.AxisOffset.k(node.offset.k, parallel=ctx.parallel_k),
        )

    def visit_BinaryOp(
        self, node: oir.BinaryOp, *, ctx: Optional[Context] = None
    ) -> npir.VectorArithmetic:
        return npir.VectorArithmetic(
            op=node.op,
            left=self.visit(node.left, ctx=ctx),
            right=self.visit(node.right, ctx=ctx),
        )

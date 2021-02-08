# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from eve.visitors import NodeTranslator

from .. import common, oir
from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    @dataclass
    class Context:
        """Context for a HorizontalExecution."""

        symbol_table: Dict[str, Any] = field(default_factory=lambda: {})

        domain_padding: Dict[str, List] = field(
            default_factory=lambda: {"lower": [0, 0, 0], "upper": [0, 0, 0]}
        )

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

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> npir.Computation:
        ctx = self.Context(symbol_table=node.symtable_)
        vertical_passes = [self.visit(vloop, ctx=ctx, **kwargs) for vloop in node.vertical_loops]
        field_names: List[str] = []
        scalar_names: List[str] = []
        for decl in node.params:
            if isinstance(decl, oir.FieldDecl):
                field_names.append(decl.name)
            else:
                scalar_names.append(decl.name)
        return npir.Computation(
            field_params=field_names,
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
            domain_padding=npir.DomainPadding(
                lower=ctx.domain_padding["lower"],
                upper=ctx.domain_padding["upper"],
            ),
        )

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, ctx: Optional[Context] = None, **kwargs: Any
    ) -> npir.VerticalPass:
        ctx = ctx or self.Context()
        kwargs.update(
            {
                "parallel_k": True if node.loop_order == common.LoopOrder.PARALLEL else False,
            }
        )
        v_assigns = [self.visit(h_exec, ctx=ctx, **kwargs) for h_exec in node.horizontal_executions]
        body = [i for j in v_assigns for i in j]
        return npir.VerticalPass(
            body=body,
            lower=self.visit(node.interval.start, ctx=ctx, **kwargs),
            upper=self.visit(node.interval.end, ctx=ctx, **kwargs),
            direction=self.visit(node.loop_order, ctx=ctx, **kwargs),
        )

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, *, ctx: Optional[Context] = None, **kwargs: Any
    ) -> List[npir.VectorAssign]:
        mask = self.visit(node.mask, ctx=ctx, **kwargs)
        return self.visit(node.body, ctx=ctx, mask=mask, **kwargs)

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        ctx: Optional[Context] = None,
        mask: Optional[npir.VectorExpression],
        **kwargs: Any,
    ) -> npir.VectorAssign:
        return npir.VectorAssign(
            left=self.visit(node.left, ctx=ctx, **kwargs),
            right=self.visit(node.right, ctx=ctx, broadcast=True, **kwargs),
            mask=mask,
        )

    def visit_Cast(
        self,
        node: oir.Cast,
        *,
        ctx: Optional[Context] = None,
        broadcast: bool = False,
        **kwargs: Any,
    ) -> Union[npir.Cast, npir.BroadCast]:
        cast = npir.Cast(
            dtype=self.visit(node.dtype, ctx=ctx, **kwargs),
            expr=self.visit(node.expr, ctx=ctx, broadcast=False, **kwargs),
        )
        if broadcast:
            return npir.BroadCast(expr=cast, dtype=node.dtype)
        return cast

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, ctx: Context, parallel_k: bool, **kwargs: Any
    ) -> Union[npir.FieldSlice, npir.VectorTemp]:
        if isinstance(ctx.symbol_table.get(node.name, None), oir.Temporary):
            return npir.VectorTemp(name=node.name)
        ctx.add_offsets(node.offset.i, node.offset.j, node.offset.k)
        return npir.FieldSlice(
            name=str(node.name),
            i_offset=npir.AxisOffset.i(node.offset.i),
            j_offset=npir.AxisOffset.j(node.offset.j),
            k_offset=npir.AxisOffset.k(node.offset.k, parallel=parallel_k),
        )

    def visit_BinaryOp(
        self, node: oir.BinaryOp, *, ctx: Optional[Context] = None, **kwargs: Any
    ) -> npir.VectorArithmetic:
        kwargs["broadcast"] = True
        return npir.VectorArithmetic(
            op=node.op,
            left=self.visit(node.left, ctx=ctx, **kwargs),
            right=self.visit(node.right, ctx=ctx, **kwargs),
        )

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> npir.VectorUnaryOp:
        kwargs["broadcast"] = True
        return npir.VectorUnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> npir.VectorTernaryOp:
        kwargs["broadcast"] = True

        return npir.VectorTernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> npir.NativeFuncCall:
        kwargs["broadcast"] = True
        return npir.NativeFuncCall(
            func=self.visit(node.func, **kwargs),
            args=self.visit(node.args, **kwargs),
        )

    def visit_Literal(
        self, node: oir.Literal, *, broadcast: bool = False, **kwargs: Any
    ) -> Union[npir.Literal, npir.BroadCast]:
        literal = npir.Literal(value=node.value, dtype=node.dtype)
        if broadcast:
            return npir.BroadCast(expr=literal, dtype=node.dtype)
        return literal

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, *, broadcast: bool = False, **kwargs: Any
    ) -> Union[npir.BroadCast, npir.NamedScalar]:
        name = npir.NamedScalar(
            name=self.visit(node.name, **kwargs), dtype=self.visit(node.dtype, **kwargs)
        )
        if broadcast:
            return npir.BroadCast(expr=name, dtype=name.dtype)
        return name

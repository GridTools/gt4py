# -*- coding: utf-8 -*-
import itertools
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from eve.visitors import NodeTranslator

from .. import common, oir
from . import npir


@dataclass
class Padding:
    lower: List[int] = field(default_factory=lambda: [0, 0, 0])
    upper: List[int] = field(default_factory=lambda: [0, 0, 0])

    def add_offset(self, axis: int, value: int) -> "Padding":
        if value > 0:
            current_value = self.upper[axis]
            self.upper[axis] = max(current_value, value)
        elif value < 0:
            current_value = self.lower[axis]
            self.lower[axis] = max(current_value, abs(value))
        return self

    def add_offsets(self, i: int, j: int, k: int) -> "Padding":
        for axis_index, value in enumerate([i, j, k]):
            self.add_offset(axis_index, value)
        return self


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    @dataclass
    class ComputationContext:
        """Top Level Context."""

        symbol_table: Dict[str, Any] = field(default_factory=lambda: {})

        domain_padding: Padding = field(default_factory=lambda: Padding())

        field_padding: Dict[str, Padding] = field(default_factory=lambda: {})

        temp_defs: typing.OrderedDict[str, npir.VectorAssign] = field(
            default_factory=lambda: OrderedDict({})
        )

        mask_temp_counter: int = 0

        def add_offset(self, name: str, axis: int, value: int) -> "OirToNpir.ComputationContext":
            fpad = self.field_padding.setdefault(name, Padding())
            self.domain_padding.add_offset(axis, value)
            fpad.add_offset(axis, value)
            return self

        def add_offsets(self, name: str, i: int, j: int, k: int) -> "OirToNpir.ComputationContext":
            for axis_index, value in enumerate([i, j, k]):
                self.add_offset(name, axis_index, value)
            return self

        def ensure_temp_defined(self, temp: Union[oir.FieldAccess, npir.FieldSlice]) -> None:
            if temp.name not in self.temp_defs:
                self.temp_defs[str(temp.name)] = npir.VectorAssign(
                    left=npir.VectorTemp(name=str(temp.name), dtype=temp.dtype),
                    right=npir.EmptyTemp(dtype=temp.dtype),
                )

    @dataclass
    class HorizontalContext:
        """Context for a single HorizontalExecution."""

        padding: Padding = field(default_factory=lambda: Padding())

        def add_offsets(self, i: int, j: int) -> "OirToNpir.HorizontalContext":
            for axis_index, value in enumerate([i, j, 0]):
                self.padding.add_offset(axis_index, value)
            return self

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> npir.Computation:
        ctx = self.ComputationContext(symbol_table=node.symtable_)
        vertical_passes = list(
            itertools.chain(
                *[self.visit(vloop, ctx=ctx, **kwargs) for vloop in node.vertical_loops]
            )
        )
        field_names: List[str] = []
        scalar_names: List[str] = []
        for decl in node.params:
            if isinstance(decl, oir.FieldDecl):
                field_names.append(decl.name)
            else:
                scalar_names.append(decl.name)
        return npir.Computation(
            field_params=field_names,
            field_paddings={
                key: {"lower": val.lower, "upper": val.upper}
                for key, val in ctx.field_padding.items()
            },
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
            domain_padding=npir.DomainPadding(
                lower=ctx.domain_padding.lower,
                upper=ctx.domain_padding.upper,
            ),
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs) -> List[npir.VerticalPass]:
        return self.visit(node.sections, v_caches=node.caches, loop_order=node.loop_order, **kwargs)

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        loop_order: common.LoopOrder,
        ctx: Optional[ComputationContext] = None,
        v_caches: List[oir.CacheDesc] = None,
        **kwargs: Any,
    ) -> npir.VerticalPass:
        ctx = ctx or self.ComputationContext()
        defined_temps = set(ctx.temp_defs.keys())
        kwargs.update(
            {
                "parallel_k": True if loop_order == common.LoopOrder.PARALLEL else False,
                "lower_k": node.interval.start,
                "upper_k": node.interval.end,
            }
        )
        body = self.visit(node.horizontal_executions, ctx=ctx, **kwargs)
        undef_temps = [
            temp_def for name, temp_def in ctx.temp_defs.items() if name not in defined_temps
        ]
        return npir.VerticalPass(
            body=body,
            temp_defs=undef_temps,
            lower=self.visit(node.interval.start, ctx=ctx, **kwargs),
            upper=self.visit(node.interval.end, ctx=ctx, **kwargs),
            direction=self.visit(loop_order, ctx=ctx, **kwargs),
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        ctx: Optional[ComputationContext] = None,
        h_ctx: Optional[HorizontalContext] = None,
        **kwargs: Any,
    ) -> npir.HorizontalRegion:
        h_ctx = h_ctx or self.HorizontalContext()
        return npir.HorizontalRegion(
            body=self.visit(node.body, ctx=ctx, h_ctx=h_ctx, **kwargs),
            padding=npir.DomainPadding(lower=h_ctx.padding.lower, upper=h_ctx.padding.upper),
        )

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        ctx: ComputationContext,
        h_ctx: Optional[HorizontalContext] = None,
        parallel_k: bool,
        **kwargs,
    ) -> npir.MaskBlock:
        mask_expr = self.visit(
            node.mask, ctx=ctx, h_ctx=h_ctx, parallel_k=parallel_k, broadcast=True, **kwargs
        )
        if isinstance(mask_expr, npir.FieldSlice):
            mask_name = mask_expr.name
            mask = mask_expr
        else:
            mask_name = f"_mask_{ctx.mask_temp_counter}"
            mask = npir.VectorTemp(
                name=mask_name,
            )
            ctx.mask_temp_counter += 1

        return npir.MaskBlock(
            mask=mask_expr,
            mask_name=mask_name,
            body=self.visit(
                node.body, ctx=ctx, h_ctx=h_ctx, parallel_k=parallel_k, mask=mask, **kwargs
            ),
        )

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        ctx: Optional[ComputationContext] = None,
        mask: Optional[npir.VectorExpression] = None,
        **kwargs: Any,
    ) -> npir.VectorAssign:
        ctx = ctx or self.ComputationContext()
        if isinstance(ctx.symbol_table.get(node.left.name, None), oir.Temporary):
            ctx.ensure_temp_defined(node.left)
        return npir.VectorAssign(
            left=self.visit(node.left, ctx=ctx, is_lvalue=True, **kwargs),
            right=self.visit(node.right, ctx=ctx, broadcast=True, **kwargs),
            mask=mask,
        )

    def visit_Cast(
        self,
        node: oir.Cast,
        *,
        ctx: Optional[ComputationContext] = None,
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
        self,
        node: oir.FieldAccess,
        *,
        ctx: ComputationContext,
        h_ctx: HorizontalContext,
        parallel_k: bool,
        **kwargs: Any,
    ) -> npir.FieldSlice:
        if not isinstance(ctx.symbol_table.get(node.name, None), oir.Temporary):
            k_offset = node.offset.k
            if (lower_k := kwargs.get("lower_k")) and k_offset < 0:
                if lower_k.level is common.LevelMarker.START:
                    k_offset = min(k_offset + lower_k.offset, 0)
                else:
                    k_offset = 0
            if (upper_k := kwargs.get("upper_k")) and k_offset > 0:
                if upper_k.level is common.LevelMarker.END:
                    k_offset = max(k_offset - upper_k.offset, 0)
                else:
                    k_offset = 0
            ctx.add_offsets(str(node.name), node.offset.i, node.offset.j, k_offset)
            h_ctx.add_offsets(node.offset.i, node.offset.j)
        return npir.FieldSlice(
            name=str(node.name),
            i_offset=npir.AxisOffset.i(node.offset.i),
            j_offset=npir.AxisOffset.j(node.offset.j),
            k_offset=npir.AxisOffset.k(node.offset.k, parallel=parallel_k),
        )

    def visit_BinaryOp(
        self, node: oir.BinaryOp, *, ctx: Optional[ComputationContext] = None, **kwargs: Any
    ) -> Union[npir.VectorArithmetic, npir.VectorLogic]:
        kwargs["broadcast"] = True
        left = self.visit(node.left, ctx=ctx, **kwargs)
        right = self.visit(node.right, ctx=ctx, **kwargs)

        if isinstance(node.op, common.LogicalOperator):
            return npir.VectorLogic(op=node.op, left=left, right=right)

        return npir.VectorArithmetic(
            op=node.op,
            left=left,
            right=right,
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
        literal = npir.Literal(value=self.visit(node.value, **kwargs), dtype=node.dtype)
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

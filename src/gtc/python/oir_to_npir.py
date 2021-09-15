# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import itertools
import typing
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Dict, Tuple

from eve.traits import SymbolTableTrait
from eve.visitors import NodeTranslator

from .. import common, oir
from . import npir


class AssignStmt(oir.AssignStmt):
    """AssignStmt used in the lowering from oir to npir."""

    horiz_mask: Optional[oir.HorizontalMask] = None


class HorizontalMaskInliner(NodeTranslator):
    def __init__(self):
        self.horiz_mask: Optional[oir.HorizontalMask] = None

    @classmethod
    def apply(cls, stencil: oir.Stencil) -> Tuple[oir.Stencil, Dict[int, oir.HorizontalMask]]:
        transformer = cls()
        return transformer.visit(copy.deepcopy(stencil))

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> oir.Stencil:
        return oir.Stencil(
            name=node.name,
            params=node.params,
            vertical_loops=self.visit(node.vertical_loops, **kwargs),
            declarations=node.declarations,
        )

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> oir.Expr:
        if isinstance(node.left, oir.HorizontalMask):
            self.horiz_mask = node.left
            return node.right
        elif isinstance(node.right, oir.HorizontalMask):
            self.horiz_mask = node.right
            return node.left
        else:
            return node

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs: Any) -> oir.HorizontalMask:
        self.horiz_mask = node
        return None

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> AssignStmt:
        return AssignStmt(left=node.left, right=node.right, horiz_mask=self.horiz_mask)

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, **kwargs: Any
    ) -> oir.HorizontalExecution:
        body = []
        for stmt in node.body:
            # NOTE(jdahm): The following assumes with horizontal(...) cannot be nested under MaskStmts, For, While, etc.
            if isinstance(stmt, oir.MaskStmt):
                # (Re)set horiz_mask to None before visit call
                self.horiz_mask = None
                mask_wout_regions = self.visit(stmt.mask, **kwargs)
                stmts = [self.visit(sub_stmt, **kwargs) for sub_stmt in stmt.body]
                if mask_wout_regions is None:
                    # The mask was only a horizontal restriction, so inline the sub-stmts
                    body.extend(stmts)
                else:
                    # This mask had more than a horizontal restriction
                    body.append(oir.MaskStmt(mask=mask_wout_regions, body=stmts))
                # (Re)set horiz_mask to None after visit call
                self.horiz_mask = None
            else:
                body.append(self.visit(stmt, **kwargs))

        return oir.HorizontalExecution(body=body, declarations=node.declarations)


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    @dataclass
    class ComputationContext:
        """Top Level Context."""

        temp_defs: typing.OrderedDict[str, npir.VectorAssign] = field(
            default_factory=lambda: OrderedDict({})
        )

        mask_temp_counter: int = 0

        def ensure_temp_defined(self, temp: Union[oir.FieldAccess, npir.FieldSlice]) -> None:
            if temp.name not in self.temp_defs:
                self.temp_defs[str(temp.name)] = npir.VectorAssign(
                    left=npir.VectorTemp(name=str(temp.name), dtype=temp.dtype),
                    right=npir.EmptyTemp(dtype=temp.dtype),
                )

    contexts = (SymbolTableTrait.symtable_merger,)

    def visit_Stencil(self, oir_node: oir.Stencil, **kwargs: Any) -> npir.Computation:
        node = HorizontalMaskInliner.apply(oir_node)
        ctx = self.ComputationContext()
        vertical_passes = list(
            itertools.chain(
                *[self.visit(vloop, ctx=ctx, **kwargs) for vloop in node.vertical_loops]
            )
        )
        field_names: List[str] = []
        scalar_names: List[str] = []
        field_decls: List[npir.FieldDecl] = []
        for decl in node.params:
            if isinstance(decl, oir.FieldDecl):
                field_names.append(str(decl.name))
                field_decls.append(self.visit(decl))
            else:
                scalar_names.append(decl.name)
        return npir.Computation(
            field_decls=field_decls,
            field_params=field_names,
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
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
        **kwargs: Any,
    ) -> npir.HorizontalBlock:
        return npir.HorizontalBlock(
            body=self.visit(node.body, ctx=ctx, **kwargs),
        )

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs: Any) -> npir.HorizontalMask:
        return npir.HorizontalMask(i=node.i, j=node.j)

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        ctx: ComputationContext,
        parallel_k: bool,
        **kwargs,
    ) -> npir.MaskBlock:
        mask_expr = self.visit(node.mask, ctx=ctx, parallel_k=parallel_k, broadcast=True, **kwargs)
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
            body=self.visit(node.body, ctx=ctx, parallel_k=parallel_k, mask=mask, **kwargs),
        )

    def visit_AssignStmt(
        self,
        node: AssignStmt,
        *,
        ctx: Optional[ComputationContext] = None,
        mask: Optional[npir.VectorExpression] = None,
        **kwargs: Any,
    ) -> npir.VectorAssign:
        ctx = ctx or self.ComputationContext()
        if isinstance(kwargs["symtable"].get(node.left.name, None), oir.Temporary):
            ctx.ensure_temp_defined(node.left)
        return npir.VectorAssign(
            left=self.visit(node.left, ctx=ctx, is_lvalue=True, **kwargs),
            right=self.visit(node.right, ctx=ctx, broadcast=True, **kwargs),
            mask=mask,
            horiz_mask=node.horiz_mask,
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
        parallel_k: bool,
        **kwargs: Any,
    ) -> npir.FieldSlice:
        dims = (
            decl.dimensions if (decl := kwargs["symtable"].get(node.name)) else (True, True, True)
        )
        return npir.FieldSlice(
            name=str(node.name),
            i_offset=npir.AxisOffset.i(node.offset.i) if dims[0] else None,
            j_offset=npir.AxisOffset.j(node.offset.j) if dims[1] else None,
            k_offset=npir.AxisOffset.k(node.offset.k, parallel=parallel_k) if dims[2] else None,
        )

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs) -> npir.FieldDecl:
        return npir.FieldDecl(
            name=node.name,
            dtype=self.visit(node.dtype),
            dimensions=node.dimensions,
            data_dims=node.data_dims,
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

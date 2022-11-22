# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union, cast

from typing_extensions import Protocol

import eve
from gtc import common, oir
from gtc.cuir import cuir
from gtc.passes.oir_optimizations.utils import (
    collect_symbol_names,
    compute_horizontal_block_extents,
    symbol_name_creator,
)


class SymbolNameCreator(Protocol):
    def __call__(self, name: str) -> str:
        ...


def _make_axis_offset_expr(bound: common.AxisBound, axis_index: int) -> cuir.Expr:
    if bound.level == common.LevelMarker.END:
        base = cuir.ScalarAccess(
            name="{}_size".format(["i", "j"][axis_index]), dtype=common.DataType.INT32
        )
        return cuir.BinaryOp(
            op=common.ArithmeticOperator.ADD,
            left=base,
            right=cuir.Literal(value=str(bound.offset), dtype=common.DataType.INT32),
        )
    else:
        return cuir.Literal(value=str(bound.offset), dtype=common.DataType.INT32)


class OIRToCUIR(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    @dataclass
    class Context:
        new_symbol_name: SymbolNameCreator
        accessed_fields: Set[str] = field(default_factory=set)
        positionals: Dict[int, cuir.Positional] = field(default_factory=dict)

        def make_positional(self, axis: int) -> cuir.FieldAccess:
            axis_name = ["i", "j", "k"][axis]
            positional = self.positionals.setdefault(
                axis,
                cuir.Positional(
                    name=self.new_symbol_name(f"axis_{axis_name}_index"), axis_name=axis_name
                ),
            )
            return cuir.FieldAccess(
                name=positional.name,
                offset=common.CartesianOffset.zero(),
                dtype=common.DataType.INT32,
            )

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> cuir.Literal:
        return cuir.Literal(value=node.value, dtype=node.dtype)

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.FieldDecl(
            name=node.name, dtype=node.dtype, dimensions=node.dimensions, data_dims=node.data_dims
        )

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> cuir.ScalarDecl:
        return cuir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> cuir.LocalScalar:
        return cuir.LocalScalar(name=node.name, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> cuir.UnaryOp:
        return cuir.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs), dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> cuir.BinaryOp:
        return cuir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
        )

    def visit_Temporary(self, node: oir.Temporary, **kwargs: Any) -> cuir.Temporary:
        return cuir.Temporary(
            name=node.name, data_dims=self.visit(node.data_dims, **kwargs), dtype=node.dtype
        )

    def visit_VariableKOffset(
        self, node: cuir.VariableKOffset, **kwargs: Any
    ) -> cuir.VariableKOffset:
        return cuir.VariableKOffset(k=self.visit(node.k, **kwargs))

    def _mask_to_expr(self, mask: common.HorizontalMask, ctx: "Context") -> cuir.Expr:
        mask_expr: List[cuir.Expr] = []
        for axis_index, interval in enumerate(mask.intervals):
            if interval.is_single_index():
                assert interval.start is not None
                mask_expr.append(
                    cuir.BinaryOp(
                        op=common.ComparisonOperator.EQ,
                        left=ctx.make_positional(axis_index),
                        right=_make_axis_offset_expr(interval.start, axis_index),
                    )
                )
            else:
                for op, endpt in zip(
                    (common.ComparisonOperator.GE, common.ComparisonOperator.LT),
                    (interval.start, interval.end),
                ):
                    if endpt is None:
                        continue
                    mask_expr.append(
                        cuir.BinaryOp(
                            op=op,
                            left=ctx.make_positional(axis_index),
                            right=_make_axis_offset_expr(endpt, axis_index),
                        )
                    )
        return (
            functools.reduce(
                lambda a, b: cuir.BinaryOp(op=common.LogicalOperator.AND, left=a, right=b),
                mask_expr,
            )
            if mask_expr
            else cuir.Literal(value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL)
        )

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, **kwargs: Any
    ) -> cuir.MaskStmt:
        mask = self._mask_to_expr(node.mask, kwargs["ctx"])
        return cuir.MaskStmt(mask=mask, body=self.visit(node.body, **kwargs))

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        ij_caches: Dict[str, cuir.IJCacheDecl],
        k_caches: Dict[str, cuir.KCacheDecl],
        ctx: "Context",
        **kwargs: Any,
    ) -> Union[cuir.FieldAccess, cuir.IJCacheAccess, cuir.KCacheAccess]:
        data_index = self.visit(
            node.data_index,
            ij_caches=ij_caches,
            k_caches=k_caches,
            ctx=ctx,
            **kwargs,
        )
        offset = self.visit(
            node.offset,
            ij_caches=ij_caches,
            k_caches=k_caches,
            ctx=ctx,
            **kwargs,
        )
        if node.name in ij_caches:
            return cuir.IJCacheAccess(
                name=ij_caches[node.name].name,
                offset=offset,
                dtype=node.dtype,
                data_index=data_index,
            )
        if node.name in k_caches:
            return cuir.KCacheAccess(
                name=k_caches[node.name].name,
                offset=offset,
                dtype=node.dtype,
                data_index=data_index,
            )
        ctx.accessed_fields.add(node.name)
        return cuir.FieldAccess(
            name=node.name,
            offset=offset,
            data_index=data_index,
            dtype=node.dtype,
        )

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> Union[cuir.ScalarAccess, cuir.FieldAccess]:
        if isinstance(symtable.get(node.name, None), oir.ScalarDecl):
            return cuir.FieldAccess(
                name=node.name, offset=common.CartesianOffset.zero(), dtype=node.dtype
            )
        return cuir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> cuir.AssignStmt:
        return cuir.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> cuir.MaskStmt:
        return cuir.MaskStmt(
            mask=self.visit(node.mask, **kwargs), body=self.visit(node.body, **kwargs)
        )

    def visit_While(self, node: oir.While, **kwargs: Any) -> cuir.While:
        return cuir.While(
            cond=self.visit(node.cond, **kwargs), body=self.visit(node.body, **kwargs)
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> cuir.Cast:
        return cuir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> cuir.NativeFuncCall:
        return cuir.NativeFuncCall(
            func=node.func, args=self.visit(node.args, **kwargs), dtype=node.dtype
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> cuir.TernaryOp:
        return cuir.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, **kwargs: Any
    ) -> cuir.HorizontalExecution:
        block_extents = kwargs["block_extents"][id(node)]
        extent = cuir.IJExtent(i=block_extents[0], j=block_extents[1])
        return cuir.HorizontalExecution(
            body=self.visit(node.body, **kwargs),
            declarations=self.visit(node.declarations),
            extent=extent,
        )

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> cuir.VerticalLoopSection:
        return cuir.VerticalLoopSection(
            start=node.interval.start,
            end=node.interval.end,
            horizontal_executions=self.visit(node.horizontal_executions, **kwargs),
        )

    def visit_VerticalLoop(
        self,
        node: oir.VerticalLoop,
        *,
        symtable: Dict[str, Any],
        ctx: "Context",
        **kwargs: Any,
    ) -> cuir.Kernel:
        assert not any(c.fill or c.flush for c in node.caches if isinstance(c, oir.KCache))
        ij_caches = {
            c.name: cuir.IJCacheDecl(name=ctx.new_symbol_name(c.name), dtype=symtable[c.name].dtype)
            for c in node.caches
            if isinstance(c, oir.IJCache)
        }
        k_caches = {
            c.name: cuir.KCacheDecl(name=ctx.new_symbol_name(c.name), dtype=symtable[c.name].dtype)
            for c in node.caches
            if isinstance(c, oir.KCache)
        }
        return cuir.Kernel(
            vertical_loops=[
                cuir.VerticalLoop(
                    loop_order=node.loop_order,
                    sections=self.visit(
                        node.sections,
                        ij_caches=ij_caches,
                        k_caches=k_caches,
                        symtable=symtable,
                        ctx=ctx,
                        **kwargs,
                    ),
                    ij_caches=list(ij_caches.values()),
                    k_caches=list(k_caches.values()),
                )
            ],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> cuir.Program:
        block_extents = compute_horizontal_block_extents(node)
        ctx = self.Context(
            new_symbol_name=cast(SymbolNameCreator, symbol_name_creator(collect_symbol_names(node)))
        )
        kernels = self.visit(
            node.vertical_loops,
            ctx=ctx,
            block_extents=block_extents,
            **kwargs,
        )
        temporaries = [self.visit(d) for d in node.declarations if d.name in ctx.accessed_fields]
        return cuir.Program(
            name=node.name,
            params=self.visit(node.params),
            positionals=list(ctx.positionals.values()),
            temporaries=temporaries,
            kernels=kernels,
        )

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

from typing import Any, Callable, Dict, Set, Union

import eve
from gtc import common, oir
from gtc.cuir import cuir
from gtc.passes.oir_optimizations.utils import symbol_name_creator


class OIRToCUIR(eve.NodeTranslator):
    contexts = (eve.SymbolTableTrait.symtable_merger,)

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> cuir.Literal:
        return cuir.Literal(value=node.value, dtype=node.dtype)

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.FieldDecl(
            name=node.name, dtype=node.dtype, dimensions=node.dimensions, data_dims=node.data_dims
        )

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> cuir.UnaryOp:
        return cuir.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs), dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> cuir.BinaryOp:
        return cuir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            dtype=node.dtype,
        )

    def visit_Temporary(self, node: oir.Temporary, **kwargs: Any) -> cuir.Temporary:
        return cuir.Temporary(name=node.name, dtype=node.dtype)

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        ij_caches: Dict[str, cuir.IJCacheDecl],
        k_caches: Dict[str, cuir.KCacheDecl],
        accessed_fields: Set[str],
        **kwargs: Any,
    ) -> Union[cuir.FieldAccess, cuir.IJCacheAccess, cuir.KCacheAccess]:
        in_horizontal_mask = kwargs.get("in_horizontal_mask", False)
        if node.name in ij_caches:
            return cuir.IJCacheAccess(
                name=ij_caches[node.name].name,
                offset=node.offset,
                dtype=node.dtype,
                in_horizontal_mask=in_horizontal_mask,
            )
        if node.name in k_caches:
            return cuir.KCacheAccess(
                name=k_caches[node.name].name, offset=node.offset, dtype=node.dtype
            )
        accessed_fields.add(node.name)
        return cuir.FieldAccess(
            name=node.name,
            offset=node.offset,
            data_index=node.data_index,
            dtype=node.dtype,
            in_horizontal_mask=in_horizontal_mask,
        )

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> cuir.ScalarAccess:
        if isinstance(symtable.get(node.name, None), oir.ScalarDecl):
            return cuir.FieldAccess(
                name=node.name, offset=common.CartesianOffset.zero(), dtype=node.dtype
            )
        return cuir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> cuir.AssignStmt:
        return cuir.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def _ref_from_axis_bound(self, axis_bound: common.AxisBound, *, axis: str) -> cuir.Expr:
        scalar_name = f"{axis}_size"
        int_type = common.DataType.INT32

        if axis_bound.level == common.LevelMarker.END:
            return cuir.BinaryOp(
                op=common.ArithmeticOperator.ADD,
                left=cuir.ScalarAccess(name=scalar_name, dtype=int_type),
                right=cuir.Literal(value=str(axis_bound.offset), dtype=int_type),
                dtype=int_type,
            )
        else:
            return cuir.Literal(value=str(axis_bound.offset), dtype=int_type)

    def _expr_from_horizontal_interval(
        self,
        interval: common.HorizontalInterval,
        *,
        axis: str,
        **kwargs: Any,
    ) -> cuir.Expr:
        scalar_name = f"{axis}_pos(0_c, 0_c, 0_c)"
        int_type = common.DataType.INT32

        if interval.is_single_index:
            return cuir.BinaryOp(
                op=common.ComparisonOperator.EQ,
                left=cuir.ScalarAccess(name=scalar_name, dtype=int_type),
                right=self._ref_from_axis_bound(interval.start, axis=axis),
                dtype=int_type,
            )
        else:
            if interval.start:
                start_expr = cuir.BinaryOp(
                    op=common.ComparisonOperator.GE,
                    left=cuir.ScalarAccess(
                        name=scalar_name,
                        dtype=int_type,
                    ),
                    right=self._ref_from_axis_bound(interval.start, axis=axis),
                    dtype=int_type,
                )
            else:
                start_expr = None

            if interval.end:
                end_expr = cuir.BinaryOp(
                    op=common.ComparisonOperator.LT,
                    left=cuir.ScalarAccess(
                        name=scalar_name,
                        dtype=int_type,
                    ),
                    right=self._ref_from_axis_bound(interval.end, axis=axis),
                    dtype=int_type,
                )
            else:
                end_expr = None

            if start_expr and end_expr:
                return cuir.BinaryOp(
                    op=common.LogicalOperator.AND, left=start_expr, right=end_expr, dtype=int_type
                )
            else:
                # Return the first non-None expr, or if all are None, then return None
                return next((expr for expr in (start_expr, end_expr) if expr is not None), None)

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs: Any) -> cuir.Expr:
        i_expr = self._expr_from_horizontal_interval(node.i, axis="i", **kwargs)
        j_expr = self._expr_from_horizontal_interval(node.j, axis="j", **kwargs)
        if i_expr and j_expr:
            return cuir.BinaryOp(
                op=common.LogicalOperator.AND,
                left=i_expr,
                right=j_expr,
                dtype=common.DataType.INT32,
            )
        else:
            true_value = cuir.Literal(value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL)
            # Return the first non-None expr, or if all are None, then return True
            return next((expr for expr in (i_expr, j_expr) if expr is not None), true_value)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> cuir.MaskStmt:
        return cuir.MaskStmt(
            mask=self.visit(node.mask, **kwargs),
            body=self.visit(
                node.body, in_horizontal_mask=isinstance(node.mask, oir.HorizontalMask), **kwargs
            ),
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
            dtype=node.dtype,
        )

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, **kwargs: Any
    ) -> cuir.HorizontalExecution:
        return cuir.HorizontalExecution(
            body=self.visit(node.body, **kwargs),
            declarations=self.visit(node.declarations),
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
        new_symbol_name: Callable[[str], str],
        **kwargs: Any,
    ) -> cuir.Kernel:
        assert not any(c.fill or c.flush for c in node.caches if isinstance(c, oir.KCache))
        has_horizontal_masks = list(node.iter_tree().if_isinstance(oir.HorizontalMask)) != []
        ij_caches = {
            c.name: cuir.IJCacheDecl(name=new_symbol_name(c.name), dtype=symtable[c.name].dtype)
            for c in node.caches
            if isinstance(c, oir.IJCache)
        }
        k_caches = {
            c.name: cuir.KCacheDecl(name=new_symbol_name(c.name), dtype=symtable[c.name].dtype)
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
                        has_horizontal_masks=has_horizontal_masks,
                        symtable=symtable,
                        **kwargs,
                    ),
                    ij_caches=list(ij_caches.values()),
                    k_caches=list(k_caches.values()),
                )
            ],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> cuir.Program:
        accessed_fields: Set[str] = set()
        kernels = self.visit(
            node.vertical_loops,
            new_symbol_name=symbol_name_creator(set(kwargs["symtable"])),
            accessed_fields=accessed_fields,
            **kwargs,
        )
        temporaries = [self.visit(d) for d in node.declarations if d.name in accessed_fields]
        return cuir.Program(
            name=node.name,
            params=self.visit(node.params),
            temporaries=temporaries,
            kernels=kernels,
        )

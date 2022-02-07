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

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from eve.concepts import BaseNode
from eve.traits import SymbolTableTrait
from eve.visitors import NodeTranslator
from gtc import common, oir, utils
from gtc.passes.oir_optimizations.utils import StencilExtentComputer

from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    contexts = (SymbolTableTrait.symtable_merger,)

    # --- Decls ---
    def visit_FieldDecl(
        self, node: oir.FieldDecl, *, extents: StencilExtentComputer.Context, **kwargs: Any
    ) -> npir.FieldDecl:
        extent = cast(npir.HorizontalExtent, extents.fields.get(node.name, ((0, 0), (0, 0))))
        return npir.FieldDecl(
            name=node.name,
            dtype=node.dtype,
            dimensions=node.dimensions,
            data_dims=node.data_dims,
            extent=extent,
        )

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> npir.ScalarDecl:
        return npir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> npir.ScalarDecl:
        return npir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_Temporary(
        self, node: oir.Temporary, *, extents: StencilExtentComputer.Context, **kwargs: Any
    ) -> npir.TemporaryDecl:
        temp_extent = extents.fields[node.name]
        offset = tuple([-ext[0] for ext in temp_extent])
        assert all(off >= 0 for off in offset)
        boundary = tuple([ext[1] - ext[0] for ext in temp_extent])
        return npir.TemporaryDecl(
            name=node.name,
            dtype=node.dtype,
            data_dims=node.data_dims,
            offset=offset,
            boundary=boundary,
        )

    # --- Expressions ---
    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> npir.ScalarLiteral:
        assert node.kind == common.ExprKind.SCALAR
        return npir.ScalarLiteral(value=node.value, dtype=node.dtype, kind=node.kind)

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, *, symtable: Dict[str, oir.Decl], **kwargs: Any
    ) -> Union[npir.ParamAccess, npir.LocalScalarAccess]:
        assert node.kind == common.ExprKind.SCALAR
        if isinstance(symtable[node.name], oir.LocalScalar):
            return npir.LocalScalarAccess(name=node.name)
        else:
            return npir.ParamAccess(name=node.name)

    def visit_CartesianOffset(
        self, node: common.CartesianOffset, **kwargs: Any
    ) -> Tuple[int, int, int]:
        return node.i, node.j, node.k

    def visit_VariableKOffset(
        self, node: oir.VariableKOffset, **kwargs: Any
    ) -> Tuple[int, int, BaseNode]:
        return 0, 0, npir.VarKOffset(k=self.visit(node.k, **kwargs))

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> npir.FieldSlice:
        i_offset, j_offset, k_offset = self.visit(node.offset, **kwargs)
        data_index = [self.visit(index, **kwargs) for index in node.data_index]
        return npir.FieldSlice(
            name=node.name,
            i_offset=i_offset,
            j_offset=j_offset,
            k_offset=k_offset,
            data_index=data_index,
            dtype=node.dtype,
        )

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> npir.VectorUnaryOp:
        return npir.VectorUnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_BinaryOp(
        self, node: oir.BinaryOp, **kwargs: Any
    ) -> Union[npir.VectorArithmetic, npir.VectorLogic]:
        args = dict(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
        )
        if isinstance(node.op, common.LogicalOperator):
            return npir.VectorLogic(**args)
        else:
            return npir.VectorArithmetic(**args)

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> npir.VectorTernaryOp:
        return npir.VectorTernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> Union[npir.VectorCast, npir.ScalarCast]:
        expr = self.visit(node.expr, **kwargs)
        args = dict(dtype=node.dtype, expr=expr)
        return (
            npir.VectorCast(**args)
            if expr.kind == common.ExprKind.FIELD
            else npir.ScalarCast(**args)
        )

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> npir.NativeFuncCall:
        return npir.NativeFuncCall(
            func=self.visit(node.func, **kwargs), args=self.visit(node.args, **kwargs)
        )

    # --- Statements ---
    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        mask: Optional[npir.Expr] = None,
        **kwargs: Any,
    ) -> List[npir.VectorAssign]:
        mask_expr = self.visit(node.mask, **kwargs)
        if mask:
            mask_expr = npir.VectorLogic(op=common.LogicalOperator.AND, left=mask, right=mask_expr)
        return self.visit(node.body, mask=mask_expr, **kwargs)

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        mask: Optional[npir.Expr] = None,
        **kwargs: Any,
    ) -> npir.VectorAssign:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)
        if right.kind == common.ExprKind.SCALAR:
            right = npir.Broadcast(expr=right, dims=3)
        return npir.VectorAssign(left=left, right=right, mask=mask)

    # --- Control Flow ---
    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        extents: Optional[StencilExtentComputer.Context] = None,
        **kwargs: Any,
    ) -> npir.HorizontalBlock:
        stmts = utils.flatten_list(self.visit(node.body, **kwargs))
        if extents:
            extent = extents.blocks[id(node)]
        else:
            extent = ((0, 0), (0, 0))
        return npir.HorizontalBlock(
            declarations=self.visit(node.declarations, **kwargs), body=stmts, extent=extent
        )

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, *, loop_order: common.LoopOrder, **kwargs: Any
    ) -> npir.VerticalPass:
        return npir.VerticalPass(
            body=self.visit(node.horizontal_executions, **kwargs),
            lower=node.interval.start,
            upper=node.interval.end,
            direction=loop_order,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> List[npir.VerticalPass]:
        return self.visit(node.sections, loop_order=node.loop_order, **kwargs)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> npir.Computation:
        extents = StencilExtentComputer().visit(node)

        arguments = [decl.name for decl in node.params]
        param_decls = [
            self.visit(decl, **kwargs) for decl in node.params if isinstance(decl, oir.ScalarDecl)
        ]
        api_field_decls = [
            self.visit(decl, extents=extents)
            for decl in node.params
            if isinstance(decl, oir.FieldDecl)
        ]
        temp_decls = [self.visit(decl, extents=extents, **kwargs) for decl in node.declarations]

        vertical_passes = utils.flatten_list(
            self.visit(node.vertical_loops, extents=extents, **kwargs)
        )

        return npir.Computation(
            arguments=arguments,
            api_field_decls=api_field_decls,
            param_decls=param_decls,
            temp_decls=temp_decls,
            vertical_passes=vertical_passes,
        )

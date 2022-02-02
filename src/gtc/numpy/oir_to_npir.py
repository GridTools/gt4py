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

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from eve.concepts import BaseNode
from eve.traits import SymbolTableTrait
from eve.visitors import NodeTranslator
from gt4py.definitions import Extent
from gtc import common, oir
from gtc.passes.oir_optimizations.utils import StencilExtentComputer

from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    contexts = (SymbolTableTrait.symtable_merger,)

    @dataclass
    class HorizontalExecutionContext:
        stmts: List[npir.VectorAssign] = field(default_factory=list)

        def add_stmt(self, stmt: npir.VectorAssign):
            self.stmts.append(stmt)

    @dataclass
    class StencilContext:
        vpasses: List[npir.VerticalPass] = field(default_factory=list)

        def add_pass(self, vpass: npir.VerticalPass):
            self.vpasses.append(vpass)

    # --- Decls ---
    def visit_FieldDecl(
        self, node: oir.FieldDecl, *, extents: StencilExtentComputer.Context, **kwargs: Any
    ) -> npir.FieldDecl:
        return npir.FieldDecl(
            name=node.name,
            dtype=node.dtype,
            dimensions=node.dimensions,
            data_dims=node.data_dims,
            extent=extents.fields.get(node.name, Extent.zeros(ndims=2)),
        )

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> npir.ScalarDecl:
        return npir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> npir.LocalDecl:
        return npir.LocalDecl(name=node.name, dtype=node.dtype)

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
        cls = (
            npir.LocalScalarAccess
            if isinstance(symtable[node.name], oir.LocalScalar)
            else npir.ParamAccess
        )
        return cls(name=node.name)

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
        cls = (
            npir.VectorLogic
            if isinstance(node.op, common.LogicalOperator)
            else npir.VectorArithmetic
        )
        return cls(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> npir.VectorTernaryOp:
        return npir.VectorTernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> Union[npir.VectorCast, npir.ScalarCast]:
        expr = self.visit(node.expr, **kwargs)
        cls = npir.VectorCast if expr.kind == common.ExprKind.FIELD else npir.ScalarCast
        return cls(dtype=node.dtype, expr=expr)

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> npir.NativeFuncCall:
        return npir.NativeFuncCall(
            func=self.visit(node.func, **kwargs), args=self.visit(node.args, **kwargs)
        )

    # --- Statements ---
    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        ctx: "HorizontalExecutionContext",
        mask: Optional[npir.Expr] = None,
        **kwargs: Any,
    ) -> None:
        mask_expr = self.visit(node.mask, **kwargs)
        if mask:
            mask_expr = npir.VectorLogic(op=common.LogicalOperator.AND, left=mask, right=mask_expr)

        for stmt in node.body:
            self.visit(stmt, mask=mask_expr, ctx=ctx, **kwargs)

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        ctx: "HorizontalExecutionContext",
        mask: Optional[npir.Expr] = None,
        **kwargs: Any,
    ) -> None:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)
        if right.kind == common.ExprKind.SCALAR:
            right = npir.Broadcast(expr=right, dims=3)
        assign = npir.VectorAssign(left=left, right=right, mask=mask)
        ctx.add_stmt(assign)

    # --- Control Flow ---
    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        extents: StencilExtentComputer.Context,
        **kwargs: Any,
    ) -> npir.HorizontalBlock:
        ctx = self.HorizontalExecutionContext()
        self.visit(node.body, ctx=ctx, **kwargs)
        extent = extents.blocks[id(node)]
        return npir.HorizontalBlock(
            declarations=self.visit(node.declarations, **kwargs), body=ctx.stmts, extent=extent
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

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, ctx: "StencilContext", **kwargs: Any
    ) -> None:
        for section in node.sections:
            ctx.add_pass(self.visit(section, loop_order=node.loop_order, **kwargs))

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

        ctx = self.StencilContext()
        self.visit(node.vertical_loops, ctx=ctx, extents=extents, **kwargs)

        return npir.Computation(
            arguments=arguments,
            api_field_decls=api_field_decls,
            param_decls=param_decls,
            temp_decls=temp_decls,
            vertical_passes=ctx.vpasses,
        )

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

import typing
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from eve.concepts import BaseNode
from eve.iterators import TreeIterationItem
from eve.traits import SymbolTableTrait
from eve.utils import as_xiter
from eve.visitors import NodeTranslator
from gt4py.definitions import Extent
from gtc import common, oir, utils
from gtc.passes.oir_masks import compute_relative_mask
from gtc.passes.oir_optimizations.utils import compute_extents

from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    contexts = (SymbolTableTrait.symtable_merger,)

    # --- Decls ---
    def visit_FieldDecl(
        self, node: oir.FieldDecl, *, field_extents: Dict[str, Extent], **kwargs: Any
    ) -> npir.FieldDecl:
        extent = typing.cast(npir.HorizontalExtent, field_extents.get(node.name, ((0, 0), (0, 0))))
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
        self, node: oir.Temporary, *, field_extents: Dict[str, Extent], **kwargs: Any
    ) -> npir.TemporaryDecl:
        temp_extent = field_extents[node.name]
        offset = [-ext[0] for ext in temp_extent]
        assert all(off >= 0 for off in offset)
        padding = [ext[1] - ext[0] for ext in temp_extent]
        return npir.TemporaryDecl(
            name=node.name,
            dtype=node.dtype,
            data_dims=node.data_dims,
            offset=offset,
            padding=padding,
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

    def visit_HorizontalMask(self, node: oir.HorizontalMask, **kwargs: Any) -> npir.Expr:
        return npir.ScalarLiteral(value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL)

    # --- Statements ---
    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        extent: Extent,
        mask: Optional[npir.Expr] = None,
        **kwargs: Any,
    ) -> List[npir.VectorAssign]:
        @as_xiter
        def _iter_tree(mask: oir.Expr) -> Generator[TreeIterationItem, None, None]:
            local_assigns = kwargs["local_assigns"]
            sub_exprs = (
                local_assigns[name]
                for name in mask.iter_tree().if_isinstance(oir.ScalarAccess).getattr("name")
                if name in local_assigns
            )
            for elem in (mask, *sub_exprs):
                yield from elem.iter_tree()

        try:
            absolute_mask = next(iter(_iter_tree(node.mask).if_isinstance(oir.HorizontalMask)))
            # Lower to npir.HorizontalMask relative to horizontal execution extent
            relative_mask = compute_relative_mask(extent, absolute_mask)
            horizontal_mask: Optional[oir.HorizontalMask] = npir.HorizontalMask(
                i=relative_mask.i, j=relative_mask.j
            )
        except StopIteration:
            horizontal_mask = None

        mask_expr = self.visit(node.mask, **kwargs)
        if mask:
            mask_expr = npir.VectorLogic(op=common.LogicalOperator.AND, left=mask, right=mask_expr)

        # --- Quick optimization: remove the mask entirely if mask_expr is only a horizontal mask. ---
        mask_is_hmask = (
            isinstance(mask_expr, npir.ScalarLiteral)
            and mask_expr.value == common.BuiltInLiteral.TRUE
        )
        mask_is_indirect_hmask = isinstance(mask_expr, npir.LocalScalarAccess) and isinstance(
            kwargs["local_assigns"].get(mask_expr.name, None), oir.HorizontalMask
        )
        if mask_is_hmask or mask_is_indirect_hmask:
            mask_expr = None
        # ---

        return self.visit(node.body, horizontal_mask=horizontal_mask, mask=mask_expr, **kwargs)

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        mask: Optional[npir.Expr] = None,
        horizontal_mask: Optional[npir.HorizontalMask] = None,
        **kwargs: Any,
    ) -> npir.VectorAssign:
        kwargs["local_assigns"][node.left.name] = node.right
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)
        if right.kind == common.ExprKind.SCALAR:
            right = npir.Broadcast(expr=right, dims=3)
        return npir.VectorAssign(left=left, right=right, mask=mask, horizontal_mask=horizontal_mask)

    # --- Control Flow ---
    def visit_While(
        self, node: oir.While, *, mask: Optional[npir.Expr] = None, **kwargs: Any
    ) -> npir.While:
        cond = self.visit(node.cond, mask=mask, **kwargs)
        if mask:
            mask = npir.VectorLogic(op=common.LogicalOperator.AND, left=mask, right=cond)
        else:
            mask = cond
        return npir.While(cond=cond, body=self.visit(node.body, mask=mask, **kwargs))

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        block_extents: Optional[Dict[int, Extent]] = None,
        **kwargs: Any,
    ) -> npir.HorizontalBlock:
        if block_extents:
            extent = block_extents[id(node)]
        else:
            extent = ((0, 0), (0, 0))
        stmts = utils.flatten_list(self.visit(node.body, local_assigns={}, extent=extent, **kwargs))
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
        field_extents, block_extents = compute_extents(node)

        arguments = [decl.name for decl in node.params]
        param_decls = [
            self.visit(decl, **kwargs) for decl in node.params if isinstance(decl, oir.ScalarDecl)
        ]
        api_field_decls = [
            self.visit(decl, field_extents=field_extents)
            for decl in node.params
            if isinstance(decl, oir.FieldDecl)
        ]
        temp_decls = [
            self.visit(decl, field_extents=field_extents, **kwargs) for decl in node.declarations
        ]

        vertical_passes = utils.flatten_list(
            self.visit(node.vertical_loops, block_extents=block_extents, **kwargs)
        )

        return npir.Computation(
            arguments=arguments,
            api_field_decls=api_field_decls,
            param_decls=param_decls,
            temp_decls=temp_decls,
            vertical_passes=vertical_passes,
        )

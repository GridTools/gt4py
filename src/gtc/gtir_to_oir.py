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

from typing import Any, List

from eve import NodeTranslator
from gtc import gtir, oir
from gtc.common import CartesianOffset, DataType, LogicalOperator, UnaryOperator


class GTIRToOIR(NodeTranslator):
    # --- Exprs ---
    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs: Any) -> oir.FieldAccess:
        return oir.FieldAccess(
            name=node.name,
            offset=node.offset,
            data_index=self.visit(node.data_index),
            dtype=node.dtype,
        )

    def visit_ScalarAccess(self, node: gtir.ScalarAccess, **kwargs: Any) -> oir.ScalarAccess:
        return oir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_Literal(self, node: gtir.Literal, **kwargs: Any) -> oir.Literal:
        return oir.Literal(value=self.visit(node.value), dtype=node.dtype, kind=node.kind)

    def visit_UnaryOp(self, node: gtir.UnaryOp, **kwargs: Any) -> oir.UnaryOp:
        return oir.UnaryOp(op=node.op, expr=self.visit(node.expr))

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs: Any) -> oir.BinaryOp:
        return oir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs: Any) -> oir.TernaryOp:
        return oir.TernaryOp(
            cond=self.visit(node.cond),
            true_expr=self.visit(node.true_expr),
            false_expr=self.visit(node.false_expr),
        )

    def visit_Cast(self, node: gtir.Cast, **kwargs: Any) -> oir.Cast:
        return oir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_FieldDecl(self, node: gtir.FieldDecl, **kwargs: Any) -> oir.FieldDecl:
        return oir.FieldDecl(
            name=node.name, dtype=node.dtype, dimensions=node.dimensions, data_dims=node.data_dims
        )

    def visit_ScalarDecl(self, node: gtir.ScalarDecl, **kwargs: Any) -> oir.ScalarDecl:
        return oir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs: Any) -> oir.NativeFuncCall:
        return oir.NativeFuncCall(
            func=node.func, args=self.visit(node.args), dtype=node.dtype, kind=node.kind
        )

    # --- Stmts ---
    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        level: int,
        mask: oir.Expr = None,
        **kwargs: Any,
    ) -> oir.AssignStmt:
        stmt = oir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))
        if mask is not None:
            # Wrap inside MaskStmt
            stmt = oir.MaskStmt(body=[stmt], mask=mask)
        return stmt

    def visit_For(
        self, node: gtir.For, *, level: int, scalars: List[oir.ScalarDecl], **kwargs: Any
    ) -> oir.For:
        body_stmts = []
        for stmt in node.body:
            stmt_or_stmts = self.visit(stmt, level=level + 1, scalars=scalars, **kwargs)
            if isinstance(stmt_or_stmts, oir.Stmt):
                body_stmts.append(stmt_or_stmts)
            else:
                body_stmts.extend(stmt_or_stmts)
        stmt = oir.For(
            target_name=node.target.name,
            start=self.visit(node.start),
            end=self.visit(node.end),
            inc=node.inc,
            body=body_stmts,
        )
        scalars.append(oir.LocalScalar(name=node.target.name, dtype=node.target.dtype))
        return stmt

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, level: int, mask: oir.Expr = None, **kwargs: Any
    ) -> List[oir.Stmt]:
        mask_field_decl = oir.Temporary(
            name=f"mask_{id(node)}", dtype=DataType.BOOL, dimensions=(True, True, True)
        )
        stmts = [
            oir.AssignStmt(
                left=oir.FieldAccess(
                    name=mask_field_decl.name,
                    offset=CartesianOffset.zero(),
                    dtype=DataType.BOOL,
                ),
                right=self.visit(node.cond),
            )
        ]

        current_mask = oir.FieldAccess(
            name=mask_field_decl.name, offset=CartesianOffset.zero(), dtype=mask_field_decl.dtype
        )
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
        stmts.extend(
            self.visit(node.true_branch.body, level=level + 1, mask=combined_mask, **kwargs)
        )

        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            stmts.extend(
                self.visit(
                    node.false_branch.body,
                    level=level + 1,
                    mask=combined_mask,
                    **kwargs,
                )
            )

        return stmts

    # For now we represent ScalarIf (and FieldIf) both as masks on the HorizontalExecution.
    # This is not meant to be set in stone...
    def visit_ScalarIfStmt(
        self, node: gtir.ScalarIfStmt, *, level: int, mask: oir.Expr = None, **kwargs: Any
    ) -> List[oir.Stmt]:
        current_mask = self.visit(node.cond)
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=current_mask)

        stmts = self.visit(node.true_branch.body, mask=combined_mask, level=level + 1, **kwargs)
        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            stmts.extend(
                self.visit(node.false_branch.body, mask=combined_mask, level=level + 1, **kwargs)
            )

        return stmts

    def visit_Interval(self, node: gtir.Interval, **kwargs: Any) -> oir.Interval:
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    # --- Control flow ---
    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        horiz_execs: List[oir.HorizontalExecution] = []
        for stmt in node.body:
            scalars: List[oir.ScalarDecl] = []
            stmt_or_stmts = self.visit(stmt, level=0, scalars=scalars, **kwargs)
            body = [stmt_or_stmts] if isinstance(stmt_or_stmts, oir.Stmt) else stmt_or_stmts
            horiz_execs.append(oir.HorizontalExecution(body=body, declarations=scalars))

        kwargs["temps"] += [
            oir.Temporary(name=temp.name, dtype=temp.dtype, dimensions=temp.dimensions)
            for temp in node.temporaries
        ]

        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=[
                oir.VerticalLoopSection(
                    interval=self.visit(node.interval, **kwargs),
                    horizontal_executions=horiz_execs,
                )
            ],
            caches=[],
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> oir.Stencil:
        temps: List[oir.Temporary] = []
        vertical_loops = self.visit(node.vertical_loops, temps=temps)
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params),
            vertical_loops=vertical_loops,
            declarations=temps,
        )

    def visit_AxisPosition(self, node: gtir.AxisPosition) -> oir.AxisPosition:
        return oir.AxisPosition(axis=node.axis)

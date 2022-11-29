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

from dataclasses import dataclass, field
from typing import Any, List, Set, Union

import eve
from gtc import common, gtir, oir, utils
from gtc.common import CartesianOffset, DataType, LogicalOperator, UnaryOperator
from gtc.passes.oir_optimizations.utils import compute_fields_extents


def validate_stencil_memory_accesses(node: oir.Stencil) -> oir.Stencil:
    """Check that no memory races occur in GridTools backends.

    Since this is required for GridTools backends, it's imposed on all backends
    at the OIR level. This is similar to the check at the gtir level for read-with-offset
    and writes, but more complete because it involves extent analysis, so it catches
    indirect read-with-offset through temporaries.

    """

    def _writes(node: oir.Stencil) -> Set[str]:
        result = set()
        for left in node.walk_values().if_isinstance(oir.AssignStmt).getattr("left"):
            result |= left.walk_values().if_isinstance(oir.FieldAccess).getattr("name").to_set()
        return result

    field_names = {decl.name for decl in node.params if isinstance(decl, oir.FieldDecl)}
    write_fields = _writes(node) & field_names

    field_extents = compute_fields_extents(node)

    names: Set[str] = set()
    for name in write_fields:
        if not field_extents[name].is_zero:
            names.add(name)

    if names:
        raise ValueError(f"Found non-zero read extent on written fields: {', '.join(names)}")

    return node


class GTIRToOIR(eve.NodeTranslator):
    @dataclass
    class Context:
        local_scalars: List[oir.ScalarDecl] = field(default_factory=list)
        temp_fields: List[oir.FieldDecl] = field(default_factory=list)

        def reset_local_scalars(self):
            self.local_scalars = []

    # --- Exprs ---
    def visit_FieldAccess(self, node: gtir.FieldAccess) -> oir.FieldAccess:
        return oir.FieldAccess(
            name=node.name,
            offset=self.visit(node.offset),
            data_index=self.visit(node.data_index),
            dtype=node.dtype,
            loc=node.loc,
        )

    def visit_VariableKOffset(self, node: gtir.VariableKOffset) -> oir.VariableKOffset:
        return oir.VariableKOffset(k=self.visit(node.k))

    def visit_ScalarAccess(self, node: gtir.ScalarAccess) -> oir.ScalarAccess:
        return oir.ScalarAccess(name=node.name, dtype=node.dtype, loc=node.loc)

    def visit_Literal(self, node: gtir.Literal) -> oir.Literal:
        return oir.Literal(
            value=self.visit(node.value), dtype=node.dtype, kind=node.kind, loc=node.loc
        )

    def visit_UnaryOp(self, node: gtir.UnaryOp) -> oir.UnaryOp:
        return oir.UnaryOp(op=node.op, expr=self.visit(node.expr), dtype=node.dtype, loc=node.loc)

    def visit_BinaryOp(self, node: gtir.BinaryOp) -> oir.BinaryOp:
        return oir.BinaryOp(
            op=node.op, left=self.visit(node.left), right=self.visit(node.right), loc=node.loc
        )

    def visit_TernaryOp(self, node: gtir.TernaryOp) -> oir.TernaryOp:
        return oir.TernaryOp(
            cond=self.visit(node.cond),
            true_expr=self.visit(node.true_expr),
            false_expr=self.visit(node.false_expr),
            loc=node.loc,
        )

    def visit_Cast(self, node: gtir.Cast) -> oir.Cast:
        return oir.Cast(dtype=node.dtype, expr=self.visit(node.expr), loc=node.loc)

    def visit_FieldDecl(self, node: gtir.FieldDecl) -> oir.FieldDecl:
        return oir.FieldDecl(
            name=node.name,
            dtype=node.dtype,
            dimensions=node.dimensions,
            data_dims=node.data_dims,
            loc=node.loc,
        )

    def visit_ScalarDecl(self, node: gtir.ScalarDecl) -> oir.ScalarDecl:
        return oir.ScalarDecl(name=node.name, dtype=node.dtype, loc=node.loc)

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall) -> oir.NativeFuncCall:
        return oir.NativeFuncCall(
            func=node.func,
            args=self.visit(node.args),
            dtype=node.dtype,
            kind=node.kind,
            loc=node.loc,
        )

    # --- Stmts ---
    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, **kwargs: Any
    ) -> Union[oir.AssignStmt, oir.MaskStmt]:
        stmt: Union[oir.AssignStmt, oir.MaskStmt] = oir.AssignStmt(
            left=self.visit(node.left), right=self.visit(node.right)
        )
        if mask is not None:
            # Wrap inside MaskStmt
            stmt = oir.MaskStmt(body=[stmt], mask=mask, loc=node.loc)
        return stmt

    def visit_HorizontalRestriction(
        self, node: gtir.HorizontalRestriction, **kwargs: Any
    ) -> oir.HorizontalRestriction:
        body_stmts = []
        for stmt in node.body:
            stmt_or_stmts = self.visit(stmt, **kwargs)
            stmts = utils.flatten_list(
                [stmt_or_stmts] if isinstance(stmt_or_stmts, oir.Stmt) else stmt_or_stmts
            )
            body_stmts.extend(stmts)

        return oir.HorizontalRestriction(mask=node.mask, body=body_stmts)

    def visit_While(
        self, node: gtir.While, *, mask: oir.Expr = None, **kwargs: Any
    ) -> Union[oir.While, oir.MaskStmt]:
        body_stmts: List[oir.Stmt] = []
        for st in node.body:
            st_or_sts = self.visit(st, **kwargs)
            sts = utils.flatten_list([st_or_sts] if isinstance(st_or_sts, oir.Stmt) else st_or_sts)
            body_stmts.extend(sts)

        cond: oir.Expr = self.visit(node.cond)
        if mask:
            cond = oir.BinaryOp(op=common.LogicalOperator.AND, left=mask, right=cond)
        stmt: Union[oir.While, oir.MaskStmt] = oir.While(cond=cond, body=body_stmts, loc=node.loc)
        if mask is not None:
            stmt = oir.MaskStmt(body=[stmt], mask=mask, loc=node.loc)
        return stmt

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> List[oir.Stmt]:
        mask_field_decl = oir.Temporary(
            name=f"mask_{id(node)}", dtype=DataType.BOOL, dimensions=(True, True, True)
        )
        ctx.temp_fields.append(mask_field_decl)
        stmts: List[oir.Stmt] = [
            oir.AssignStmt(
                left=oir.FieldAccess(
                    name=mask_field_decl.name,
                    offset=CartesianOffset.zero(),
                    dtype=DataType.BOOL,
                    loc=node.loc,
                ),
                right=self.visit(node.cond),
            )
        ]

        current_mask = oir.FieldAccess(
            name=mask_field_decl.name,
            offset=CartesianOffset.zero(),
            dtype=mask_field_decl.dtype,
            loc=node.loc,
        )

        combined_mask: oir.Expr = current_mask
        if mask:
            combined_mask = oir.BinaryOp(
                op=LogicalOperator.AND, left=mask, right=combined_mask, loc=node.loc
            )
        stmts.extend(self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx, **kwargs))

        if node.false_branch:
            combined_mask_not: oir.Expr = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask_not = oir.BinaryOp(
                    op=LogicalOperator.AND, left=mask, right=combined_mask_not, loc=node.loc
                )
            stmts.extend(
                self.visit(node.false_branch.body, mask=combined_mask_not, ctx=ctx, **kwargs)
            )

        return stmts

    # For now we represent ScalarIf (and FieldIf) both as masks on the HorizontalExecution.
    # This is not meant to be set in stone...
    def visit_ScalarIfStmt(
        self, node: gtir.ScalarIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> List[oir.Stmt]:
        current_mask = self.visit(node.cond)
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(
                op=LogicalOperator.AND, left=mask, right=current_mask, loc=node.loc
            )

        stmts = self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx, **kwargs)
        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask, loc=node.loc)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            stmts.extend(self.visit(node.false_branch.body, mask=combined_mask, ctx=ctx, **kwargs))

        return stmts

    # --- Misc ---
    def visit_Interval(self, node: gtir.Interval) -> oir.Interval:
        return oir.Interval(start=self.visit(node.start), end=self.visit(node.end), loc=node.loc)

    # --- Control flow ---
    def visit_VerticalLoop(self, node: gtir.VerticalLoop, *, ctx: Context) -> oir.VerticalLoop:
        horiz_execs: List[oir.HorizontalExecution] = []
        for stmt in node.body:
            ctx.reset_local_scalars()
            ret = self.visit(stmt, ctx=ctx)
            stmts = utils.flatten_list([ret] if isinstance(ret, oir.Stmt) else ret)
            horiz_execs.append(oir.HorizontalExecution(body=stmts, declarations=ctx.local_scalars))

        ctx.temp_fields += [
            oir.Temporary(
                name=temp.name,
                dtype=temp.dtype,
                dimensions=temp.dimensions,
                data_dims=temp.data_dims,
            )
            for temp in node.temporaries
        ]

        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=[
                oir.VerticalLoopSection(
                    interval=self.visit(node.interval),
                    horizontal_executions=horiz_execs,
                    loc=node.loc,
                )
            ],
        )

    def visit_Stencil(self, node: gtir.Stencil) -> oir.Stencil:
        ctx = self.Context()
        vertical_loops = self.visit(node.vertical_loops, ctx=ctx)
        return validate_stencil_memory_accesses(
            oir.Stencil(
                name=node.name,
                params=self.visit(node.params),
                vertical_loops=vertical_loops,
                declarations=ctx.temp_fields,
                loc=node.loc,
            )
        )

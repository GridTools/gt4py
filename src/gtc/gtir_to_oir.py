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

from dataclasses import dataclass, field
from typing import Any, Dict, List

from eve import NodeTranslator
from eve.iterators import TraversalOrder, iter_tree
from gtc import gtir, oir
from gtc.common import CartesianOffset, DataType, ExprKind, LogicalOperator, UnaryOperator


def _create_mask(ctx: "GTIRToOIR.Context", name: str, cond: oir.Expr) -> oir.Temporary:
    mask_field_decl = oir.Temporary(name=name, dtype=DataType.BOOL, dimensions=(True, True, True))
    ctx.add_decl(mask_field_decl)

    fill_mask_field = oir.HorizontalExecution(
        body=[
            oir.AssignStmt(
                left=oir.FieldAccess(
                    name=mask_field_decl.name,
                    offset=CartesianOffset.zero(),
                    dtype=mask_field_decl.dtype,
                ),
                right=cond,
            )
        ],
        declarations=[],
    )
    ctx.add_horizontal_execution(fill_mask_field)
    return mask_field_decl


class OIRIterationSpaceTranslator(NodeTranslator):
    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        iteration_spaces: Dict[int, oir.CartesianIterationSpace],
    ) -> oir.HorizontalExecution:
        assert id(node) in iteration_spaces
        return oir.HorizontalExecution(
            body=node.body,
            mask=node.mask,
            declarations=node.declarations,
            iteration_space=iteration_spaces[id(node)],
        )


def oir_iteration_space_computation(stencil: oir.Stencil) -> oir.Stencil:
    iteration_spaces = dict()

    offsets: Dict[str, List[CartesianOffset]] = dict()
    outputs = set()
    access_spaces: Dict[str, oir.CartesianIterationSpace] = dict()
    # reversed pre_order traversal is post-order but the last nodes per node come first.
    # this is not possible with visitors unless a (reverseed) visit is implemented for every node
    for node in reversed(list(iter_tree(stencil, traversal_order=TraversalOrder.PRE_ORDER))):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.AssignStmt):
            if node.left.kind == ExprKind.FIELD:
                outputs.add(node.left.name)
        elif isinstance(node, oir.HorizontalExecution):
            iteration_spaces[id(node)] = oir.CartesianIterationSpace.domain()
            for name in outputs:
                access_space = access_spaces.get(name, oir.CartesianIterationSpace.domain())
                iteration_spaces[id(node)] = iteration_spaces[id(node)] | access_space
            for name in offsets:
                access_space = oir.CartesianIterationSpace.domain()
                for offset in offsets[name]:
                    access_space = access_space | oir.CartesianIterationSpace.from_offset(offset)
                accumulated_extent = iteration_spaces[id(node)].compose(access_space)
                access_spaces[name] = (
                    access_spaces.get(name, oir.CartesianIterationSpace.domain())
                    | accumulated_extent
                )

            offsets = dict()
            outputs = set()

    return OIRIterationSpaceTranslator().visit(stencil, iteration_spaces=iteration_spaces)


def oir_field_boundary_computation(stencil: oir.Stencil) -> Dict[str, oir.CartesianIterationSpace]:
    offsets: Dict[str, List[CartesianOffset]] = dict()
    access_spaces: Dict[str, oir.CartesianIterationSpace] = dict()
    for node in iter_tree(stencil, traversal_order=TraversalOrder.POST_ORDER):
        if isinstance(node, oir.FieldAccess):
            if node.name not in offsets:
                offsets[node.name] = list()
            offsets[node.name].append(node.offset)
        elif isinstance(node, oir.HorizontalExecution):
            if node.iteration_space is not None:
                for name in offsets:
                    access_space = oir.CartesianIterationSpace.domain()
                    for offset in offsets[name]:
                        access_space = access_space | oir.CartesianIterationSpace.from_offset(
                            offset
                        )
                    access_spaces[name] = access_spaces.get(
                        name, oir.CartesianIterationSpace.domain()
                    ) | (node.iteration_space.compose(access_space))

            offsets = dict()

    return access_spaces


class GTIRToOIR(NodeTranslator):
    @dataclass
    class Context:
        """
        Context for Stmts.

        `Stmt` nodes create `Temporary` nodes and `HorizontalExecution` nodes.
        All visit()-methods for `Stmt` have no return value,
        they attach their result to the Context object.
        """

        decls: List = field(default_factory=list)
        horizontal_executions: List = field(default_factory=list)

        def add_decl(self, decl: oir.Decl) -> "GTIRToOIR.Context":
            self.decls.append(decl)
            return self

        def add_horizontal_execution(
            self, horizontal_execution: oir.HorizontalExecution
        ) -> "GTIRToOIR.Context":
            self.horizontal_executions.append(horizontal_execution)
            return self

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        body = [oir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))]
        if mask is not None:
            body = [oir.MaskStmt(body=body, mask=mask)]
        ctx.add_horizontal_execution(
            oir.HorizontalExecution(
                body=body,
                declarations=[],
            ),
        )

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs: Any) -> oir.FieldAccess:
        return oir.FieldAccess(name=node.name, offset=node.offset, dtype=node.dtype)

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
        return oir.FieldDecl(name=node.name, dtype=node.dtype, dimensions=node.dimensions)

    def visit_ScalarDecl(self, node: gtir.ScalarDecl, **kwargs: Any) -> oir.ScalarDecl:
        return oir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs: Any) -> oir.NativeFuncCall:
        return oir.NativeFuncCall(
            func=node.func, args=self.visit(node.args), dtype=node.dtype, kind=node.kind
        )

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        mask_field_decl = _create_mask(ctx, f"mask_{id(node)}", self.visit(node.cond))
        current_mask = oir.FieldAccess(
            name=mask_field_decl.name, offset=CartesianOffset.zero(), dtype=mask_field_decl.dtype
        )
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
        self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx)

        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    # For now we represent ScalarIf (and FieldIf) both as masks on the HorizontalExecution.
    # This is not meant to be set in stone...
    def visit_ScalarIfStmt(
        self, node: gtir.ScalarIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs: Any
    ) -> None:
        current_mask = self.visit(node.cond)
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=current_mask)

        self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx)
        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=combined_mask)
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    def visit_Interval(self, node: gtir.Interval, **kwargs: Any) -> oir.Interval:
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    def visit_VerticalLoop(
        self, node: gtir.VerticalLoop, *, ctx: Context, **kwargs: Any
    ) -> oir.VerticalLoop:
        ctx.horizontal_executions.clear()
        self.visit(node.body, ctx=ctx)

        for temp in node.temporaries:
            ctx.add_decl(
                oir.Temporary(name=temp.name, dtype=temp.dtype, dimensions=temp.dimensions)
            )

        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=[
                oir.VerticalLoopSection(
                    interval=self.visit(node.interval, **kwargs),
                    horizontal_executions=ctx.horizontal_executions,
                )
            ],
            caches=[],
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> oir.Stencil:
        ctx = self.Context()
        return oir_iteration_space_computation(
            oir.Stencil(
                name=node.name,
                params=self.visit(node.params),
                vertical_loops=self.visit(node.vertical_loops, ctx=ctx),
                declarations=ctx.decls,
            )
        )

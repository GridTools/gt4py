from typing import List

from dataclasses import dataclass, field
from eve import NodeTranslator

from gt4py.gtc import gtir, oir
from gt4py.gtc.common import CartesianOffset, DataType, LogicalOperator, UnaryOperator


def _create_mask(ctx: "GTIRToOIR.Context", name: str, cond: oir.Expr) -> oir.Temporary:
    mask_field_decl = oir.Temporary(name=name, dtype=DataType.BOOL)
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
        ]
    )
    ctx.add_horizontal_execution(fill_mask_field)
    return mask_field_decl


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

        def add_decl(self, decl) -> "GTIRToOIR.Context":
            self.decls.append(decl)
            return self

        def add_horizontal_execution(self, horizontal_execution) -> "GTIRToOIR.Context":
            self.horizontal_executions.append(horizontal_execution)
            return self

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs
    ):
        tmp = oir.Temporary(name="tmp_" + node.left.name + "_" + node.id_, dtype=node.left.dtype)

        ctx.add_decl(tmp)

        ctx.add_horizontal_execution(
            oir.HorizontalExecution(
                body=[
                    oir.AssignStmt(
                        left=oir.FieldAccess(
                            name=tmp.name, offset=CartesianOffset.zero(), dtype=tmp.dtype
                        ),
                        right=self.visit(node.right),
                    )
                ],
                mask=mask,
            )
        )
        ctx.add_horizontal_execution(
            oir.HorizontalExecution(
                body=[
                    oir.AssignStmt(
                        left=self.visit(node.left),
                        right=oir.FieldAccess(
                            name=tmp.name, offset=CartesianOffset.zero(), dtype=tmp.dtype
                        ),
                    )
                ],
                mask=mask,
            ),
        )

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        return oir.FieldAccess(name=node.name, offset=node.offset, dtype=node.dtype)

    def visit_ScalarAccess(self, node: gtir.ScalarAccess, **kwargs):
        return oir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_Literal(self, node: gtir.Literal, **kwargs):
        return oir.Literal(value=self.visit(node.value), dtype=node.dtype, kind=node.kind)

    def visit_UnaryOp(self, node: gtir.UnaryOp, **kwargs):
        return oir.UnaryOp(op=node.op, expr=self.visit(node.expr))

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        return oir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs):
        return oir.TernaryOp(
            cond=self.visit(node.cond),
            true_expr=self.visit(node.true_expr),
            false_expr=self.visit(node.false_expr),
        )

    def visit_Cast(self, node: gtir.Cast, **kwargs):
        return oir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_FieldDecl(self, node: gtir.FieldDecl, **kwargs):
        return oir.FieldDecl(name=node.name, dtype=node.dtype)

    def visit_ScalarDecl(self, node: gtir.ScalarDecl, **kwargs):
        return oir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs):
        return oir.NativeFuncCall(
            func=node.func, args=self.visit(node.args), dtype=node.dtype, kind=node.kind
        )

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs
    ):
        mask_field_decl = _create_mask(ctx, "mask_" + node.id_, self.visit(node.cond))
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
                combined_mask = oir.BinaryOp(
                    op=LogicalOperator.AND, left=mask, right=combined_mask
                )
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    # For now we represent ScalarIf (and FieldIf) both as masks on the HorizontalExecution.
    # This is not meant to be set in stone...
    def visit_ScalarIfStmt(
        self, node: gtir.ScalarIfStmt, *, mask: oir.Expr = None, ctx: Context, **kwargs
    ):
        current_mask = self.visit(node.cond)
        combined_mask = current_mask
        if mask:
            combined_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=current_mask)

        self.visit(node.true_branch.body, mask=combined_mask, ctx=ctx)
        if node.false_branch:
            combined_mask = oir.UnaryOp(op=UnaryOperator.NOT, expr=current_mask)
            if mask:
                combined_mask = oir.BinaryOp(
                    op=LogicalOperator.AND, left=mask, right=combined_mask
                )
            self.visit(
                node.false_branch.body,
                mask=combined_mask,
                ctx=ctx,
            )

    def visit_Interval(self, node: gtir.Interval, **kwargs):
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        ctx = self.Context()
        self.visit(node.body, ctx=ctx)

        # TODO review this
        for temp in node.temporaries:
            ctx.add_decl(oir.Temporary(name=temp.name, dtype=temp.dtype))

        return oir.VerticalLoop(
            interval=self.visit(node.interval),
            loop_order=node.loop_order,
            declarations=ctx.decls,
            horizontal_executions=ctx.horizontal_executions,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params),
            vertical_loops=self.visit(node.vertical_loops),
        )

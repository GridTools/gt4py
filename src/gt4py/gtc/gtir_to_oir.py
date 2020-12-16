from typing import List, Sequence, Tuple

import eve  # noqa: F401
from devtools import debug

from gt4py.gtc import gtir, oir
from gt4py.gtc.common import CartesianOffset, DataType, LogicalOperator, UnaryOperator
from gt4py.gtc.utils import ListTuple, flatten_list


def _create_mask(name: str, cond: oir.Expr) -> Tuple:
    mask_field_decl = oir.Temporary(name=name, dtype=DataType.BOOL)
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
    return mask_field_decl, fill_mask_field


class GTIRToOIR(eve.NodeTranslator):
    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, **kwargs
    ) -> Tuple[List[oir.Temporary], List[oir.AssignStmt]]:
        tmp = oir.Temporary(name="tmp_" + node.left.name + "_" + node.id_, dtype=node.left.dtype)
        return ListTuple(
            [tmp],
            [
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
                ),
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
            ],
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

    def visit_FieldIfStmt(self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, **kwargs):
        mask_field_decl, fill_mask_h_exec = _create_mask("mask_" + node.id_, self.visit(node.cond))
        decls_and_h_execs = ListTuple([mask_field_decl], [fill_mask_h_exec])

        new_mask = oir.FieldAccess(
            name=mask_field_decl.name, offset=CartesianOffset.zero(), dtype=mask_field_decl.dtype
        )
        if mask:
            new_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=new_mask)

        decls_and_h_execs += self.visit(node.true_branch.body, mask=new_mask)
        if node.false_branch:
            decls_and_h_execs += self.visit(
                node.false_branch.body, mask=oir.UnaryOp(op=UnaryOperator.NOT, expr=new_mask)
            )

        return decls_and_h_execs

    def visit_ScalarIfStmt(self, node: gtir.ScalarIfStmt, **kwargs):
        # TODO Not sure if we should use a mask as well or generate an IfStmt with nesting.
        # Needs discussion
        raise NotImplementedError()

    def visit_Interval(self, node: gtir.Interval, **kwargs):
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    def tuple_visit(self, node, **kwargs):
        """Visits a list node and transforms a list of tuples to a ListTuple."""
        assert isinstance(node, Sequence)
        return ListTuple(*map(flatten_list, zip(*self.visit(node, **kwargs))))

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        decls, h_execs = self.tuple_visit(node.body)

        # TODO review this
        for temp in node.temporaries:
            decls.append(oir.Temporary(name=temp.name, dtype=temp.dtype))
        return oir.VerticalLoop(
            interval=self.visit(node.interval),
            loop_order=node.loop_order,
            declarations=decls,
            horizontal_executions=h_execs,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return oir.Stencil(
            name=node.name,
            params=self.visit(node.params),
            vertical_loops=self.visit(node.vertical_loops),
        )

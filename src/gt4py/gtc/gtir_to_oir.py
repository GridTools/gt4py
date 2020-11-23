import eve  # noqa: F401
from gt4py.gtc import gtir, oir
from typing import Sequence, Tuple, List

from gt4py.gtc.common import CartesianOffset, LogicalOperator, UnaryOperator

from devtools import debug

from gt4py.gtc.utils import flatten_list, ListTuple


class GTIRToOIR(eve.NodeTranslator):
    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, **kwargs
    ) -> Tuple[List[oir.Temporary], List[oir.AssignStmt]]:
        tmp = oir.Temporary(name="tmp_" + node.left.name + "_" + node.id_)
        return ListTuple(
            [tmp],
            [
                oir.HorizontalExecution(
                    body=[
                        oir.AssignStmt(
                            left=oir.FieldAccess(name=tmp.name, offset=CartesianOffset.zero()),
                            right=self.visit(node.right),
                        )
                    ],
                    mask=mask,
                ),
                oir.HorizontalExecution(
                    body=[
                        oir.AssignStmt(
                            left=self.visit(node.left),
                            right=oir.FieldAccess(name=tmp.name, offset=CartesianOffset.zero()),
                        )
                    ],
                    mask=mask,
                ),
            ],
        )

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        return oir.FieldAccess(name=node.name, offset=node.offset)

    def visit_Literal(self, node: gtir.Literal, **kwargs):
        return oir.Literal(value=self.visit(node.value), dtype=node.dtype, kind=node.kind)

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        debug(node.left)
        return oir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def create_mask(self, node: gtir.FieldIfStmt) -> Tuple:
        mask_field_decl = oir.Temporary(name="mask_" + node.id_)
        fill_mask_field = oir.HorizontalExecution(
            body=[
                oir.AssignStmt(
                    left=oir.FieldAccess(name=mask_field_decl.name, offset=CartesianOffset.zero()),
                    right=self.visit(node.cond),
                )
            ]
        )
        return mask_field_decl, fill_mask_field

    def visit_FieldIfStmt(self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, **kwargs):
        mask_field_decl, fill_mask_h_exec = self.create_mask(node)
        decls_and_h_execs = ListTuple([mask_field_decl], [fill_mask_h_exec])

        new_mask = oir.FieldAccess(name=mask_field_decl.name, offset=CartesianOffset.zero())
        if mask:
            new_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=new_mask)

        decls_and_h_execs += self.visit(node.true_branch.body, mask=new_mask)
        if node.false_branch:
            decls_and_h_execs += self.visit(
                node.false_branch.body, mask=oir.UnaryOp(op=UnaryOperator.NOT, expr=new_mask)
            )

        return decls_and_h_execs

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
        return oir.VerticalLoop(
            interval=self.visit(node.interval),
            loop_order=node.loop_order,
            declarations=decls,
            horizontal_executions=h_execs,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return oir.Stencil(
            name=node.name, params=[], vertical_loops=self.visit(node.vertical_loops)
        )

import eve  # noqa: F401
from gt4py.gtc import gtir, oir
from typing import Tuple, List
import itertools

from gt4py.gtc.common import CartesianOffset, LogicalOperator, UnaryOperator

from devtools import debug


class GTIRToOIR(eve.NodeTranslator):
    # def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
    #     return stageir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, *, mask: oir.Expr = None, **kwargs
    ) -> Tuple[List[oir.Temporary], List[oir.AssignStmt]]:
        tmp = oir.Temporary(name="tmp_" + node.left.name + "_" + node.id_)
        return [tmp], [
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
        ]

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        return oir.FieldAccess(name=node.name, offset=node.offset)

    def visit_Literal(self, node: gtir.Literal, **kwargs):
        return oir.Literal(value=self.visit(node.value), dtype=node.dtype, kind=node.kind)

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        debug(node.left)
        return oir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_FieldIfStmt(self, node: gtir.FieldIfStmt, *, mask: oir.Expr = None, **kwargs):
        decls = list()
        h_execs = list()

        mask_field = oir.Temporary(name="mask_" + node.id_)
        decls.append(mask_field)
        debug(node.cond)
        debug(self.visit(node.cond))
        fill_mask = oir.AssignStmt(
            left=oir.FieldAccess(name=mask_field.name, offset=CartesianOffset.zero()),
            right=self.visit(node.cond),
        )
        h_execs.append(oir.HorizontalExecution(body=[fill_mask]))

        new_mask = oir.FieldAccess(name=mask_field.name, offset=CartesianOffset.zero())
        if mask:
            new_mask = oir.BinaryOp(op=LogicalOperator.AND, left=mask, right=new_mask)

        decls_and_h_execs = self.visit(node.true_branch, mask=new_mask)
        true_decls, true_h_execs = tuple(map(list, zip(*decls_and_h_execs)))
        decls.extend(itertools.chain.from_iterable(true_decls))
        h_execs.extend(itertools.chain.from_iterable(true_h_execs))

        if node.false_branch:
            decls_and_h_execs = self.visit(
                node.false_branch, mask=oir.UnaryOp(op=UnaryOperator.NOT, expr=new_mask)
            )
            false_decls, false_h_execs = tuple(map(list, zip(*decls_and_h_execs)))
            decls.extend(itertools.chain.from_iterable(false_decls))
            h_execs.extend(itertools.chain.from_iterable(false_h_execs))

        return decls, h_execs

    def visit_Interval(self, node: gtir.Interval, **kwargs):
        return oir.Interval(
            start=self.visit(node.start),
            end=self.visit(node.end),
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        decls_and_h_execs: List[
            Tuple[List[oir.Temporary], List[oir.HorizontalExecution]]
        ] = self.visit(node.body)
        decls, h_execs = tuple(map(list, zip(*decls_and_h_execs)))
        debug(decls)
        return oir.VerticalLoop(
            interval=self.visit(node.interval),
            loop_order=node.loop_order,
            declarations=list(itertools.chain.from_iterable(decls)),
            horizontal_executions=list(itertools.chain.from_iterable(h_execs)),
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return oir.Stencil(
            name=node.name, params=[], vertical_loops=self.visit(node.vertical_loops)
        )

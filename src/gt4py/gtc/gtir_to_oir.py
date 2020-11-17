import eve  # noqa: F401
from gt4py.gtc import gtir, oir
from typing import Tuple, List
import itertools

from gt4py.gtc.common import CartesianOffset

from devtools import debug
from gt4py.gtc.gtir import VerticalInterval
from gt4py.gtc.oir import HorizontalExecution, Temporary


class GTIRToOIR(eve.NodeTranslator):
    # def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
    #     return stageir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_ParAssignStmt(
        self, node: gtir.ParAssignStmt, **kwargs
    ) -> Tuple[oir.Temporary, List[oir.AssignStmt]]:
        tmp = oir.Temporary(name="tmp_" + node.id_)
        return tmp, [
            oir.HorizontalExecution(
                body=[
                    oir.AssignStmt(
                        left=oir.FieldAccess(name=tmp.name, offset=CartesianOffset.zero()),
                        right=self.visit(node.right),
                    )
                ]
            ),
            oir.HorizontalExecution(
                body=[
                    oir.AssignStmt(
                        left=self.visit(node.left),
                        right=oir.FieldAccess(name=tmp.name, offset=CartesianOffset.zero()),
                    )
                ]
            ),
        ]

    def visit_VerticalInterval(self, node: gtir.VerticalInterval, **kwargs):
        # [decls, horizontal_executions] = self.visit(node.body)
        decls_and_hexecs: List[Tuple[Temporary, List[HorizontalExecution]]] = self.visit(node.body)
        decls, hexecs = tuple(map(list, zip(*decls_and_hexecs)))

        return decls, oir.VerticalInterval(
            start=self.visit(node.start),
            end=self.visit(node.end),
            horizontal_executions=list(itertools.chain.from_iterable(hexecs)),
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        decls_and_intervals: List[Tuple[List[Temporary], List[VerticalInterval]]] = self.visit(
            node.vertical_intervals
        )
        decls, intervals = tuple(map(list, zip(*decls_and_intervals)))
        debug(decls)
        debug(intervals)
        return oir.VerticalLoop(
            vertical_intervals=intervals,
            loop_order=node.loop_order,
            declarations=list(itertools.chain.from_iterable(decls)),
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return oir.Stencil(
            name=node.name, params=[], vertical_loops=self.visit(node.vertical_loops)
        )

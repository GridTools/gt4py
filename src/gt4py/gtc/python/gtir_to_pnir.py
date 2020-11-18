from itertools import chain
from typing import List, Tuple

from eve.visitors import NodeTranslator

from gt4py.gtc import gtir
from gt4py.gtc.gtir import AxisBound
from gt4py.gtc.python import pnir


class GtirToPnir(NodeTranslator):
    def visit_Stencil(self, node: gtir.Stencil) -> pnir.Stencil:
        stencil_obj = pnir.StencilObject(
            name=node.name, params=node.params, fields_metadata=node.fields_metadata
        )
        comp_module = pnir.Module(
            run=pnir.RunFunction(
                field_params=node.param_names,
                scalar_params=[],
                k_loops=list(
                    chain(*(self.visit(vertical_loop) for vertical_loop in node.vertical_loops))
                ),
            )
        )
        return pnir.Stencil(computation=comp_module, stencil_obj=stencil_obj)

    def visit_VerticalLoop(self, node: gtir.VerticalLoop) -> List[pnir.KLoop]:
        lower, upper = self.visit(node.interval)
        return [
            pnir.KLoop(
                lower=lower,
                upper=upper,
                ij_loops=[self.visit(stmt) for stmt in node.body],
            )
        ]

    def visit_Interval(self, node: gtir.Interval) -> Tuple[AxisBound, AxisBound]:
        return self.visit(node.start), self.visit(node.end)

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt) -> pnir.IJLoop:
        return pnir.IJLoop(
            body=[pnir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))]
        )

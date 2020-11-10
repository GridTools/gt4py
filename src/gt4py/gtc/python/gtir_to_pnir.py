from itertools import chain
from typing import List

from eve.visitors import NodeTranslator

from gt4py.gtc import gtir
from gt4py.gtc.python import pnir


class GtirToPnir(NodeTranslator):
    def visit_Computation(self, node: gtir.Computation) -> pnir.Stencil:
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

    # def visit_Stencil(self, node: gtir.Stencil) -> List[pnir.KLoop]:
    #     res = list(chain(*(self.visit(vertical_loop) for vertical_loop in node.vertical_loops)))
    #     return res

    def visit_VerticalLoop(self, node: gtir.VerticalLoop) -> List[pnir.KLoop]:
        res = [self.visit(vertical_interval) for vertical_interval in node.vertical_intervals]
        return res

    def visit_VerticalInterval(self, node: gtir.VerticalInterval) -> pnir.KLoop:
        return pnir.KLoop(
            lower=self.visit(node.start),
            upper=self.visit(node.end),
            ij_loops=[self.visit(stmt) for stmt in node.body],
        )

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt) -> pnir.IJLoop:
        return pnir.IJLoop(
            body=[pnir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))]
        )

    # def visit_HorizontalLoop(self, node: gtir.HorizontalLoop) -> pnir.IJLoop:
    #     return pnir.IJLoop(body=[self.visit(node.stmt)])

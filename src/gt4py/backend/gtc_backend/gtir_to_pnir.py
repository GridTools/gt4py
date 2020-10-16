import ast
from itertools import chain

from eve.visitors import NodeTranslator

from . import common, gtir, pnir


class GtirToPnir(NodeTranslator):
    def accumulate_visits(self, children):
        accum_nodes = []
        for child in children:
            accum_nodes.extend(self.visit(child))
        return accum_nodes

    def visit_Computation(self, node):
        stencil_obj = pnir.StencilObject(
            name=node.name, params=node.params, fields_metadata=node.fields_metadata
        )
        comp_module = pnir.Module(
            run=pnir.RunFunction(
                field_params=node.param_names,
                scalar_params=[],
                k_loops=list(chain(*(self.visit(stencil) for stencil in node.stencils))),
            )
        )
        return pnir.Stencil(computation=comp_module, stencil_obj=stencil_obj)

    def visit_Stencil(self, node):
        res = list(chain(*(self.visit(vertical_loop) for vertical_loop in node.vertical_loops)))
        print(res)
        return res

    def visit_VerticalLoop(self, node):
        res = [self.visit(vertical_interval) for vertical_interval in node.vertical_intervals]
        print(res)
        return res

    def visit_VerticalInterval(self, node):
        return pnir.KLoop(
            lower=self.visit(node.start),
            upper=self.visit(node.end),
            ij_loops=[self.visit(horizontal_loop) for horizontal_loop in node.horizontal_loops],
        )

    def visit_HorizontalLoop(self, node):
        return pnir.IJLoop(body=[self.visit(node.stmt)])

from eve.visitors import NodeTranslator

from . import npir


class GtirToNpir(NodeTranslator):
    def visit_Computation(self, node: gtir.Computation) -> npir.Computation:
        return npir.Computation(
            field_params = node.param_names,
            scalar_params = [],
            vertical_passes = [self.visit(stencil) for stencil in node.stencils]

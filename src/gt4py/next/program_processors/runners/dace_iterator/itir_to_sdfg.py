import gt4py.eve as eve
from gt4py.next.iterator import ir as itir
import dace
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts
from .utility import type_spec_to_dtype


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    offset_providers: dict[str, NeighborTableOffsetProvider]

    def __init__(self, param_types: list[ts.TypeSpec], offset_provider: dict[str, NeighborTableOffsetProvider]):
        self.param_types = param_types
        self.offset_providers = offset_provider

    def visit_StencilClosure(self, node: itir.StencilClosure) -> dace.SDFG:
        sdfg = dace.SDFG(name="stencil_closure")

        assert isinstance(node.output, itir.SymRef)
        sdfg.add_array(node.output.id, shape=domain,  )




    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        sdfg = dace.SDFG(name=node.id)
        last_state = sdfg.add_state("state", True)

        for closure in node.closures:
            self.visit(closure)
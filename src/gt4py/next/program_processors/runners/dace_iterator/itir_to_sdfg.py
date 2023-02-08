import sympy

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

        for input in node.inputs:
            shape = [rng[1] - rng[0] for _, rng in domain]
            sdfg.add_array(node.output.id, shape=shape, dtype=dace.float64)

        domain = self._visit_named_range(node.domain)
        shape = [rng[1] - rng[0] for _, rng in domain]
        sdfg.add_array(node.output.id, shape=shape, dtype=dace.float64)



    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        sdfg = dace.SDFG(name=node.id)
        last_state = sdfg.add_state("state", True)

        for closure in node.closures:
            self.visit(closure)

    def _visit_named_range(self, node: itir.FunCall) -> tuple[sympy.Basic, ...]:
        # cartesian_domain(named_range(IDim, start, end))
        assert isinstance(node.fun, itir.SymRef)
        assert node.fun.id == "cartesian_domain" or node.fun.id == "unstructured_domain"

        bounds: list[tuple[str, tuple[sympy.Basic, sympy.Basic]]] = []

        for named_range in node.args:
            assert isinstance(named_range, itir.FunCall)
            assert isinstance(named_range.fun, itir.SymRef)
            assert len(named_range.args) == 3
            dimension = named_range.args[0]
            lower_bound = named_range.args[1]
            upper_bound = named_range.args[2]
            sym_lower_bound = dace.symbolic.pystr_to_symbolic(str(lower_bound))
            sym_upper_bound = dace.symbolic.pystr_to_symbolic(str(upper_bound))
            bounds.append((dimension.value, (sym_lower_bound, sym_upper_bound)))

        return tuple(sorted(bounds, key=lambda item: item[0]))
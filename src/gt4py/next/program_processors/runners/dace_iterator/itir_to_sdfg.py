import textwrap

import dace
import sympy

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts

from .itir_to_tasklet import PythonTaskletCodegen
from .utility import type_spec_to_dtype


class _ShiftCollector(eve.NodeVisitor):
    def __init__(self, sdfg: dace.SDFG, last_state: dace.SDFGState):
        self.sdfg = sdfg
        self.last_state = last_state
        self.offset_table = dict()

    def visit_FunCall(self, node: itir.FunCall):
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "shift":
            pass
        else:
            pass


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    offset_provider: dict[str, NeighborTableOffsetProvider]

    def __init__(
            self,
            param_types: list[ts.TypeSpec],
            offset_provider: dict[str, NeighborTableOffsetProvider],
    ):
        self.param_types = param_types
        self.offset_provider = offset_provider

    @staticmethod
    def _check_no_lifts(node: itir.StencilClosure):
        if any(
                getattr(fun, "id", "") == "lift"
                for fun in eve.walk_values(node).if_isinstance(itir.FunCall).getattr("fun")
        ):
            return False
        return True

    @staticmethod
    def _check_no_inner_lambdas(node: itir.StencilClosure):
        if len(eve.walk_values(node.stencil).if_isinstance(itir.Lambda).to_list()) > 1:
            return False
        return True

    @staticmethod
    def _check_shift_offsets_are_literals(node: itir.StencilClosure):
        fun_calls = eve.walk_values(node).if_isinstance(itir.FunCall)
        shifts = [nd for nd in fun_calls if getattr(nd.fun, "id", "") == "shift"]
        for shift in shifts:
            if not all(isinstance(arg, itir.Literal) for arg in shift.args[1:]):
                return False
        return True

    def visit_StencilClosure(
            self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> dace.SDFG:
        """
        Preconditions:
            node has no lifts
            node has exactly 1 itir.Lambda node which is at the top level
            offset arguments to shifts are literals
        """
        assert ItirToSDFG._check_no_lifts(node)
        assert ItirToSDFG._check_no_inner_lambdas(node)
        assert ItirToSDFG._check_shift_offsets_are_literals(node)

        sdfg = dace.SDFG(name="stencil_closure")
        last_state = sdfg.add_state()

        shift_collector = _ShiftCollector(sdfg, last_state)
        shift_collector.visit(node.stencil)
        last_state = shift_collector.last_state

        assert isinstance(node.output, itir.SymRef)
        assert len(eve.walk_values(node).if_isinstance(itir.Lambda).to_list()) == 1

        domain = self._visit_named_range(node.domain)
        shape = array_table[node.output.id].shape
        sdfg.add_array(node.output.id, shape=shape, dtype=dace.float64)

        for inp in node.inputs:
            shape = array_table[node.output.id].shape  # Assume same as output
            sdfg.add_array(inp.id, shape=shape, dtype=dace.float64)

        input_memlets = {
            f"{lambda_param.id}_full": dace.Memlet(
                data=str(inp.id), subset=",".join(f"0:{size}" for size in sdfg.arrays[inp.id].shape)
            )
            for inp, lambda_param in zip(node.inputs, node.stencil.params)
        }
        output_memlets = {
            f"{node.output.id}_element": dace.Memlet(
                data=str(node.output.id),
                subset=",".join(f"i_{dim}" for dim, _ in domain),
            )
        }

        map_range = {f"i_{dim}": f"{lb}:{ub}" for dim, (lb, ub) in domain}

        stencil_args = '\n'.join(f"{param} = {param}_full" for param in node.stencil.params)
        stencil_body = PythonTaskletCodegen(offset_provider=self.offset_provider).visit(node.stencil)
        stencil_code = textwrap.dedent(
            f"{stencil_args}\n"
            f"{node.output.id}_element = {stencil_body}"
        )

        last_state.add_mapped_tasklet(
            name="addition",
            map_ranges=map_range,
            inputs=input_memlets,
            code=stencil_code,
            # language=dace.Language.CPP,
            # code=f"{node.output.id}_element = (lambda : 7)()",
            language=dace.Language.Python,
            outputs=output_memlets,
            external_edges=True,
            schedule=dace.ScheduleType.Sequential,
        )

        return sdfg

    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        sdfg = dace.SDFG(name=node.id)
        last_state = sdfg.add_state()

        for param, type_ in zip(node.params, self.param_types):
            if isinstance(type_, ts.FieldType):
                shape = (dace.symbol(sdfg.temp_data_name()) for _ in range(len(type_.dims)))
                sdfg.add_array(str(param.id), shape=shape, dtype=dace.float64)
            elif isinstance(type_, ts.ScalarType):
                sdfg.add_symbol(param.id, type_spec_to_dtype(type_))
            else:
                raise NotImplementedError()

        for closure in node.closures:
            last_state = sdfg.add_state_after(last_state)

            closure_sdfg = self.visit(closure, array_table=sdfg.arrays)
            nsdfg_node = last_state.add_nested_sdfg(
                closure_sdfg,
                None,
                inputs={str(inp.id) for inp in closure.inputs},
                outputs={str(closure.output.id)},
            )

            input_accesses = [last_state.add_access(inp.id) for inp in closure.inputs]
            output_access = last_state.add_access(closure.output.id)

            for inner_name, access_node in zip(closure.inputs, input_accesses):
                last_state.add_edge(
                    access_node,
                    None,
                    nsdfg_node,
                    inner_name.id,
                    dace.Memlet(
                        data=access_node.data,
                        subset=", ".join(f"0:{sh}" for sh in sdfg.arrays[access_node.data].shape),
                    ),
                )

            last_state.add_edge(
                nsdfg_node,
                closure.output.id,
                output_access,
                None,
                dace.Memlet(
                    data=output_access.data,
                    subset=", ".join(f"0:{sh}" for sh in sdfg.arrays[output_access.data].shape),
                ),
            )

        sdfg.validate()
        return sdfg

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

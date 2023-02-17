import textwrap

import dace
import sympy
from typing import Any

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts

from .itir_to_tasklet import closure_to_tasklet
from .utility import type_spec_to_dtype


def _connectivity_identifier(name: str):
    return f"__connectivity_{name}"


def _filter_neighbor_tables(offset_provider: dict[str, Any]):
    return [
        (offset, table)
        for offset, table in offset_provider.items()
        if isinstance(table, NeighborTableOffsetProvider)
    ]


def _create_memlet_full(source_identifier: str, source_array: dace.data.Array):
    bounds = [(0, size) for size in source_array.shape]
    subset = ", ".join(f"{lb}:{ub}" for lb, ub in bounds)
    return dace.Memlet(data=source_identifier, subset=subset)


def _create_memlet_at(source_identifier: str, index: tuple[str, ...]):
    subset = ", ".join(index)
    return dace.Memlet(data=source_identifier, subset=subset)


class ItirToSDFG(eve.NodeVisitor):
    param_types: list[ts.TypeSpec]
    offset_provider: dict[str, Any]

    def __init__(
            self,
            param_types: list[ts.TypeSpec],
            offset_provider: dict[str, NeighborTableOffsetProvider],
    ):
        self.param_types = param_types
        self.offset_provider = offset_provider

    def visit_StencilClosure(
            self, node: itir.StencilClosure, array_table: dict[str, dace.data.Array]
    ) -> dace.SDFG:
        assert ItirToSDFG._check_no_lifts(node)
        assert ItirToSDFG._check_no_inner_lambdas(node)
        assert ItirToSDFG._check_shift_offsets_are_literals(node)
        assert isinstance(node.output, itir.SymRef)

        # Filter neighbor tables from offset providers.
        neighbor_tables = _filter_neighbor_tables(self.offset_provider)

        # Create the closure's nested SDFG and single state.
        closure_sdfg = dace.SDFG(name="stencil_closure")
        closure_state = closure_sdfg.add_state()

        # Add DaCe arrays for inputs, output and connectivities to closure SDFG.
        for inp in node.inputs:
            shape = array_table[inp.id].shape
            dtype = array_table[inp.id].dtype
            closure_sdfg.add_array(inp.id, shape=shape, dtype=dtype)

        shape = array_table[node.output.id].shape
        dtype = array_table[node.output.id].dtype
        closure_sdfg.add_array(node.output.id, shape=shape, dtype=dtype)

        for offset, table in neighbor_tables:
            name = _connectivity_identifier(offset)
            shape = array_table[name].shape
            dtype = array_table[name].dtype
            closure_sdfg.add_array(name, shape=shape, dtype=dtype)

        # Get computational domain of the closure
        closure_domain = self._visit_domain(node.domain)
        map_domain = {f"i_{dim}": f"{lb}:{ub}" for dim, (lb, ub) in closure_domain}

        # Add memlets as subsets of the input, output and connectivity arrays.
        input_memlets = {
            f"{stencil_param.id}_full": _create_memlet_full(str(closure_input.id), closure_sdfg.arrays[closure_input.id])
            for closure_input, stencil_param in zip(node.inputs, node.stencil.params)
        }
        output_memlets = {
            f"{node.output.id}_element": _create_memlet_at(str(node.output.id), tuple(idx for idx in map_domain.keys()))
        }
        connectivity_memlets = {
            f"{_connectivity_identifier(conn)}_full": _create_memlet_full(
                _connectivity_identifier(conn),
                closure_sdfg.arrays[_connectivity_identifier(conn)]
            )
            for conn, _ in neighbor_tables
        }

        # Translate the stencil's code into a DaCe tasklet.
        index_domain = {dim: f"i_{dim}" for dim, _ in closure_domain}
        stencil_args = '\n'.join(f"{param} = {param}_full" for param in node.stencil.params)
        stencil_expr = closure_to_tasklet(node, self.offset_provider, index_domain)
        stencil_code = textwrap.dedent(
            f"{stencil_args}\n"
            f"{node.output.id}_element = {stencil_expr}"
        )

        closure_state.add_mapped_tasklet(
            name=node.stencil.id if isinstance(node.stencil, itir.FunctionDefinition) else "lambda",
            map_ranges=map_domain,
            inputs={**input_memlets, **connectivity_memlets},
            code=stencil_code,
            language=dace.Language.Python,
            outputs=output_memlets,
            external_edges=True,
            schedule=dace.ScheduleType.Sequential,
        )

        return closure_sdfg

    def visit_FencilDefinition(self, node: itir.FencilDefinition):
        program_sdfg = dace.SDFG(name=node.id)
        last_state = program_sdfg.add_state()

        # Filter neighbor tables from offset providers.
        neighbor_tables = [
            (offset, table)
            for offset, table in self.offset_provider.items()
            if isinstance(table, NeighborTableOffsetProvider)
        ]
        connectivity_names = [_connectivity_identifier(offset) for offset, _ in neighbor_tables]

        # Add program parameters as SDFG arrays and symbols.
        for param, type_ in zip(node.params, self.param_types):
            if isinstance(type_, ts.FieldType):
                shape = (dace.symbol(program_sdfg.temp_data_name()) for _ in range(len(type_.dims)))
                dtype = type_spec_to_dtype(type_.dtype)
                program_sdfg.add_array(str(param.id), shape=shape, dtype=dtype)
            elif isinstance(type_, ts.ScalarType):
                program_sdfg.add_symbol(param.id, type_spec_to_dtype(type_))
            else:
                raise NotImplementedError()

        # Add connectivities as SDFG arrays.
        for offset, table in neighbor_tables:
            shape = (dace.symbol(program_sdfg.temp_data_name()) for _ in range(2))
            program_sdfg.add_array(_connectivity_identifier(offset), shape=shape, dtype=dace.int64)

        # Create a nested SDFG for all stencil closures.
        for closure in node.closures:
            # Translate the closure and its stencil's body to an SDFG.
            closure_sdfg = self.visit(closure, array_table=program_sdfg.arrays)

            # Create a new state for the closure.
            last_state = program_sdfg.add_state_after(last_state)

            # Insert the closure's SDFG as a nested SDFG of the program.
            nsdfg_inputs = {str(inp.id) for inp in closure.inputs}
            nsdfg_connectivities = {name for name in connectivity_names}
            nsdfg_outputs = {str(closure.output.id)}
            nsdfg_all_inputs = {*nsdfg_inputs, *nsdfg_connectivities}
            nsdfg_node = last_state.add_nested_sdfg(closure_sdfg, None, inputs=nsdfg_all_inputs, outputs=nsdfg_outputs)

            # Add access nodes for the program parameters to the closure's state in the program.
            input_accesses = [last_state.add_access(inp.id) for inp in closure.inputs]
            connectivity_accesses = [last_state.add_access(name) for name in connectivity_names]
            output_access = last_state.add_access(closure.output.id)

            # Connect the access nodes to the nested SDFG's inputs via edges.
            for inner_name, access_node in zip(closure.inputs, input_accesses):
                input_memlet = _create_memlet_full(access_node.data, program_sdfg.arrays[access_node.data])
                last_state.add_edge(access_node, None, nsdfg_node, inner_name.id, input_memlet)

            for inner_name, access_node in zip(connectivity_names, connectivity_accesses):
                conn_memlet = _create_memlet_full(access_node.data, program_sdfg.arrays[access_node.data])
                last_state.add_edge(access_node, None, nsdfg_node, inner_name, conn_memlet)

            output_memlet = _create_memlet_full(output_access.data, program_sdfg.arrays[output_access.data])
            last_state.add_edge(nsdfg_node, closure.output.id, output_access, None, output_memlet)

        program_sdfg.view()
        program_sdfg.validate()
        return program_sdfg

    def _visit_domain(self, node: itir.FunCall) -> tuple[sympy.Basic, ...]:
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
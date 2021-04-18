# -*- coding: utf-8 -*-
import dace
import dace.data
import networkx as nx

import eve
from gtc import common, oir
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode


class HorizontalExecutionFieldRenamer(eve.NodeTranslator):
    def visit_FieldAccess(self, node: oir.FieldAccess):
        if node.name not in self._field_table:
            return node
        return oir.FieldAccess(
            name=self._field_table[node.name], offset=node.offset, dtype=node.dtype
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess):
        if node.name not in self._field_table:
            return node
        return oir.ScalarAccess(name=self._field_table[node.name], dtype=node.dtype)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        assert all(
            decl.name not in self._field_table or self._field_table[decl.name] == decl.name
            for decl in node.declarations
        )
        return node

    def __init__(self, field_table):
        self._field_table = field_table


def get_node_name_mapping(state: dace.SDFGState, node: dace.nodes.LibraryNode):

    name_mapping = dict()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            assert edge.dst_conn.startswith("IN_")
            internal_name = edge.dst_conn[len("IN_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to "
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    for edge in state.out_edges(node):
        if edge.src_conn is not None:
            assert edge.src_conn.startswith("OUT_")
            internal_name = edge.src_conn[len("OUT_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to"
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    return name_mapping


def convert(sdfg: dace.SDFG) -> oir.Stencil:
    vertical_loops = list()
    params = list()
    decls = list()

    array: dace.data.Data
    for name, array in sdfg.arrays.items():
        dtype = common.typestr_to_data_type(array.dtype.as_numpy_dtype().str)
        if isinstance(array, dace.data.Scalar):
            params.append(oir.ScalarDecl(name=name, dtype=dtype))
        else:
            dimensions = list(
                any(dace.symbol(f"__{name}_{k}_stride") in s.free_symbols for s in array.strides)
                for k in "IJK"
            )
            if not array.transient:
                params.append(oir.FieldDecl(name=name, dtype=dtype, dimensions=dimensions))
            else:
                decls.append(oir.Temporary(name=name, dtype=dtype, dimensions=dimensions))

    internal_symbols = ["__I", "__J", "__K"] + list(
        f"__{name}_{var}_stride" for name in sdfg.arrays.keys() for var in "IJK"
    )
    for sym, stype in sdfg.symbols.items():
        if sym not in internal_symbols:
            params.append(
                oir.ScalarDecl(
                    name=sym, dtype=common.typestr_to_data_type(stype.as_numpy_dtype().str)
                )
            )

    is_correct_node_types = all(
        isinstance(n, (dace.SDFGState, dace.nodes.AccessNode, VerticalLoopLibraryNode))
        for n, _ in sdfg.all_nodes_recursive()
    )
    is_correct_data_and_dtype = all(
        isinstance(array, (dace.data.Scalar, dace.data.Array))
        and common.typestr_to_data_type(array.dtype.as_numpy_dtype().str) != common.DataType.INVALID
        for array in sdfg.arrays.values()
    )
    if not is_correct_node_types or not is_correct_data_and_dtype:
        raise ValueError("Tried to convert incompatible SDFG to OIR.")

    for state in sdfg.topological_sort(sdfg.start_state):

        for node in (
            n for n in nx.topological_sort(state.nx) if isinstance(n, VerticalLoopLibraryNode)
        ):
            field_name_table = get_node_name_mapping(state, node)
            sections = []
            for interval, sdfg in node.sections:
                he_name_table = get_node_name_mapping(state, node)
                he_name_table.update(field_name_table)
                horizontal_executions = [
                    HorizontalExecutionFieldRenamer(he_name_table).visit(n.oir_node)
                    for n in nx.topological_sort(sdfg.states()[0].nx)
                    if isinstance(n, HorizontalExecutionLibraryNode)
                ]
                sections.append(
                    oir.VerticalLoopSection(
                        interval=interval, horizontal_executions=horizontal_executions
                    )
                )

            new_node = oir.VerticalLoop(
                sections=sections,
                loop_order=node.loop_order,
                caches=node.caches,
            )
            vertical_loops.append(new_node)

    return oir.Stencil(
        name=sdfg.name, params=params, declarations=decls, vertical_loops=vertical_loops
    )

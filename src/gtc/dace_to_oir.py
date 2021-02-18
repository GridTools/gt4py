# -*- coding: utf-8 -*-
import dace
import dace.data
import networkx as nx

from gtc import common, oir
from gtc.dace.nodes import VerticalLoopLibraryNode


def _is_list_of_states(sdfg: dace.SDFG):
    state = sdfg.start_state
    seen_states = set(state)
    while sdfg.out_degree(state) > 0:
        if sdfg.out_degree(state) != 1:
            return False
        state = next(iter(sdfg.out_edges(state))).dst
        if state in seen_states:
            return False
        seen_states.add(state)
    return True


def convert(sdfg: dace.SDFG) -> oir.Stencil:
    vertical_loops = list()
    params = list()
    declared = set()

    array: dace.data.Data
    for name, array in sdfg.arrays.items():
        dtype = common.typestr_to_data_type(array.dtype.as_numpy_dtype().str)
        if isinstance(array, dace.data.Scalar):
            params.append(oir.ScalarDecl(name=name, dtype=dtype))
        elif not array.transient:
            params.append(oir.FieldDecl(name=name, dtype=dtype))

    is_correct_node_types = all(
        isinstance(n, (dace.SDFGState, dace.nodes.AccessNode, VerticalLoopLibraryNode))
        for n, _ in sdfg.all_nodes_recursive()
    )
    is_correct_data_and_dtype = all(
        isinstance(array, (dace.data.Scalar, dace.data.Array))
        and common.typestr_to_data_type(array.dtype.as_numpy_dtype().str) != common.DataType.INVALID
        for array in sdfg.arrays.values()
    )
    if not is_correct_node_types or not _is_list_of_states(sdfg) or not is_correct_data_and_dtype:
        raise ValueError("Tried to convert incompatible SDFG to OIR.")

    for state in sdfg.topological_sort(sdfg.start_state):

        for node in nx.topological_sort(state.nx):
            if isinstance(node, VerticalLoopLibraryNode):
                decls = []
                for edge in state.in_edges(node):
                    if (
                        edge.dst_conn is not None
                        and isinstance(edge.src, dace.nodes.AccessNode)
                        and edge.src.data not in declared
                        and edge.src.data not in params
                        and sdfg.arrays[edge.src.data].transient
                    ):
                        declared.add(edge.src.data)
                        dtype = common.typestr_to_data_type(
                            sdfg.arrays[edge.src.data].dtype.as_numpy_dtype().str
                        )
                        decls.append(oir.FieldDecl(name=edge.src.data, dtype=dtype))
                for edge in state.out_edges(node):
                    if (
                        edge.src_conn is not None
                        and isinstance(edge.dst, dace.nodes.AccessNode)
                        and edge.dst.data not in declared
                        and edge.dst.data not in params
                        and sdfg.arrays[edge.dst.data].transient
                    ):
                        declared.add(edge.dst.data)
                        dtype = common.typestr_to_data_type(
                            sdfg.arrays[edge.dst.data].dtype.as_numpy_dtype().str
                        )
                        decls.append(oir.FieldDecl(name=edge.dst.data, dtype=dtype))
                new_node = oir.VerticalLoop(
                    interval=node.oir_node.interval,
                    horizontal_executions=node.oir_node.horizontal_executions,
                    loop_order=node.oir_node.loop_order,
                    declarations=decls,
                )
                vertical_loops.append(new_node)

    return oir.Stencil(name=sdfg.name, params=params, vertical_loops=vertical_loops)

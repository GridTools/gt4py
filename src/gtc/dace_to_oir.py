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
    decls = list()

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
                vertical_loops.append(node._oir_node)

    array: dace.data.Data
    for name, array in sdfg.arrays.items():
        dtype = common.typestr_to_data_type(array.dtype.as_numpy_dtype().str)
        if isinstance(array, dace.data.Scalar):
            decls.append(oir.ScalarDecl(name=name, dtype=dtype))
        else:
            decls.append(oir.FieldDecl(name=name, dtype=dtype))
    return oir.Stencil(name=sdfg.name, params=decls, vertical_loops=vertical_loops)

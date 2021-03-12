# -*- coding: utf-8 -*-
import dace
import dace.data
import networkx as nx

from gtc import common, oir
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode


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
                any(dace.symbol(k) in s.free_symbols for s in array.shape) for k in "IJK"
            )
            if not array.transient:
                params.append(oir.FieldDecl(name=name, dtype=dtype, dimensions=dimensions))
            else:
                decls.append(oir.Temporary(name=name, dtype=dtype, dimensions=dimensions))

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
            sections = []
            for interval, sdfg in node.sections:
                horizontal_executions = [
                    n.oir_node
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

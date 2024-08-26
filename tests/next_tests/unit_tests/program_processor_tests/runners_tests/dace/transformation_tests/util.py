# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union, Literal, overload

import dace
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

__all__ = [
    "_count_nodes",
]


@overload
def _count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[False],
) -> int: ...


@overload
def _count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[True],
) -> list[dace_nodes.Node]: ...


def _count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: bool = False,
) -> Union[int, list[dace_nodes.Node]]:
    """Counts the number of nodes in of a particular type in `graph`.

    If `graph` is an SDFGState then only count the nodes inside this state,
    but if `graph` is an SDFG count in all states.

    Args:
        graph: The graph to scan.
        node_type: The type or sequence of types of nodes to look for.
    """

    states = graph.states() if isinstance(graph, dace.SDFG) else [graph]
    found_nodes: list[dace_nodes.Node] = []
    for state_nodes in states:
        for node in state_nodes.nodes():
            if isinstance(node, node_type):
                found_nodes.append(node)
    if return_nodes:
        return found_nodes
    return len(found_nodes)
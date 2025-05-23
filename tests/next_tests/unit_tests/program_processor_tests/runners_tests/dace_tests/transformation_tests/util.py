# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import uuid
from typing import Literal, Union, overload, Any

import dace
import copy
from dace.sdfg import nodes as dace_nodes


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[False],
) -> int: ...


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[True],
) -> list[dace_nodes.Node]: ...


def count_nodes(
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


def unique_name(name: str) -> str:
    """Adds a unique string to `name`."""
    maximal_length = 200
    unique_sufix = str(uuid.uuid1()).replace("-", "_")
    if len(name) > (maximal_length - len(unique_sufix)):
        name = name[: (maximal_length - len(unique_sufix) - 1)]
    return f"{name}_{unique_sufix}"


def compile_and_run_sdfg(
    sdfg: dace.SDFG,
    *args: Any,
    **kwargs: Any,
) -> dace.codegen.CompiledSDFG:
    """This function guarantees that the SDFG is compiled and run.

    This function will modify the name of the SDFG to ensure that the code is
    regenerated and recompiled properly. It will also suppress warnings about
    shared objects that are loaded multiple times.
    """
    sdfg_clone = copy.deepcopy(sdfg)

    sdfg_clone.name = unique_name(sdfg_clone.name)
    sdfg_clone._recompile = True
    sdfg_clone._regenerate_code = True  # Does this anything?
    csdfg = sdfg_clone.compile()
    csdfg(*args, **kwargs)

    return csdfg

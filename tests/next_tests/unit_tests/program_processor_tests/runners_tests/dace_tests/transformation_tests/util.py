# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import uuid
from typing import Literal, Union, overload, Any

import numpy as np
import copy

import dace
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
    return f"{name}_{str(uuid.uuid1()).replace('-', '_')}"


def make_sdfg_args(
    sdfg: dace.SDFG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generates the arguments to call the SDFG.

    The function returns data for the reference and the result call.
    You can compare the two using `compar_sdfg_res()`.
    """
    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)
    return ref, res


def evaluate_sdfg(
    sdfg: dace.SDFG,
    args: dict[str, Any],
) -> Any:
    """Compile and evaluate the SDFG."""
    csdfg = sdfg.compile()
    return csdfg(**args)


def compare_sdfg_res(
    ref: dict[str, Any],
    res: dict[str, Any],
) -> bool:
    """Compares if `res` and  `ref` are the same."""
    return all(np.allclose(ref[name], res[name]) for name in ref.keys())

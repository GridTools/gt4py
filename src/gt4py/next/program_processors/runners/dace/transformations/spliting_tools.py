# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Optional

import dace
from dace import data as dace_data, subsets as dace_sbs
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dataclasses.dataclass
class EdgeConnectionSpec:
    """Describes an edge in an abstract way, that is kind of independent of the direction.

    It is a tuple of length three. The first element is the node of interest, this can
    either be the source or destination node. The second element is the subset of the
    Memlet at the node. The third and last element is the actual edge.
    """

    node: dace_nodes.Node
    subset: dace_sbs.Subset
    edge: dace_graph.MultiConnectorEdge


def describe_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
    incoming_edges: bool,
) -> list[EdgeConnectionSpec]:
    """Generate the description of the edge."""
    get_sbs = lambda e: e.data.dst_subset if incoming_edges else e.data.src_subset  # noqa: E731 [lambda-assignment]
    get_node = lambda e: e.dst if incoming_edges else e.src  # noqa: E731 [lambda-assignment]
    edges = state.in_edges(node) if incoming_edges else state.out_edges(node)
    description = [
        EdgeConnectionSpec(
            node=get_node(e),
            subset=get_sbs(e),
            edge=e,
        )
        for e in edges
    ]
    assert not any(desc.subset is None for desc in description)
    return description


def describe_incoming_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
) -> list[EdgeConnectionSpec]:
    return describe_edges(state, node, True)


def describe_outgoing_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
) -> list[EdgeConnectionSpec]:
    return describe_edges(state, node, False)


def describes_incoming_edge(desc: EdgeConnectionSpec) -> bool:
    return desc.node is desc.edge.dst


def split_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    split_description: list[dace_sbs.Subset],
    already_reconfigured_nodes: Optional[set[tuple[dace_nodes.Node, str]]] = None,
) -> dict[dace_sbs.Subset, dace_nodes.AccessNode]:
    """The function will split `node_to_split` into several smaller AccessNodes.

    How the split is performed is described by `split_description`, essentially it
    is a list of subsets that describes the sizes of the new AccessNodes.
    There are some special cases in this function.
    """

    if already_reconfigured_nodes is None:
        already_reconfigured_nodes = set()

    desc_to_split = node_to_split.desc(sdfg)
    assert desc_to_split.transient
    assert gtx_transformations.utils.is_view(desc_to_split)
    assert isinstance(desc_to_split, dace_data.Array)

    input_descriptions = describe_incoming_edges(state, node_to_split)
    output_descriptions = describe_outgoing_edges(state, node_to_split)
    edge_descriptions = input_descriptions + output_descriptions

    # Ensure that no edge is on multiple new fragments and that every edge
    #  lands on a fragment.
    assert all(
        any(split.covers(desc.subset) for split in split_description) for desc in edge_descriptions
    ), "There is an edge that would be split."

    assignment = _compute_assignement_for_split(edge_descriptions, split_description)
    new_access_nodes = _generate_desc_and_access_nodes_for_split(
        state, sdfg, node_to_split, assignment
    )

    _perform_node_split(
        state=state,
        sdfg=sdfg,
        node_to_split=node_to_split,
        new_access_nodes=new_access_nodes,
        assignment=assignment,
        already_reconfigured_nodes=already_reconfigured_nodes,
    )

    return new_access_nodes


def _perform_node_split(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    new_access_nodes: dict[dace_sbs.Subset, dace_nodes.AccessNode],
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]],
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> None:
    """Performs the actual split of `node_to_split` based on `assignment`.

    `assignment` describes how the edges of `node_to_split` should be distributed
    among the `new_access_nodes`. It maps a "split", i.e. a subset that describes
    one of the shards of the original AccessNode, to the edges that should be
    moved to another AccessNode, which is described through `new_access_nodes`.

    The function will also reconfigure the dataflow of the old edges. The argument
    `already_reconfigured_nodes` is used to keep track which dataflow has already
    been reconfigured.
    The function will then propagate the strides starting from the new functions.

    Furthermore, the function will remove the `node_to_split` AccessNode, but it
    will not remove the data.
    """
    assert all(sbs in assignment for sbs in new_access_nodes.keys())
    assert state.degree(node_to_split) == sum(len(assig) for assig in assignment.values())

    # Iterate over all splits.
    handled_edges: set[dace_graph.MultiConnectorEdge] = set()
    for split in new_access_nodes:
        edges_to_relocate = assignment[split]
        new_access_node = new_access_nodes[split]
        assert state.degree(new_access_node) == 0

        # Iterate over all edges, in that split to relocate.
        for edge_to_relocate in edges_to_relocate:
            assert edge_to_relocate.edge not in handled_edges
            _perform_node_split_impl(
                state=state,
                sdfg=sdfg,
                original_node=node_to_split,
                new_access_node=new_access_node,
                split_description=split,
                edge_to_relocate=edge_to_relocate,
                already_reconfigured_nodes=already_reconfigured_nodes,
            )
            handled_edges.add(edge_to_relocate.edge)
        assert state.degree(new_access_node) == len(edges_to_relocate)
    assert state.degree(node_to_split) == 0

    # Propagate the strides starting from the new access nodes.
    for new_access_node in new_access_nodes.values():
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=state,
            outer_node=new_access_node,
        )

    # We remove the AccessNode but we do not remove the data. This is done because
    #  the function
    state.remove_node(node_to_split)


def _perform_node_split_impl(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    original_node: dace_nodes.AccessNode,
    new_access_node: dace_nodes.AccessNode,
    split_description: dace_sbs.Subset,
    edge_to_relocate: EdgeConnectionSpec,
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> dace_graph.MultiConnectorEdge:
    """Performs the actual split.

    In essence this function will create a new edge, based on but `edge_to_relocate`,
    but where `original_node` is replaced with `new_access_node`. Furthermore, the
    subset of the edge is modified to reflect this change and the dataflow on the
    "other side of the edge" is also modified by means of `reconfigure_dataflow_after_rerouting()`.
    However, it is important that the function does not propagate strides.

    The old edge is removed and the new one is returned.
    """
    # TODO(phimuell): In case the other node is an AccessNode it is potentially
    #   possible to bypass `new_access_node`. This is an optimization that we
    #   should do as it also acts as kind of redundant array removal.

    # Because the edge no longer writes into `node_to_split` but into `new_access_node`,
    #  which is smaller, we have to adapt the size. However, because we assume that
    #  the split covers the subset the edge transfers, the offset is simply given
    #  by where the split starts. The minus is needed for the correct format in
    #  `reroute_edge()`.
    assert split_description.covers(edge_to_relocate.subset)
    subset_correction: Any = [-min_elem for min_elem in split_description.min_element()]

    is_producer_edge = describes_incoming_edge(edge_to_relocate)

    new_edge = gtx_transformations.utils.reroute_edge(
        is_producer_edge=is_producer_edge,
        current_edge=edge_to_relocate.edge,
        ss_offset=subset_correction,
        state=state,
        sdfg=sdfg,
        old_node=original_node,
        new_node=new_access_node,
    )

    # Depending on the situation we do not need a reconfiguration.
    if is_producer_edge:
        other_node: dace_nodes.Node = edge_to_relocate.edge.src
        other_node_conn = edge_to_relocate.edge.src_conn
    else:
        other_node = edge_to_relocate.edge.dst
        other_node_conn = edge_to_relocate.edge.dst_conn

    # Reconfigure the data flow on the other side.
    if (other_node, other_node_conn) not in already_reconfigured_nodes:
        already_reconfigured_nodes.add((other_node, other_node_conn))
        if isinstance(other_node.desc(sdfg), dace_data.Scalar):
            subset_correction = None

        elif isinstance(
            other_node, (dace_nodes.AccessNode, dace_nodes.MapExit, dace_nodes.NestedSDFG)
        ):
            # There is nothing special to do. In case of a nested SDFG, we will also have to do
            #  stride propagation, but we will postpone that.
            pass

        else:
            raise TypeError(f"Can not handle a producer of type '{type(other_node).__name__}'")

        gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
            is_producer_edge=is_producer_edge,
            new_edge=new_edge,
            ss_offset=subset_correction,
            state=state,
            sdfg=sdfg,
            old_node=original_node,
            new_node=new_access_node,
        )

    state.remove_edge(edge_to_relocate.edge)
    return new_edge


def _generate_desc_and_access_nodes_for_split(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]],
) -> dict[dace_sbs.Subset, dace_nodes.AccessNode]:
    """Creates the data container and AccessNodes for the split data.

    The function will generate a data descriptor and AccessNode in `state`.
    If there is no assignment for a split, then nothing is created.
    """
    new_access_nodes: dict[dace_sbs.Subset, dace_nodes.AccessNode] = {}
    desc_to_split = node_to_split.desc(sdfg)

    for i, (split_subset, edges_to_relocate) in enumerate(assignment.items()):
        if len(edges_to_relocate) == 0:
            continue

        tmp_shape = split_subset.size()
        tmp_name, tmp_desc = sdfg.add_transient(
            name=f"{node_to_split.data}_split_{i}",
            shape=tmp_shape,
            dtype=desc_to_split.dtype,
            storage=desc_to_split.storage,
            find_new_name=True,
        )
        new_access_nodes[split_subset] = state.add_access(tmp_name, node_to_split.debuginfo)

    return new_access_nodes


def _compute_assignement_for_split(
    edge_descriptions: list[EdgeConnectionSpec],
    split_description: list[dace_sbs.Subset],
) -> dict[dace_sbs.Subset, set[EdgeConnectionSpec]]:
    """For every subset, that defines a split find the set of edges that belongs into it.

    Note that it might happens that some splits have zero assigned edges.
    """
    assert all(split is not None for split in split_description)
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]] = {
        split: set() for split in split_description
    }

    for edge_description in edge_descriptions:
        assert edge_description.subset is not None
        assigned_split = next(
            iter(split for split in split_description if split.covers(edge_description.subset)),
            None,
        )
        assert assigned_split is not None
        assignment[assigned_split].add(edge_description)

    return assignment

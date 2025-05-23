# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Iterable, Optional, Sequence

import dace
from dace import data as dace_data, subsets as dace_sbs, symbolic as dace_sym
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dataclasses.dataclass(frozen=True)
class EdgeConnectionSpec:
    """Describes an edge in an abstract way, that is kind of independent of the direction.

    It is a tuple of length three. The first element is the node of interest, this can
    either be the source or destination node. The second element is the subset of the
    Memlet at the node. The third and last element is the actual edge.

    It has the same hash as an edge.
    """

    node: dace_nodes.Node
    subset: dace_sbs.Subset
    edge: dace_graph.MultiConnectorEdge

    def __hash__(self) -> int:
        return hash(self.edge)


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


def get_other_node(desc: EdgeConnectionSpec) -> dace_nodes.Node:
    return desc.edge.src if describes_incoming_edge(desc) else desc.edge.dst


def split_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    split_description: Sequence[dace_sbs.Subset],
    allow_to_bypass_nodes: bool = False,
    already_reconfigured_nodes: Optional[set[tuple[dace_nodes.Node, str]]] = None,
) -> dict[dace_sbs.Subset, dace_nodes.AccessNode]:
    """The function will split `node_to_split` into several smaller AccessNodes.

    How the split is performed is described by `split_description`, essentially it
    is a list of subsets that describes the sizes of the new AccessNodes.
    There are some special cases in this function.

    In some cases it might be possible to avoid to create an intermediate AccessNode.
    For example this happens when there is a Map that writes into `node_to_split`
    and a consumer to that node happens to be an AccessNode. In that case, the
    function might skip the creations of the intermediates.
    """

    if already_reconfigured_nodes is None:
        already_reconfigured_nodes = set()

    desc_to_split = node_to_split.desc(sdfg)
    assert desc_to_split.transient
    assert not gtx_transformations.utils.is_view(desc_to_split)
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

    # We will always create the new AccessNodes even if we might skip them. In that
    #  case we will simply remove them afterwards.
    new_access_nodes = _generate_desc_and_access_nodes_for_split(
        state, sdfg, node_to_split, assignment
    )

    _perform_node_split(
        state=state,
        sdfg=sdfg,
        node_to_split=node_to_split,
        new_access_nodes=new_access_nodes,
        assignment=assignment,
        allow_to_bypass_nodes=allow_to_bypass_nodes,
        already_reconfigured_nodes=already_reconfigured_nodes,
    )

    # If were allowed to bypass the split node, then we now clean up.
    #  This might be not the ideal solution.
    if allow_to_bypass_nodes:
        for split in split_description:
            if (split in new_access_nodes) and state.degree(new_access_nodes[split]) == 0:
                # TODO(phimuell): What to do about the data descriptor?
                new_access_nodes.pop(new_access_nodes[split])

    return new_access_nodes


def _perform_node_split(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    new_access_nodes: dict[dace_sbs.Subset, dace_nodes.AccessNode],
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]],
    allow_to_bypass_nodes: bool,
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

    handled_edges: set[dace_graph.MultiConnectorEdge] = set()
    for split in new_access_nodes:
        edges_to_relocate = assignment[split]
        new_access_node = new_access_nodes[split]
        assert state.degree(new_access_node) == 0

        assert all(
            edge_to_relocate.edge not in handled_edges for edge_to_relocate in edges_to_relocate
        )
        if allow_to_bypass_nodes and _can_use_bypass_version_of_node_spliter(edges_to_relocate):
            _perform_node_split_with_bypass_impl(
                state=state,
                sdfg=sdfg,
                node_to_split=node_to_split,
                edges_to_relocate=edges_to_relocate,
                already_reconfigured_nodes=already_reconfigured_nodes,
            )
        else:
            for edge_to_relocate in edges_to_relocate:
                assert edge_to_relocate.edge not in handled_edges
                _perform_node_split_impl(
                    state=state,
                    sdfg=sdfg,
                    node_to_split=node_to_split,
                    new_access_node=new_access_node,
                    split_description=split,
                    edge_to_relocate=edge_to_relocate,
                    already_reconfigured_nodes=already_reconfigured_nodes,
                )
        handled_edges.update(edesc.edge for edesc in edges_to_relocate)
    assert state.degree(node_to_split) == 0

    # Propagate the strides starting from the new access nodes.
    # NOTE: If the bypass version was used then the propagation has been done there
    #   already.
    for new_access_node in new_access_nodes.values():
        if state.degree(new_access_node) == 0:
            continue
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=state,
            outer_node=new_access_node,
        )

    # While we remove the AccessNode we do not remove the underlying data. We do this
    #  because it might be that there is another node that is still referring to
    #  it, which happens if we split across multiple states.
    state.remove_node(node_to_split)


def _perform_node_split_impl(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    new_access_node: dace_nodes.AccessNode,
    split_description: dace_sbs.Subset,
    edge_to_relocate: EdgeConnectionSpec,
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> dace_graph.MultiConnectorEdge:
    """Performs the actual split.

    In essence this function will create a new edge, based on but `edge_to_relocate`,
    but where `node_to_split` is replaced with `new_access_node`. Furthermore, the
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
        old_node=node_to_split,
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

        if isinstance(other_node, (dace_nodes.MapExit, dace_nodes.NestedSDFG)):
            # There is nothing special to do. In case of a nested SDFG, we will also have to do
            #  stride propagation, but we will postpone that.
            pass

        elif isinstance(other_node, dace_nodes.AccessNode):
            # Required by `reconfigure_dataflow_after_rerouting()`.
            if isinstance(other_node.desc(sdfg), dace_data.Scalar):
                subset_correction = None

        else:
            raise TypeError(f"Can not handle a producer of type '{type(other_node).__name__}'")

        gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
            is_producer_edge=is_producer_edge,
            new_edge=new_edge,
            ss_offset=subset_correction,
            state=state,
            sdfg=sdfg,
            old_node=node_to_split,
            new_node=new_access_node,
        )

    state.remove_edge(edge_to_relocate.edge)
    return new_edge


def _can_use_bypass_version_of_node_spliter(
    edges_to_relocate: Iterable[EdgeConnectionSpec],
) -> bool:
    producer_edges: list[EdgeConnectionSpec] = [
        desc for desc in edges_to_relocate if describes_incoming_edge(desc)
    ]
    if len(producer_edges) != 1:
        return False
    if not isinstance(get_other_node(producer_edges[0]), dace_nodes.AccessNode):
        return False
    return True


def _perform_node_split_with_bypass_impl(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    edges_to_relocate: set[EdgeConnectionSpec],
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> list[dace_graph.MultiConnectorEdge]:
    """Performs the splitting but the edge might go directly to the consumer.

    # TODO: Remove the producer edge, run reconfiguration, split operation.
    # TODO ADDING PRODUCER TO THE SET OF PROCESSED NODES

    """
    producer_edge_desc = next(edesc for edesc in edges_to_relocate)
    producer_edge = producer_edge_desc.edge
    data_producer: dace_nodes.Node = get_other_node(producer_edge)
    assert producer_edge.dst is node_to_split
    assert isinstance(data_producer, dace_nodes.AccessNode)

    old_producer_read = producer_edge.data.src_subset.min_element()
    old_producer_write = producer_edge.data.dst_subset.min_element()

    consumer_edges = [edesc for edesc in edges_to_relocate if edesc is not producer_edge]
    new_consumer_edges: list[dace_graph.MultiConnectorEdge] = []
    for consumer_edge_desc in consumer_edges:
        consumer_edge = consumer_edge_desc.edge
        consumer_node: dace_nodes.Node = consumer_edge.dst
        consumer_conn = consumer_edge.dst_conn

        # Index from where the consumer should start reading from the producer
        #  directly. But since it has gone through the intermediate AccessNode,
        #  the indexes are different and we have to compute them. At the end it
        #  is some kind of projection starting at what was read from the
        #  intermediate to the indexes where it has been written to, to the
        #  intermediate and finally from where it is originally coming from.
        #  We only do this projection for the start index, the end index
        #  is computed by adding the length to the start index.
        old_consumer_read = consumer_edge.data.src_subset.min_element()
        consumer_read_size = consumer_edge.data.src_subset.size()
        consumer_direct_read: list[tuple[dace_sym.SymbolicType, dace_sym.SymbolicType, int]] = []
        for i in range(len(old_producer_read)):
            old_producer_read_start = old_producer_read[i]
            old_consumer_read_start = old_consumer_read[i]
            old_producer_write_start = old_producer_write[i]
            transfer_size = consumer_read_size[i]

            consumer_direct_read_start = dace.symbolic.pystr_to_symbolic(
                f"({old_producer_read_start}) + (({old_consumer_read_start}) - ({old_producer_write_start}))",
                simplify=True,
            )
            # The `-1` is because the end is considered inclusive in DaCe.
            consumer_direct_read_end = dace.symbolic.pystr_to_symbolic(
                f"({consumer_direct_read_start}) + ({transfer_size}) - 1", simplify=True
            )
            consumer_direct_read.append((consumer_direct_read_start, consumer_direct_read_end, 1))

        new_consumer_direct_read_subset = dace_sbs.Range(consumer_direct_read)

        # Create a new edge that reads from the producer directly and remove the
        #  old edge.
        new_consumer_edge = state.add_edge(
            producer_edge.src,
            producer_edge.src_conn,
            consumer_edge.dst,
            consumer_edge.dst_conn,
            dace.Memlet(
                data=data_producer.data,
                subset=new_consumer_direct_read_subset,
                other_subset=consumer_edge.data.dst_subset,
                dynamic=consumer_edge.data.dynamic or producer_edge.data.dynamic,
            ),
        )
        state.remove_edge(consumer_edge)
        new_consumer_edges.append(new_consumer_edge)

        # If needed reconfigure the consumers since it now involves the
        #  original producer.
        #  The stride propagation is done after all edges have been updated.
        if (consumer_node, consumer_conn) not in already_reconfigured_nodes:
            already_reconfigured_nodes.add((consumer_node, consumer_conn))

            # The subset correct we have to apply to the consumer depends on the
            #  type of the consumer.
            if isinstance(data_producer.desc(sdfg), dace_data.Scalar):
                # This is required by the reconfigure function.
                consumer_subset_correction = None

            elif isinstance(consumer_node, dace_nodes.AccessNode):
                # Here we only have to consider the offset between the reading
                #  from the intermediate and where we write into it. The minus
                #  is because we have to subtract this value.
                consumer_subset_correction = [
                    dace.symbolic.pystr_to_symbolic(
                        f"-(({old_consumer_read_start}) - ({old_producer_write_start}))",
                        simplify=True,
                    )
                    for old_consumer_read_start, old_producer_write_start in zip(
                        old_consumer_read,
                        old_producer_write,
                        strict=True,
                    )
                ]
            elif isinstance(consumer_node, dace_nodes.MapEntry):
                # Things are different here, because the map ranges (and the
                #  offsets in the inner Memlets handle everything, Thus, in
                #  this case we have to correct the case that they now read
                #  from the producer instead of the intermediate.
                consumer_subset_correction = [
                    dace.symbolic.pystr_to_symbolic(
                        f"-(({old_producer_write_start}) - ({old_producer_read_start}))",
                        simplify=True,
                    )
                    for old_producer_read_start, old_producer_write_start in zip(
                        old_producer_read,
                        old_producer_write,
                        strict=True,
                    )
                ]

            elif isinstance(consumer_node, dace_nodes.NestedSDFG):
                # Since a NestedSDFG can only read from an AccessNode there is
                #  nothing to do, except for the stride propagation, which is
                #  done later.
                continue

            else:
                raise TypeError(
                    f"Can not correct a consumer of type '{type(consumer_node).__name__}'"
                )

            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=False,
                new_edge=new_consumer_edge,
                ss_offset=consumer_subset_correction,
                state=state,
                sdfg=sdfg,
                old_node=node_to_split,
                new_node=data_producer,
            )

    # We do not reconfigure the dataflow of the producer, because upstream of
    #  `data_producer` nothing has changed. In fact, we can not do it since
    #  there is also no edge. For that reason we do not add the producer to the
    #  `already_reconfigured_nodes` set, although one could do that.
    #  However, we have to propagate the strides, we have to do it here because on
    #  the outside it would only propagate from the new AccessNodes that we have
    #  bypassed.
    # TODO(phimuell): Find a way to avoid doing the propagation here, where the
    #   dataflow might be in some invalid state.
    state.remove_edge(producer_edge)
    processed_nsdfgs: set = set()
    for new_consumer_edge in new_consumer_edges:
        gtx_transformations.gt_map_strides_to_dst_nested_sdfg(
            outer_node=data_producer,
            edge=new_consumer_edge,
            sdfg=sdfg,
            state=state,
            processed_nsdfgs=processed_nsdfgs,
        )

    return new_consumer_edges


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
    edge_descriptions: Sequence[EdgeConnectionSpec],
    split_description: Sequence[dace_sbs.Subset],
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

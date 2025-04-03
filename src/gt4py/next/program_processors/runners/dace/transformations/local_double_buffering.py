# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import copy

import dace
from dace import (
    data as dace_data,
    dtypes as dace_dtypes,
    symbolic as dace_symbolic,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def gt_create_local_double_buffering(
    sdfg: dace.SDFG,
) -> int:
    """Modifies the SDFG such that point wise data dependencies are stable.

    Rule 3 of the ADR18, guarantees that if data is input and output to a map,
    then it must be a non transient array and it must only have point wise
    dependency. This means that every index that is read is also written by
    the same thread and no other thread reads or writes to the same location.
    However, because the dataflow inside a map is partially asynchronous
    it might happen if something is read multiple times, i.e. Tasklets,
    the data might already be overwritten.
    This function will scan the SDFG for potential cases and insert an
    access node to cache this read. This is essentially a double buffer, but
    it is not needed that the whole data is stored, but only the working set
    of a single thread.

    Todo:
        Simplify this function.
    """
    processed_maps = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        processed_maps += _create_local_double_buffering_non_recursive(nsdfg)
    return processed_maps


def _create_local_double_buffering_non_recursive(
    sdfg: dace.SDFG,
) -> int:
    """Implementation of the point wise transformation.

    This function does not handle nested SDFGs.
    """
    # First we call `EdgeConsolidation`, because of that we know that
    #  every incoming edge of a `MapEntry` refers to distinct data.
    #  We do this to simplify our implementation.
    edge_consolidation = dace_transformation.passes.ConsolidateEdges()
    edge_consolidation.apply_pass(sdfg, None)

    processed_maps = 0
    for state in sdfg.states():
        assert isinstance(state, dace.SDFGState)
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if not isinstance(node, dace_nodes.MapEntry):
                continue
            if scope_dict[node] is not None:
                continue
            inout_nodes = _check_if_map_must_be_handled(
                map_entry=node,
                state=state,
                sdfg=sdfg,
            )
            if inout_nodes is not None:
                processed_maps += _add_local_double_buffering_to(
                    map_entry=node,
                    inout_nodes=inout_nodes,
                    state=state,
                    sdfg=sdfg,
                )
    return processed_maps


def _add_local_double_buffering_to(
    inout_nodes: dict[str, tuple[dace_nodes.AccessNode, dace_nodes.AccessNode]],
    map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> int:
    """Adds the double buffering to `map_entry` for `inout_nodes`.

    The function assumes that there is only in incoming edge per data
    descriptor at the map entry. If the data is needed multiple times,
    then the distribution must be done inside the map.

    The function will now channel all reads to the data descriptor
    through an access node, this ensures that the read happens
    before the write.
    """
    processed_maps = 0
    for inout_node in inout_nodes.values():
        _add_local_double_buffering_to_single_data(
            map_entry=map_entry,
            inout_node=inout_node,
            state=state,
            sdfg=sdfg,
        )
        processed_maps += 1
    return processed_maps


def _add_local_double_buffering_to_single_data(
    inout_node: tuple[dace_nodes.AccessNode, dace_nodes.AccessNode],
    map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> None:
    """Adds the local double buffering for a single data."""
    map_exit: dace_nodes.MapExit = state.exit_node(map_entry)
    input_node, output_node = inout_node
    input_edges = state.edges_between(input_node, map_entry)
    output_edges = state.edges_between(map_exit, output_node)
    assert len(input_edges) == 1
    assert len(output_edges) == 1
    inner_read_edges = _get_inner_edges(input_edges[0], map_entry, state, False)
    inner_write_edges = _get_inner_edges(output_edges[0], map_exit, state, True)

    # Create the data container, we know that it is a scalar from the filtering
    assert _is_inner_read_a_scalar(inner_read_edges[0], state, sdfg)
    new_double_inner_buff_name: str = f"__inner_double_buffer_for_{input_node.data}"
    new_double_inner_buff_name, _ = sdfg.add_scalar(
        new_double_inner_buff_name,
        dtype=input_node.desc(sdfg).dtype,
        transient=True,
        storage=dace_dtypes.StorageType.Register,
        find_new_name=True,
    )
    new_double_inner_buff_node = state.add_access(new_double_inner_buff_name)

    # Now reroute the data flow through the new access node.
    reconfigured_dataflow: set[tuple[dace_nodes.Node, str]] = set()
    for old_inner_read_edge in inner_read_edges:
        # TODO(phimuell): Handle the case the memlet is "fancy"
        new_edge = state.add_edge(
            new_double_inner_buff_node,
            None,
            old_inner_read_edge.dst,
            old_inner_read_edge.dst_conn,
            dace.Memlet(
                data=new_double_inner_buff_name,
                subset="0",
                other_subset=copy.deepcopy(
                    old_inner_read_edge.data.get_dst_subset(old_inner_read_edge, state)
                ),
            ),
        )
        state.remove_edge(old_inner_read_edge)
        if (new_edge.dst, new_edge.dst_conn) not in reconfigured_dataflow:
            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=False,
                new_edge=new_edge,
                ss_offset=None,
                state=state,
                sdfg=sdfg,
                old_node=input_node,
                new_node=new_double_inner_buff_node,
            )
            reconfigured_dataflow.add((new_edge.dst, new_edge.dst_conn))

    # Now create a connection between the map entry and the intermediate node.
    state.add_edge(
        map_entry,
        inner_read_edges[0].src_conn,
        new_double_inner_buff_node,
        None,
        dace.Memlet(
            data=input_node.data,
            subset=copy.deepcopy(
                inner_read_edges[0].data.get_src_subset(inner_read_edges[0], state)
            ),
            other_subset="0",
        ),
    )

    # To really ensure that a read happens before a write, we have to sequence
    #  the read first. We do this by connecting the double buffer node with
    #  empty Memlets to the last row of nodes that writes to the global buffer.
    #  This is needed to handle the case that some other data path performs the
    #  write.
    # TODO(phimuell): Add a test that only performs this when it is really needed.
    for inner_write_edge in inner_write_edges:
        state.add_nedge(
            new_double_inner_buff_node,
            inner_write_edge.src,
            dace.Memlet(),
        )


def _is_inner_read_a_scalar(
    inner_reading_edge: dace_graph.MultiConnectorEdge,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> bool:
    """Tests if the read by `inner_reading_edge` is a scalar.

    A condition for being pointwise is that only one scalar is read, so this function
    checks this.

    Args:
        inner_reading_edge: The edge at the inside of the Map that reads from data
            that is connected to the MapEntry from the outside.
        state: The state in which we operate.
        sdfg: The SDFG we operate on.
    """
    new_double_inner_buff_shape_raw = dace_symbolic.overapproximate(
        inner_reading_edge.data.get_src_subset(inner_reading_edge, state).size()
    )

    # We can decay it into a scalar if all shapes have size 1.
    for proposed_dim_size in new_double_inner_buff_shape_raw:
        if (proposed_dim_size == 1) == False:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
            return False
    return True


def _check_if_map_must_be_handled_classify_adjacent_access_node(
    data_node: dace_nodes.AccessNode,
    sdfg: dace.SDFG,
    known_nodes: dict[str, dace_nodes.AccessNode],
) -> bool:
    """Internal function used by `_check_if_map_must_be_handled()` to classify nodes.

    If the function returns `True` it means that the input/output, does not
    violates an internal constraint, i.e. can be handled by
    `_ensure_that_map_is_pointwise()`. If appropriate the function will add the
    node to `known_nodes`. I.e. in case of a transient the function will return
    `True` but will not add it to `known_nodes`.
    """

    # This case is indicating that the `ConsolidateEdges` has not fully worked.
    #  Currently the transformation implementation assumes that this is the
    #  case, so we can not handle this case.
    # TODO(phimuell): Implement this case.
    if data_node.data in known_nodes:
        return False
    data_desc: dace_data.Data = data_node.desc(sdfg)

    # The conflict can only occur for global data, because transients
    #  are only written once.
    if data_desc.transient:
        return False

    # Currently we do not handle view, as they need to be traced.
    #  TODO(phimuell): Implement
    if gtx_transformations.utils.is_view(data_desc, sdfg):
        return False

    # TODO(phimuell): Check if there is a access node on the inner side, then we do not have to do it.

    # Now add the node to the list.
    assert all(data_node is not known_node for known_node in known_nodes.values())
    known_nodes[data_node.data] = data_node
    return True


def _get_inner_edges(
    outer_edge: dace.sdfg.graph.MultiConnectorEdge,
    scope_node: dace_nodes.MapExit | dace_nodes.MapEntry,
    state: dace.SDFG,
    outgoing_edge: bool,
) -> list[dace.sdfg.graph.MultiConnectorEdge]:
    """Gets the edges on the inside of a map."""
    if outgoing_edge:
        assert isinstance(scope_node, dace_nodes.MapExit)
        conn_name = outer_edge.src_conn[4:]
        return list(state.in_edges_by_connector(scope_node, connector="IN_" + conn_name))
    else:
        assert isinstance(scope_node, dace_nodes.MapEntry)
        conn_name = outer_edge.dst_conn[3:]
        return list(state.out_edges_by_connector(scope_node, connector="OUT_" + conn_name))


def _check_if_map_must_be_handled(
    map_entry: dace_nodes.MapEntry,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> None | dict[str, tuple[dace_nodes.AccessNode, dace_nodes.AccessNode]]:
    """Check if the map should be processed to uphold rule 3.

    Essentially the function will check if there is a potential read-write
    conflict. The function assumes that `ConsolidateEdges` has already run.

    If there is a possible data race the function will return a `dict`, that
    maps the name of the data to the access nodes that are used as input and
    output to the Map.

    Otherwise, the function returns `None`. It is, however, important that
    `None` does not means that there is no possible race condition. It could
    also means that the function that implements the buffering, i.e.
    `_ensure_that_map_is_pointwise()`, is unable to handle this case.

    Todo:
        Improve the function
    """
    map_exit: dace_nodes.MapExit = state.exit_node(map_entry)

    # Find all the data that is accessed. Views are resolved.
    input_datas: dict[str, dace_nodes.AccessNode] = {}
    output_datas: dict[str, dace_nodes.AccessNode] = {}

    # Determine which nodes are possible conflicting.
    for in_edge in state.in_edges(map_entry):
        if in_edge.data.is_empty():
            continue
        if not isinstance(in_edge.src, dace_nodes.AccessNode):
            # TODO(phiumuell): Figuring out what this case means
            continue
        if in_edge.dst_conn and not in_edge.dst_conn.startswith("IN_"):
            # TODO(phimuell): It is very unlikely that a Dynamic Map Range causes
            #   this particular data race, so we ignore it for the time being.
            continue
        if not _check_if_map_must_be_handled_classify_adjacent_access_node(
            data_node=in_edge.src,
            sdfg=sdfg,
            known_nodes=input_datas,
        ):
            continue
    for out_edge in state.out_edges(map_exit):
        if out_edge.data.is_empty():
            continue
        if not isinstance(out_edge.dst, dace_nodes.AccessNode):
            # TODO(phiumuell): Figuring out what this case means
            continue
        if not _check_if_map_must_be_handled_classify_adjacent_access_node(
            data_node=out_edge.dst,
            sdfg=sdfg,
            known_nodes=output_datas,
        ):
            continue

    # Double buffering is only needed if there inout arguments.
    inout_datas: dict[str, tuple[dace_nodes.AccessNode, dace_nodes.AccessNode]] = {
        dname: (input_datas[dname], output_datas[dname])
        for dname in input_datas
        if dname in output_datas
    }
    if len(inout_datas) == 0:
        return None

    # TODO(phimuell): What about the case that some data descriptor needs double
    #   buffering, but some do not?
    for inout_data_name in list(inout_datas.keys()):
        input_node, output_node = inout_datas[inout_data_name]
        input_edges = state.edges_between(input_node, map_entry)
        output_edges = state.edges_between(map_exit, output_node)
        assert (
            len(input_edges) == 1
        ), f"Expected a single connection between input node and map entry, but found {len(input_edges)}."
        assert (
            len(output_edges) == 1
        ), f"Expected a single connection between map exit and write back node, but found {len(output_edges)}."

        # If there is only one edge on the inside of the map, that goes into an
        #  AccessNode, then we assume it is double buffered.
        inner_read_edges = _get_inner_edges(input_edges[0], map_entry, state, False)
        if (
            len(inner_read_edges) == 1
            and isinstance(inner_read_edges[0].dst, dace_nodes.AccessNode)
            and not gtx_transformations.utils.is_view(inner_read_edges[0].dst, sdfg)
        ):
            inout_datas.pop(inout_data_name)
            continue

        # For being pointwise the read must only be a scalar, so by definition
        #  if this does not hold, we can ignore it.
        # NOTE: This is not always true, in case of nested Maps, for example the
        #   ones that are generated by `LoopBlocking`, but this transformation
        #   should run before that anyway and it is invalid to run auto optimizer
        #   multiple time on the SDFG.
        if not all(
            _is_inner_read_a_scalar(inner_read_edge, state, sdfg)
            for inner_read_edge in inner_read_edges
        ):
            inout_datas.pop(inout_data_name)
            continue

        inner_read_subsets = [
            inner_read_edge.data.get_src_subset(inner_read_edge, state)
            for inner_read_edge in inner_read_edges
        ]
        assert all(inner_read_subset is not None for inner_read_subset in inner_read_subsets)
        inner_write_subsets = [
            inner_write_edge.data.get_dst_subset(inner_write_edge, state)
            for inner_write_edge in _get_inner_edges(output_edges[0], map_exit, state, True)
        ]
        # TODO(phimuell): Also implement a check that the volume equals the size of the subset.
        assert all(inner_write_subset is not None for inner_write_subset in inner_write_subsets)

        # For being point wise the subsets must be compatible. The correct check would be:
        #   - The write sets are unique.
        #   - For every read subset there exists one matching write subset. It could
        #       be that there are many equivalent read subsets.
        #   - For every write subset there exists at least one matching read subset.
        #  The current implementation only checks if all are the same.
        # TODO(phimuell): Implement the real check.
        all_inner_subsets = inner_read_subsets + inner_write_subsets
        if not all(
            all_inner_subsets[0] == all_inner_subsets[i] for i in range(1, len(all_inner_subsets))
        ):
            return None

    if len(inout_datas) == 0:
        return None

    return inout_datas

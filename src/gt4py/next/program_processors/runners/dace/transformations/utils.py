# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Any, Container, Optional, Sequence, TypeVar, Union

import dace
from dace import data as dace_data, subsets as dace_sbs, symbolic as dace_sym
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils


_PassT = TypeVar("_PassT", bound=dace_ppl.Pass)


def gt_make_transients_persistent(
    sdfg: dace.SDFG,
    device: dace.DeviceType,
) -> dict[int, set[str]]:
    """
    Changes the lifetime of certain transients to `Persistent`.

    A persistent lifetime means that the transient is allocated only the very first
    time the SDFG is executed and only deallocated if the underlying `CompiledSDFG`
    object goes out of scope. The main advantage is, that memory must not be
    allocated every time the SDFG is run. The downside is that the SDFG can not be
    called by different threads.

    Args:
        sdfg: The SDFG to process.
        device: The device type.

    Returns:
        A `dict` mapping SDFG IDs to a set of transient arrays that
        were made persistent.

    Note:
        This function is based on a similar function in DaCe. However, the DaCe
        function does, for unknown reasons, also reset the `wcr_nonatomic` property,
        but only for GPU.
    """
    result: dict[int, set[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        fsyms: set[str] = nsdfg.free_symbols
        modify_lifetime: set[str] = set()
        not_modify_lifetime: set[str] = set()

        for state in nsdfg.states():
            scope_dict = state.scope_dict()
            for dnode in state.data_nodes():
                if dnode.data in not_modify_lifetime:
                    continue

                if dnode.data in nsdfg.constants_prop:
                    not_modify_lifetime.add(dnode.data)
                    continue

                desc = dnode.desc(nsdfg)
                if not desc.transient or type(desc) not in {dace.data.Array, dace.data.Scalar}:
                    not_modify_lifetime.add(dnode.data)
                    continue
                if desc.storage == dace.StorageType.Register:
                    not_modify_lifetime.add(dnode.data)
                    continue

                if desc.lifetime == dace.AllocationLifetime.External:
                    not_modify_lifetime.add(dnode.data)
                    continue

                # If the data is referenced inside a scope, such as a map, it might be possible
                #  that it is only used inside that scope. If we would make it persistent, then
                #  it would essentially be allocated outside and be shared among the different
                #  map iterations. So we can not make it persistent.
                #  The downside is, that we might have to perform dynamic allocation.
                if scope_dict[dnode] is not None:
                    not_modify_lifetime.add(dnode.data)
                    continue

                try:
                    # The symbols describing the total size must be a subset of the
                    #  free symbols of the SDFG (symbols passed as argument).
                    # NOTE: This ignores the renaming of symbols through the
                    #   `symbol_mapping` property of nested SDFGs.
                    if not set(map(str, desc.total_size.free_symbols)).issubset(fsyms):
                        not_modify_lifetime.add(dnode.data)
                        continue
                except AttributeError:  # total_size is an integer / has no free symbols
                    pass

                # Make it persistent.
                modify_lifetime.add(dnode.data)

        # Now setting the lifetime.
        result[nsdfg.cfg_id] = modify_lifetime - not_modify_lifetime
        for aname in result[nsdfg.cfg_id]:
            nsdfg.arrays[aname].lifetime = dace.AllocationLifetime.Persistent

    return result


def gt_find_constant_arguments(
    call_args: dict[str, Any],
    include: Optional[Container[str]] = None,
) -> dict[str, Any]:
    """Scans the calling arguments for compile time constants.

    The output of this function can be used as input to
    `gt_substitute_compiletime_symbols()`, which then removes these symbols.

    By specifying `include` it is possible to force the function to include
    additional arguments, that would not be matched otherwise. Importantly,
    their value is not checked.

    Args:
        call_args: The full list of arguments that will be passed to the SDFG.
        include: List of arguments that should be included.
    """
    if include is None:
        include = set()
    ret_value: dict[str, Any] = {}

    for name, value in call_args.items():
        if name in include or (gtx_dace_utils.is_field_symbol(name) and value == 1):
            ret_value[name] = value

    return ret_value


def is_accessed_downstream(
    start_state: dace.SDFGState,
    sdfg: dace.SDFG,
    data_to_look: str,
    reachable_states: Optional[dict[dace.SDFGState, set[dace.SDFGState]]],
    nodes_to_ignore: Optional[set[dace_nodes.AccessNode]] = None,
    states_to_ignore: Optional[set[dace.SDFGState]] = None,
) -> bool:
    """Scans for accesses to the data container `data_to_look`.

    The function will go through states that are reachable from `start_state`
    (included) and test if there is an AccessNode that _reads_ from `data_to_look`.
    It will return `True` the first time it finds such a node.

    The function will ignore all nodes that are listed in `nodes_to_ignore`.
    Furthermore, states listed in `states_to_ignore` will be ignored, i.e.
    handled as they did not exist.

    Args:
        start_state: The state where the scanning starts.
        sdfg: The SDFG on which we operate.
        data_to_look: The data that we want to look for.
        reachable_states: Maps an `SDFGState` to all `SDFGState`s that can be reached.
            If `None` it will be computed, but this is not recommended.
        nodes_to_ignore: Ignore these nodes.
        states_to_ignore: Ignore these states.

    Note:
        Currently, the function will not only ignore the states that are listed in
        `states_to_ignore`, but all that are reachable from any of these states.
        Thus care must be taken when this option is used. Furthermore, this behaviour
        is not intended and will change in further versions.
        `reachable_states` can be computed by using the `StateReachability` analysis
        pass from DaCe.

    Todo:
        - Modify the function such that it is no longer necessary to pass the
            `reachable_states` argument.
        - Fix the behaviour for `states_to_ignore`.
    """
    # After DaCe 1 switched to a hierarchical version of the state machine. Thus
    #  it is no longer possible in a simple way to traverse the SDFG. As a temporary
    #  solution we use the `StateReachability` pass. However, this has some issues,
    #  see the note about `states_to_ignore`.
    if reachable_states is None:
        state_reachability_pass = dace_analysis.StateReachability()
        reachable_states = state_reachability_pass.apply_pass(sdfg, None)[sdfg.cfg_id]
    else:
        # Ensures that the externally generated result was passed properly.
        assert all(
            isinstance(state, dace.SDFGState) and state.sdfg is sdfg for state in reachable_states
        )

    ign_dnodes: set[dace_nodes.AccessNode] = nodes_to_ignore or set()
    ign_states: set[dace.SDFGState] = states_to_ignore or set()

    # NOTE: We have to include `start_state`, however, we must also consider the
    #  data in `reachable_states` as immutable, so we have to do it this way.
    # TODO(phimuell): Go back to a trivial scan of the graph.
    if start_state not in reachable_states:
        # This can mean different things, either there was only one state to begin
        #  with or `start_state` is the last one. In this case the `states_to_scan`
        #  set consists only of the `start_state` because we have to process it.
        states_to_scan = {start_state}
    else:
        # Ensure that `start_state` is scanned.
        states_to_scan = reachable_states[start_state].union([start_state])

    # In the first version we explored the state machine and if we encountered a
    #  state in the ignore set we simply ignored it. This is no longer possible.
    #  Instead we will remove all states from the `states_to_scan` that are reachable
    #  from an ignored state. However, this is not the same as if we would explore
    #  the state machine (as we did before). Consider the following case:
    #
    #   (STATE_1) ------------> (STATE_2)
    #       |                       /\
    #       V                       |
    #   (STATE_3) ------------------+
    #
    #  Assume that `STATE_1` is the starting state and `STATE_3` is ignored.
    #  If we would explore the state machine, we would still scan `STATE_2`.
    #  However, because `STATE_2` is also reachable from `STATE_3` it will now be
    #  ignored. In most cases this should be fine, but we have to handle it.
    states_to_scan.difference_update(ign_states)
    for ign_state in ign_states:
        states_to_scan.difference_update(reachable_states.get(ign_state, set()))
    assert start_state in states_to_scan

    for downstream_state in states_to_scan:
        if downstream_state in ign_states:
            continue
        for dnode in downstream_state.data_nodes():
            if dnode.data != data_to_look:
                continue
            if dnode in ign_dnodes:
                continue
            if downstream_state.out_degree(dnode) != 0:
                return True  # There is a read operation
    return False


def is_reachable(
    start: Union[dace_nodes.Node, Sequence[dace_nodes.Node]],
    target: Union[dace_nodes.Node, Sequence[dace_nodes.Node]],
    state: dace.SDFGState,
) -> bool:
    """Explores the graph from `start` and checks if `target` is reachable.

    The exploration of the graph is done in a way that ignores the connector names.
    It is possible to pass multiple start nodes and targets. In case of multiple target nodes, the function returns True if any of them is reachable.

    Args:
        start: The node from where to start.
        target: The node to look for.
        state: The SDFG state on which we operate.
    """
    to_visit: list[dace_nodes.Node] = [start] if isinstance(start, dace_nodes.Node) else list(start)
    targets: set[dace_nodes.Node] = {target} if isinstance(target, dace_nodes.Node) else set(target)
    seen: set[dace_nodes.Node] = set()

    while to_visit:
        node = to_visit.pop()
        if node in targets:
            return True
        seen.add(node)
        to_visit.extend(oedge.dst for oedge in state.out_edges(node) if oedge.dst not in seen)

    return False


def is_view(
    node: Union[dace_nodes.AccessNode, dace_data.Data],
    sdfg: Optional[dace.SDFG] = None,
) -> bool:
    """Tests if `node` points to a view or not."""
    if isinstance(node, dace_nodes.AccessNode):
        assert sdfg is not None
        node_desc = node.desc(sdfg)
    else:
        assert isinstance(node, dace_data.Data)
        node_desc = node
    return isinstance(node_desc, dace_data.View)


def track_view(
    view: dace_nodes.AccessNode,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
) -> dace_nodes.AccessNode:
    """Find the original data of a View.

    Given the View `view`, the function will trace the view back to the original
    access node. For convenience, if `view` is not a `View` the argument will be
    returned.

    Args:
        view: The view that should be traced.
        state: The state in which we operate.
        sdfg: The SDFG on which we operate.
    """

    # Test if it is a view at all, if not return the passed node as source.
    if not is_view(view, sdfg):
        return view

    # First determine if the view is used for reading or writing.
    curr_edge = dace.sdfg.utils.get_view_edge(state, view)
    if curr_edge is None:
        raise RuntimeError(f"Failed to determine the direction of the view '{view}'.")
    if curr_edge.dst_conn == "views":
        # The view is used for reading.
        next_node = lambda curr_edge: curr_edge.src  # noqa: E731
    elif curr_edge.src_conn == "views":
        # The view is used for writing.
        next_node = lambda curr_edge: curr_edge.dst  # noqa: E731
    else:
        raise RuntimeError(f"Failed to determine the direction of the view '{view}' | {curr_edge}.")

    # Now trace the view back.
    org_view = view
    view = next_node(curr_edge)
    while is_view(view, sdfg):
        curr_edge = dace.sdfg.utils.get_view_edge(state, view)
        if curr_edge is None:
            raise RuntimeError(f"View tracing of '{org_view}' failed at note '{view}'.")
        view = next_node(curr_edge)
    return view


def reroute_edge(
    is_producer_edge: bool,
    current_edge: dace_graph.MultiConnectorEdge,
    ss_offset: Sequence[dace_sym.SymExpr],
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    old_node: dace_nodes.AccessNode,
    new_node: dace_nodes.AccessNode,
) -> dace_graph.MultiConnectorEdge:
    """Performs the rerouting of the edge.

    Essentially the function create a new edge that replaces `old_node` with `new_node`.
    Depending on the value of `is_producer_edge` the behaviour is slightly different.

    If `is_producer_edge` is `True` then the function expects that `current_edge`
    is an edge that _ends_ at `old_node`. It will then create a new edge that starts
    at the source of `current_edge` but will end in `new_node` and have a similar
    Memlet than `current_edge`.
    If `is_producer_edge` is `False` then it assumes `current_edge` starts at
    `old_node`. It will then create a new edge that starts at `new_node` and has a
    similar Memlet than the original one.
    To modify the subsets where `new_node` is read/written the `ss_offset` is
    provided, this value is added to the current subset.

    It is important that the function will **not** do the following things:
    - Remove the old edge, i.e. `producer_edge`.
    - Modify the data flow at the other side of the edge, to account of the change
        from `old_node` to `new_node`. If this should be modified see
        `reconfigure_dataflow_after_rerouting()`.

    The function returns the new edge.

    Args:
        is_producer_edge: Indicates how to interpret `current_edge`.
        current_edge: The current edge that should be replaced.
        ss_offset: Offset that describes how much to shift writes and reads,
            that were previously associated with `old_node`.
        state: The state in which we operate.
        sdfg: The SDFG on which we operate on.
        old_node: The old that should be replaced in `current_edge`.
        new_node: The new node that should be used instead of `old_node`.
    """
    current_memlet: dace.Memlet = current_edge.data
    if is_producer_edge:
        # NOTE: See the note in `_reconfigure_dataflow()` why it is not save to
        #  use the `get_{dst, src}_subset()` function, although it would be more
        #  appropriate.
        assert current_edge.dst is old_node
        current_subset: dace_sbs.Range = current_memlet.dst_subset
        new_src = current_edge.src
        new_src_conn = current_edge._src_conn
        new_dst = new_node
        new_dst_conn = None
        assert current_edge.dst_conn is None
    else:
        assert current_edge.src is old_node
        current_subset = current_memlet.src_subset
        new_src = new_node
        new_src_conn = None
        new_dst = current_edge.dst
        new_dst_conn = current_edge.dst_conn
        assert current_edge.src_conn is None

    # If the subset we care about, which is always on the `old_node` side, was not
    #  specified we assume that the whole `old_node` has been written.
    # TODO(edopao): Fix lowering that this does not happens, it happens for example
    #  in `tests/next_tests/integration_tests/feature_tests/ffront_tests/
    #  test_execution.py::test_docstring`.
    if current_subset is None:
        current_subset = dace_sbs.Range.from_array(old_node.desc(sdfg))

    # This is the new Memlet, that we will use. We copy it from the original
    #  Memlet and modify it later.
    new_memlet: dace.Memlet = dace.Memlet.from_memlet(current_memlet)

    # Because we operate on the `subset` and `other_subset` properties directly
    #  we do not need to distinguish between the different directions. Also
    #  in both cases the offset is the same.
    if new_memlet.data == old_node.data:
        new_memlet.data = new_node.data
        new_subset = current_subset.offset_new(ss_offset, negative=False)
        new_memlet.subset = new_subset
    else:
        new_subset = current_subset.offset_new(ss_offset, negative=False)
        new_memlet.other_subset = new_subset

    new_edge = state.add_edge(
        new_src,
        new_src_conn,
        new_dst,
        new_dst_conn,
        new_memlet,
    )
    assert (  # Ensure that the edge has the right direction.
        new_subset is new_edge.data.dst_subset
        if is_producer_edge
        else new_subset is new_edge.data.src_subset
    )
    return new_edge


def reconfigure_dataflow_after_rerouting(
    is_producer_edge: bool,
    new_edge: dace_graph.MultiConnectorEdge,
    ss_offset: Sequence[dace_sym.SymExpr] | None,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    old_node: dace_nodes.AccessNode,
    new_node: dace_nodes.AccessNode,
) -> None:
    """Modify the data flow associated to `new_edge`.

    While the `reroute_edge()` function creates a new edge, it does not modify
    the dataflow at the other side of the connection, this is done by this
    function.

    Depending on the value of `is_producer_edge` the function will either modify
    the source of `new_edge` (`True`) or it will modify the data flow associated
    to the destination of `new_edge` (`False`).
    Furthermore, the specific actions depends on what kind of node is on the other
    side. However, essentially the function will modify it to account for the
    change from `old_node` to `new_node`.

    When using this function the user has to consider the following things:
    - It is the caller's responsibility to ensure that this function is not called
        multiple times on the same producer target.
    - The function will not propagate new strides, this must be done separately.
    - The function will not modify the ranges of Maps.

    Args:
        is_producer_edge: If `True` then the source of `new_edge` is processed,
            if `False` then the destination part of `new_edge` is processed.
        new_edge: The newly created edge, essentially the return value of
            `reroute__edge()`.
        ss_offsets: Offset that describes how much to shift subsets associated
            to `old_node` to account that they are now associated to `new_node`.
            If `None` then `new_node` needs to be a scalar.
        state: The state in which we operate.
        sdfg: The SDFG on which we operate.
        old_node: The old that was involved in the old, rerouted, edge.
        new_node: The new node that should be used instead of `old_node`.
    """
    other_node = new_edge.src if is_producer_edge else new_edge.dst

    if isinstance(other_node, dace_nodes.AccessNode):
        # There is nothing here to do.
        pass

    elif isinstance(other_node, dace_nodes.Tasklet):
        # A very obscure case, but I think it might happen, but as in the AccessNode
        #  case there is nothing to do here.
        pass

    elif isinstance(other_node, (dace_nodes.MapExit | dace_nodes.MapEntry)):
        # Essentially, we have to propagate the change that everything that
        #  refers to `old_node` should now refer to `new_node`, In addition we also
        #  have to modify the subsets, depending on the direction of the new edge
        #  either the source or destination subset.
        # NOTE: Also for this case we have to propagate the strides, for the case
        #   that a NestedSDFG is inside the map, but this is done externally.
        assert (
            isinstance(other_node, dace_nodes.MapExit)
            if is_producer_edge
            else isinstance(other_node, dace_nodes.MapEntry)
        )
        if ss_offset is None:
            if not isinstance(new_node.desc(sdfg), dace_data.Scalar):
                raise TypeError(
                    "Passed 'None' as 'ss_offset' but 'new_node' ({new_node.data}) is not a scalar."
                )

        for memlet_tree in state.memlet_tree(new_edge).traverse_children(include_self=False):
            edge_to_adjust = memlet_tree.edge
            memlet_to_adjust = edge_to_adjust.data

            # If needed modify the association of the Memlet.
            if memlet_to_adjust.data == old_node.data:
                memlet_to_adjust.data = new_node.data

            if ss_offset is None:
                # Scalar modification
                if is_producer_edge:
                    memlet_to_adjust.dst_subset = "0"
                else:
                    memlet_to_adjust.src_subset = "0"

            else:
                # NOTE: Actually we should use the `get_{src, dst}_subset()` functions,
                #  see https://github.com/spcl/dace/issues/1703. However, we can not
                #  do that because the SDFG is currently in an invalid state. So
                #  we have to call the properties and hope that it works.
                subset_to_adjust = (
                    memlet_to_adjust.dst_subset if is_producer_edge else memlet_to_adjust.src_subset
                )
                assert subset_to_adjust is not None
                subset_to_adjust.offset(ss_offset, negative=False)

    elif isinstance(other_node, dace_nodes.NestedSDFG):
        # We have obviously to adjust the strides, however, this is done outside
        #  this function.
        # TODO(phimuell): Look into the implication that we not necessarily pass
        #  the full array, but essentially slice a bit.
        pass

    else:
        # As we encounter them we should handle them case by case.
        raise NotImplementedError(
            f"The case for '{type(other_node).__name__}' has not been implemented."
        )


def find_upstream_nodes(
    start: dace_nodes.Node,
    state: dace.SDFGState,
    start_connector: Optional[str] = None,
    limit_node: Optional[dace_nodes.Node] = None,
) -> set[dace_nodes.Node]:
    """Finds all upstream nodes, i.e. all producers, of `start`.

    Args:
        start: Start the exploring from this node.
        state: The state in which it should be explored.
        start_connector: If given then only consider edges that terminate
            in this connector, otherwise consider all incoming edges.
        limit_node: Consider this node as "limiting wall", i.e. do not explore
            beyond it.
    """
    seen: set[dace_nodes.Node] = set()

    to_visit = [
        iedge.src
        for iedge in (
            state.in_edges(start)
            if start_connector is None
            else state.in_edges_by_connector(start, start_connector)
        )
        if iedge.src is not limit_node
    ]

    while len(to_visit) != 0:
        node = to_visit.pop()
        if node in seen:
            continue
        seen.add(node)
        to_visit.extend(iedge.src for iedge in state.in_edges(node) if iedge.src is not limit_node)

    return seen

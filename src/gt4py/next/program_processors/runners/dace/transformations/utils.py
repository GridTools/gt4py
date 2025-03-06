# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Any, Container, Optional, Sequence, Union

import dace
from dace import data as dace_data
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils


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
    It is possible to pass multiple start nodes and targets.

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

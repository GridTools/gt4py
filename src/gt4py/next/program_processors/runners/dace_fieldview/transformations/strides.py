# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Iterable, Optional, TypeAlias

import dace
from dace import data as dace_data
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


PropagatedStrideRecord: TypeAlias = tuple[str, dace_nodes.NestedSDFG]
"""Record of a stride that has been propagated into a NestedSDFG.

The type combines the NestedSDFG into which the strides were already propagated
and the data within that NestedSDFG to which we have propagated the data,
which is the connector name on the NestedSDFG.
We need the NestedSDFG because we have to know what was already processed,
however, we also need the name within because of aliasing, i.e. a data
descriptor on the outside could be mapped to multiple data descriptors
inside the NestedSDFG.
"""


def gt_change_transient_strides(
    sdfg: dace.SDFG,
    gpu: bool,
) -> dace.SDFG:
    """Modifies the strides of transients.

    The function will analyse the access patterns and set the strides of
    transients in the optimal way.
    The function should run after all maps have been created.

    Args:
        sdfg: The SDFG to process.
        gpu: If the SDFG is supposed to run on the GPU.

    Note:
        Currently the function will not scan the access pattern. Instead it will
        either use FORTRAN order for GPU or C order (which is assumed to be the
        default, so it is a no ops).

    Todo:
        - Implement the estimation correctly.
    """
    # TODO(phimeull): Implement this function correctly.

    # We assume that by default we have C order which is already correct,
    #  so in this case we have a no ops
    if not gpu:
        return sdfg

    for nsdfg in sdfg.all_sdfgs_recursive():
        _gt_change_transient_strides_non_recursive_impl(nsdfg)


def _gt_change_transient_strides_non_recursive_impl(
    sdfg: dace.SDFG,
) -> None:
    """Set optimal strides of all transients in the SDFG.

    The function will look for all top level transients, see `_gt_find_toplevel_data_accesses()`
    and set their strides such that the access is optimal, see Note. The function
    will also run `gt_propagate_strides_of()` to propagate the strides into nested SDFGs.

    This function should never be called directly but always through
    `gt_change_transient_strides()`!

    Note:
        Currently the function just reverses the strides of the data descriptor
        it processes. Since DaCe generates `C` order by default this lead to
        FORTRAN order, which is (for now) sufficient to optimize the memory
        layout to GPU.

    Todo:
        Make this function more intelligent to analyse the access pattern and then
        figuring out the best order.
    """
    # NOTE: Processing the transient here is enough. If we are inside a
    #   NestedSDFG then they were handled before on the level above us.
    top_level_transients_and_their_accesses = _gt_find_toplevel_data_accesses(
        sdfg=sdfg,
        only_transients=True,
        only_arrays=True,
    )
    for top_level_transient, accesses in top_level_transients_and_their_accesses.items():
        desc: dace_data.Array = sdfg.arrays[top_level_transient]

        # Setting the strides only make sense if we have more than two dimensions
        ndim = len(desc.shape)
        if ndim <= 1:
            continue

        # We assume that everything is in C order initially, to get FORTRAN order
        #  we simply have to reverse the order.
        new_stride_order = list(range(ndim))
        desc.set_strides_from_layout(*new_stride_order)

        # Now we have to propagate the changed strides. Because we already have
        #  collected all the AccessNodes we are using the
        #  `gt_propagate_strides_from_access_node()` function, but we have to
        #  create `processed_nsdfg` set already outside here.
        #  Furthermore, the same comment as above apply, we do not have to
        #  propagate the non-transients, because they either come from outside,
        #  or they were already handled in the levels above, where they were
        #  defined and then propagated down.
        processed_nsdfgs: set[dace_nodes.NestedSDFG] = set()
        for state, access_node in accesses:
            gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=access_node,
                processed_nsdfgs=processed_nsdfgs,
            )


def gt_propagate_strides_of(
    sdfg: dace.SDFG,
    data_name: str,
) -> None:
    """Propagates the strides of `data_name` within the whole SDFG.

    This function will call `gt_propagate_strides_from_access_node()` for every
    AccessNode that refers to `data_name`. It will also make sure that a descriptor
    inside a NestedSDFG is only processed once.

    Args:
        sdfg: The SDFG on which we operate.
        data_name: Name of the data descriptor that should be handled.
    """

    # Defining it here ensures that we will not enter an NestedSDFG multiple times.
    processed_nsdfgs: set[PropagatedStrideRecord] = set()

    for state in sdfg.states():
        for dnode in state.data_nodes():
            if dnode.data != data_name:
                continue
            gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=dnode,
                processed_nsdfgs=processed_nsdfgs,
            )


def gt_propagate_strides_from_access_node(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    outer_node: dace_nodes.AccessNode,
    processed_nsdfgs: Optional[set[PropagatedStrideRecord]] = None,
) -> None:
    """Propagates the stride of `outer_node` along all adjacent edges of `outer_node`.

    The function will propagate the strides of the data descriptor `outer_node`
    refers to along all adjacent edges of `outer_node`. If one of these edges
    leads to a NestedSDFG then the function will modify the strides of data
    descriptor within to match the strides on the outside. The function will then
    recursively process NestedSDFG.

    It is important that this function will only handle the NestedSDFGs that are
    reachable from `outer_node`. To fully propagate the strides the
    `gt_propagate_strides_of()` should be used.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that reads from the data node, the nested SDFG is expected as the destination.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of NestedSDFG that were already processed and will be ignored.
            Only specify when you know what your are doing.
        propagate_along_dataflow: Determine the direction of propagation. If `True` the
            function follows the dataflow.
    """
    if processed_nsdfgs is None:
        # For preventing the case that nested SDFGs are handled multiple time.
        #  TODO: It certainly happens if a node is input and output, but are there other cases?
        processed_nsdfgs = set()

    for in_edge in state.in_edges(outer_node):
        gt_map_strides_to_src_nested_sdfg(
            sdfg=sdfg,
            state=state,
            edge=in_edge,
            outer_node=outer_node,
            processed_nsdfgs=processed_nsdfgs,
        )
    for out_edge in state.out_edges(outer_node):
        gt_map_strides_to_dst_nested_sdfg(
            sdfg=sdfg,
            state=state,
            edge=out_edge,
            outer_node=outer_node,
            processed_nsdfgs=processed_nsdfgs,
        )


def gt_map_strides_to_dst_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.Edge,
    outer_node: dace.nodes.AccessNode,
    processed_nsdfgs: Optional[set[PropagatedStrideRecord]] = None,
) -> None:
    """Propagates the strides of `outer_node` along `edge` along the dataflow.

    For more information see the description of `_gt_map_strides_to_nested_sdfg_src_dst().
    However it is recommended to use `gt_propagate_strides_of()` directly.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that writes to the data node, the nested SDFG is expected as the source.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of Nested SDFG that were already processed. Only specify when
            you know what your are doing.
    """
    _gt_map_strides_to_nested_sdfg_src_dst(
        sdfg=sdfg,
        state=state,
        edge=edge,
        outer_node=outer_node,
        processed_nsdfgs=processed_nsdfgs,
        propagate_along_dataflow=True,
    )


def gt_map_strides_to_src_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.Edge,
    outer_node: dace.nodes.AccessNode,
    processed_nsdfgs: Optional[set[PropagatedStrideRecord]] = None,
) -> None:
    """Propagates the strides of `outer_node` along `edge` against the dataflow.

    For more information see the description of `_gt_map_strides_to_nested_sdfg_src_dst().
    However it is recommended to use `gt_propagate_strides_of()` directly.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that writes to the data node, the nested SDFG is expected as the source.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of Nested SDFG that were already processed. Only specify when
            you know what your are doing.
    """
    _gt_map_strides_to_nested_sdfg_src_dst(
        sdfg=sdfg,
        state=state,
        edge=edge,
        outer_node=outer_node,
        processed_nsdfgs=processed_nsdfgs,
        propagate_along_dataflow=False,
    )


def _gt_map_strides_to_nested_sdfg_src_dst(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
    outer_node: dace.nodes.AccessNode,
    processed_nsdfgs: Optional[set[PropagatedStrideRecord]],
    propagate_along_dataflow: bool,
) -> None:
    """Propagates the stride of `outer_node` along `edge`.

    The function will follow `edge`, the direction depends on the value of
    `propagate_along_dataflow` and propagate the strides of `outer_node`
    into every NestedSDFG that is reachable by following `edge`.

    When the function encounters a NestedSDFG it will determine the the data
    descriptor `outer_node` refers on the inside of the NestedSDFG.
    It will then replace the stride of the inner descriptor with the ones
    of the outside. Afterwards it will recursively propagates the
    stride inside the NestedSDFG.
    During this propagation the function will follow any edges.

    If the function reaches a NestedSDFG that is listed inside `processed_nsdfgs`
    then it will be skipped. NestedSDFGs that have been processed will be added
    to the `processed_nsdfgs`.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that reads from the data node, the nested SDFG is expected as the destination.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of Nested SDFG that were already processed and will be ignored.
            Only specify when you know what your are doing.
        propagate_along_dataflow: Determine the direction of propagation. If `True` the
            function follows the dataflow.

    Note:
        A user should not use this function directly, instead `gt_propagate_strides_of()`,
        `gt_map_strides_to_src_nested_sdfg()` (`propagate_along_dataflow == `False`)
        or `gt_map_strides_to_dst_nested_sdfg()` (`propagate_along_dataflow == `True`)
        should be used.

    Todo:
        Try using `MemletTree` for the propagation.
    """
    # If `processed_nsdfg` is `None` then this is the first call. We will now
    #  allocate the `set` and pass it as argument to all recursive calls, this
    #  ensures that the `set` is the same everywhere.
    if processed_nsdfgs is None:
        processed_nsdfgs = set()

    if propagate_along_dataflow:
        # Propagate along the dataflow or forward, so we are interested at the `dst` of the edge.
        ScopeNode = dace_nodes.MapEntry

        def get_node(edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]) -> dace_nodes.Node:
            return edge.dst

        def get_inner_data(edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]) -> str:
            return edge.dst_conn

        def next_edges_by_connector(
            state: dace.SDFGState, edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]
        ) -> list[dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]]:
            if edge.dst_conn is None or not edge.dst_conn.startswith("IN_"):
                return []
            return list(state.out_edges_by_connector(edge.dst, "OUT_" + edge.dst_conn[3:]))

    else:
        # Propagate against the dataflow or backward, so we are interested at the `src` of the edge.
        ScopeNode = dace_nodes.MapExit

        def get_node(edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]) -> dace_nodes.Node:
            return edge.src

        def get_inner_data(edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]) -> str:
            return edge.src_conn

        def next_edges_by_connector(
            state: dace.SDFGState, edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]
        ) -> list[dace.sdfg.graph.MultiConnectorEdge[dace.Memlet]]:
            return list(state.in_edges_by_connector(edge.src, "IN_" + edge.src_conn[4:]))

    if isinstance(get_node(edge), ScopeNode):
        for next_edge in next_edges_by_connector(state, edge):
            _gt_map_strides_to_nested_sdfg_src_dst(
                sdfg=sdfg,
                state=state,
                edge=next_edge,
                outer_node=outer_node,
                processed_nsdfgs=processed_nsdfgs,
                propagate_along_dataflow=propagate_along_dataflow,
            )

    elif isinstance(get_node(edge), dace.nodes.NestedSDFG):
        nsdfg_node = get_node(edge)
        inner_data = get_inner_data(edge)
        process_record = (inner_data, nsdfg_node)

        if process_record in processed_nsdfgs:
            # We already handled this NestedSDFG and the inner data.
            return

        # Mark this nested SDFG as processed.
        processed_nsdfgs.add(process_record)

        # Now set the stride of the data descriptor inside the nested SDFG to
        #  the ones it has outside.
        _gt_map_strides_to_nested_sdfg(
            nsdfg_node=nsdfg_node,
            inner_data=inner_data,
            edge_data=edge.data,
            outer_strides=outer_node.desc(sdfg).strides,
        )

        # Because the function call above if not recursive we have now to scan the
        #  propagate the change into the nested SDFG. Using
        #  `_gt_find_toplevel_data_accesses()` is a bit overkill, but allows for a
        #  more uniform processing.
        # TODO(phimuell): Instead of scanning every level for every data we modify
        #   we should scan the whole SDFG once and then reuse this information.
        accesses_in_nested_sdfg = _gt_find_toplevel_data_accesses(
            sdfg=nsdfg_node.sdfg,
            only_transients=False,  # Because on the nested levels they are globals.
            only_arrays=True,
        )
        for nested_state, nested_access in accesses_in_nested_sdfg.get(inner_data, list()):
            # We have to use `gt_propagate_strides_of()` here because we have to
            #  handle its entirety. We could wait until the other branch processes
            #  the nested SDFG, but this might not work, so let's do it fully now.
            gt_propagate_strides_from_access_node(
                sdfg=nsdfg_node.sdfg,
                state=nested_state,
                outer_node=nested_access,
                processed_nsdfgs=processed_nsdfgs,
            )


def _gt_map_strides_to_nested_sdfg(
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_data: str,
    edge_data: dace.Memlet,
    outer_strides: Iterable[int | dace.symbolic.SymExpr],
) -> None:
    # TODO(phimuell/edopao): Refactor this function.
    # We need to propagate the strides inside the nested SDFG on the global arrays
    new_strides = tuple(
        stride
        for stride, to_map_size in zip(
            outer_strides,
            edge_data.subset.size(),
            strict=True,
        )
        if to_map_size != 1
    )
    inner_desc = nsdfg_node.sdfg.arrays[inner_data]
    assert not inner_desc.transient

    if isinstance(inner_desc, dace.data.Scalar):
        assert len(new_strides) == 0
        return

    assert isinstance(inner_desc, dace.data.Array)
    if all(isinstance(inner_stride, dace.symbol) for inner_stride in inner_desc.strides):
        for inner_stride, outer_stride in zip(inner_desc.strides, new_strides, strict=True):
            nsdfg_node.symbol_mapping[inner_stride.name] = outer_stride
    else:
        assert len(inner_desc.shape) == len(new_strides)
        inner_desc.set_shape(inner_desc.shape, new_strides)

        new_strides_symbols: list[dace.symbol] = functools.reduce(
            lambda acc, itm: (acc + list(itm.free_symbols))  # type: ignore[union-attr]
            if dace.symbolic.issymbolic(itm)
            else acc,
            new_strides,
            [],
        )
        new_strides_free_symbols = {
            sym for sym in new_strides_symbols if sym.name not in nsdfg_node.sdfg.symbols
        }
        for sym in new_strides_free_symbols:
            nsdfg_node.sdfg.add_symbol(sym.name, sym.dtype)
            nsdfg_node.symbol_mapping[sym.name] = sym


def _gt_find_toplevel_data_accesses(
    sdfg: dace.SDFG,
    only_transients: bool,
    only_arrays: bool = False,
) -> dict[str, list[tuple[dace.SDFGState, dace_nodes.AccessNode]]]:
    """Find all data that is accessed on the top level.

    The function will scan the SDFG, ignoring nested one, and return the
    name of all data that only have AccessNodes on the top level. In data
    is found that has an AccessNode on both the top level and in a nested
    scope and error is generated.
    By default the function will return transient and non transient data,
    however, if `only_transients` is `True` then only transient data will
    be returned.
    Furthermore, the function will ignore an access in the following cases:
    - The AccessNode refers to data that is a register.
    - The AccessNode refers to a View.

    Args:
        sdfg: The SDFG to process.
        only_transients: If `True` only include transients.
        only_arrays: If `True`, defaults to `False`, only arrays are returned.

    Returns:
        A `dict` that maps the name of a data container, that should be processed
        to a list of tuples containing the state where the AccessNode was found
        and the node.
    """
    # List of data that is accessed on the top level and all its access node.
    top_level_data: dict[str, list[tuple[dace.SDFGState, dace_nodes.AccessNode]]] = dict()

    # List of all data that were found not on top level.
    not_top_level_data: set[str] = set()

    for state in sdfg.states():
        scope_dict = state.scope_dict()
        for dnode in state.data_nodes():
            data: str = dnode.data
            if scope_dict[dnode] is not None:
                # The node was not found on the top level. So we can ignore it.
                # We also check if it was ever found on the top level, this should
                #  not happen, as everything should go through Maps. But some strange
                #  DaCe transformation might do it.
                assert (
                    data not in top_level_data
                ), f"Found {data} on the top level and inside a scope."
                not_top_level_data.add(data)
                continue

            elif data in top_level_data:
                # The data is already known to be in top level data, so we must add the
                #  AccessNode to the list of known nodes. But nothing else.
                top_level_data[data].append((state, dnode))
                continue

            elif gtx_transformations.util.is_view(dnode, sdfg):
                # The AccessNode refers to a View so we ignore it anyway
                # TODO(phimuell/edopao): Should the function return them?
                continue

            # We have found a new data node that is on the top node and is unknown.
            assert (
                data not in not_top_level_data
            ), f"Found {data} on the top level and inside a scope."
            desc: dace_data.Data = dnode.desc(sdfg)

            # Check if we only accept arrays
            if only_arrays and not isinstance(desc, dace_data.Array):
                continue

            # For now we ignore registers.
            #  We do this because register are allocated on the stack, so the compiler
            #  has all information and should organize the best thing possible.
            # TODO(phimuell): verify this.
            elif desc.storage is dace.StorageType.Register:
                continue

            # We are only interested in transients
            if only_transients and (not desc.transient):
                continue

            # Now create the new entry in the list and record the AccessNode.
            top_level_data[data] = [(state, dnode)]
    return top_level_data

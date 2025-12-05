# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, TypeAlias

import dace
from dace import data as dace_data
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    sdfg_args as gtx_dace_args,
    transformations as gtx_transformations,
)


PropagatedStrideRecord: TypeAlias = tuple[str, dace_nodes.NestedSDFG]
"""Record of a stride that has been propagated into a NestedSDFG.

The type combines the NestedSDFG into which the strides were already propagated
and the data within that NestedSDFG to which we have propagated the strides,
which is the connector name on the NestedSDFG.
We need the NestedSDFG because we have to know what was already processed,
however, we also need the inner array name because of aliasing, i.e. a data
descriptor on the outside could be mapped to multiple data descriptors
inside the NestedSDFG.
"""


def gt_change_strides(
    sdfg: dace.SDFG,
    gpu: bool,
) -> dace.SDFG:
    """Modifies the strides of transients.

    The function will analyse the access patterns and set the strides of
    transients in the optimal way.
    The function should run after all maps have been created.

    After the strides have been adjusted the function will also propagate
    the strides into nested SDFG, see `gt_propagate_strides_of()` for more.
    Args:
        sdfg: The SDFG to process.
        gpu: If the SDFG is supposed to run on the GPU.

    Note:
        Currently the function will not scan the access pattern. Instead it will
        either use FORTRAN order for GPU or C order. This function needs to be called
        for both CPU and GPU to handle strides of memlets inside nested SDFGs.

    Todo:
        - Implement the estimation correctly.
    """
    # TODO(phimeull): Implement this function correctly.

    for nsdfg in sdfg.all_sdfgs_recursive():
        _gt_change_strides_non_recursive_impl(nsdfg, gpu)


def _gt_change_strides_non_recursive_impl(
    sdfg: dace.SDFG,
    gpu: bool,
) -> None:
    """Set optimal strides of all access nodes in the SDFG.

    The function will look for all top level access node, see `_gt_find_toplevel_data_accesses()`
    and set their strides such that the access is optimal, see Note. The function
    will also run `gt_propagate_strides_of()` to propagate the strides into nested SDFGs.

    This function should never be called directly but always through `gt_change_strides()`!

    Note:
        Currently the function just reverses the strides of the data descriptor
        of transient access nodes it processes. Since DaCe generates `C` order by default
        this lead to FORTRAN order, which is (for now) sufficient to optimize the memory
        layout to GPU.

    Todo:
        Make this function more intelligent to analyse the access pattern and then
        figuring out the best order.
    """
    # NOTE: We have to process all access nodes (transient and globals). If we are inside a
    #   NestedSDFG then they were handled before on the level above us.
    top_level_transients_and_their_accesses = _gt_find_toplevel_data_accesses(
        sdfg=sdfg,
        only_transients=False,
        only_arrays=True,
    )
    for top_level_transient, accesses in top_level_transients_and_their_accesses.items():
        desc: dace_data.Array = sdfg.arrays[top_level_transient]

        # Setting the strides only make sense if we have more than one dimensions
        ndim = len(desc.shape)
        if ndim <= 1:
            continue

        # We assume that everything is in C order initially, to get FORTRAN order
        #  we simply have to reverse the order. This is necessary only for transient
        #  access nodes because the non-transients come from outside and have their
        #  own strides.
        # TODO(phimuell): Set the stride based on the actual access pattern.
        if desc.transient and gpu:
            new_stride_order = list(range(ndim))
            desc.set_strides_from_layout(*new_stride_order)

        # Now we have to propagate the changed strides. Because we already have
        #  collected all the AccessNodes we are using the
        #  `gt_propagate_strides_from_access_node()` function, but we have to
        #  create `processed_nsdfg` set already outside here.
        #  While global access nodes in top level SDFGs should have the correct strides,
        #  we need to propagate those strides into the nested SDFGs that use them.
        # TODO(phimuell): Updated the functions such that only one scan is needed.
        processed_nsdfgs: set[dace_nodes.NestedSDFG] = set()
        for state, access_node in accesses:
            gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=access_node,
                processed_nsdfgs=processed_nsdfgs,
            )

    # Now handle the views.
    # TODO(phimuell): Remove once `gt_propagate_strides_from_access_node()` can handle views.
    _gt_modify_strides_of_views_non_recursive(sdfg)


def gt_propagate_strides_of(sdfg: dace.SDFG, data_name: str) -> None:
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
    """Propagates the stride of `outer_node` to any adjacent NestedSDFG.

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

    Todo:
        Once this function supports the propagation to views `_gt_modify_strides_of_views()`
        can be removed.
    """
    assert isinstance(state, dace.SDFGState)

    if processed_nsdfgs is None:
        # For preventing the case that nested SDFGs are handled multiple time.
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
    """Propagates the strides of `outer_node` along `edge` in the dataflow direction.

    In this context "along the dataflow direction" means that `edge` is an outgoing
    edge of `outer_node` and the strides are propagated into all NestedSDFGs that
    are downstream of `outer_node`.

    Except in certain cases this function should not be used directly. It is
    instead recommended to use `gt_propagate_strides_of()`, which propagates
    all edges in the SDFG.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that writes to the data node, the nested SDFG is expected as the source.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of NestedSDFGs that were already processed. Only specify when
            you know what your are doing.
    """
    assert edge.src is outer_node
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
    """Propagates the strides of `outer_node` along `edge` in the opposite direction of the dataflow

    In this context "in the opposite direction of the dataflow" means that `edge`
    is an incoming edge of `outer_node` and the strides are propagated into all
    NestedSDFGs that are upstream of `outer_node`.

    Except in certain cases this function should not be used directly. It is
    instead recommended to use `gt_propagate_strides_of()`, which propagates
    all edges in the SDFG.

    Args:
        sdfg: The SDFG to process.
        state: The state where the data node is used.
        edge: The edge that writes to the data node, the nested SDFG is expected as the source.
        outer_node: The data node whose strides should be propagated.
        processed_nsdfgs: Set of NestedSDFGs that were already processed. Only specify when
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

    When the function encounters a NestedSDFG it will determine what data
    the `outer_node` is mapped to on the inside of the NestedSDFG.
    It will then replace the stride of the inner descriptor with the ones
    of the outside. Afterwards it will recursively propagate the strides
    inside the NestedSDFG.
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

        def get_subset(
            state: dace.SDFGState,
            edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
        ) -> dace.subsets.Subset:
            return edge.data.get_src_subset(edge, state)

        def next_edges_by_connector(
            state: dace.SDFGState,
            edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
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

        def get_subset(
            state: dace.SDFGState,
            edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
        ) -> dace.subsets.Subset:
            return edge.data.get_dst_subset(edge, state)

        def next_edges_by_connector(
            state: dace.SDFGState,
            edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
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
        _gt_map_strides_into_nested_sdfg(
            sdfg=sdfg,
            nsdfg_node=nsdfg_node,
            inner_data=inner_data,
            outer_subset=get_subset(state, edge),
            outer_desc=outer_node.desc(sdfg),
        )

        # Since the function call above is not recursive we have now to propagate
        #  the change into the NestedSDFGs. Using `_gt_find_toplevel_data_accesses()`
        #  is a bit overkill, but allows for a more uniform processing.
        # TODO(phimuell): Instead of scanning every level for every data we modify
        #   we should scan the whole SDFG once and then reuse this information.
        accesses_in_nested_sdfg = _gt_find_toplevel_data_accesses(
            sdfg=nsdfg_node.sdfg,
            only_transients=False,  # Because on the nested levels they are globals.
            only_arrays=True,
        )
        for nested_state, nested_access in accesses_in_nested_sdfg.get(inner_data, list()):
            # We have to use `gt_propagate_strides_from_access_node()` here because we
            #  have to handle its entirety. We could wait until the other branch processes
            #  the nested SDFG, but this might not work, so let's do it fully now.
            gt_propagate_strides_from_access_node(
                sdfg=nsdfg_node.sdfg,
                state=nested_state,
                outer_node=nested_access,
                processed_nsdfgs=processed_nsdfgs,
            )


def _gt_map_strides_into_nested_sdfg(
    sdfg: dace.SDFG,
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_data: str,
    outer_subset: dace.subsets.Subset,
    outer_desc: dace_data.Data,
) -> None:
    """Modify the strides of `inner_data` inside `nsdfg_node` to match `outer_desc`.

    `inner_data` is the name of a data descriptor inside the NestedSDFG.
    The function will then modify the strides of `inner_data`, assuming this
    is an array, to match the ones of `outer_desc`.

    Args:
        sdfg: The SDFG containing the NestedSDFG.
        nsdfg_node: The node in the parent SDFG that contains the NestedSDFG.
        inner_data: The name of the data descriptor that should be processed
            inside the NestedSDFG (by construction also a connector name).
        outer_subset: The subset that describes what part of the outer data is
            mapped into the NestedSDFG.
        outer_desc: The data descriptor of the data on the outside.

    Todo:
        - Handle explicit dimensions of size 1.
        - What should we do if the stride symbol is used somewhere else, creating an
            alias is probably not the right thing?
        - Handle the case if the outer stride symbol is already used in another
            context inside the Neste SDFG.
    """
    # We need to compute the new strides. In the following we assume that the
    #  relative order of the dimensions does not change, but we support the case
    #  where some dimensions of the outer data descriptor are not present on the
    #  inside. For example this happens for the Memlet `a[__i0, 0:__a_size1]`. We
    #  detect this case by checking if the Memlet subset in that dimension has size 1.
    # TODO(phimuell): Handle the case were some additional size 1 dimensions are added.
    inner_desc: dace_data.Data = nsdfg_node.sdfg.arrays[inner_data]
    inner_shape = inner_desc.shape

    outer_shape = outer_desc.shape
    outer_strides = outer_desc.strides
    outer_inflow = outer_subset.size()

    if isinstance(inner_desc, dace_data.Scalar):
        # A scalar does not have a stride that must be propagated.
        return

    # Now determine the new stride that is needed on the inside.
    new_strides: list = []
    if len(outer_shape) == len(inner_shape):
        # The inner and the outer descriptor have the same dimensionality.
        #  We now have to decide if we should take the stride from the outside,
        #  which happens for example in case of `A[0:N, 0:M] -> B[N, M]`, or if we
        #  must take 1, which happens if we do `A[0:N, i] -> B[N, 1]`, we detect that
        #  based on the volume that flows in.
        for dim_ostride, dim_oinflow in zip(outer_strides, outer_inflow, strict=True):
            new_strides.append(1 if dim_oinflow == 1 else dim_ostride)

    elif len(inner_shape) < len(outer_shape):
        # There are less dimensions on the inside than on the outside. This means
        #  that some were sliced away. We detect this case by checking if the Memlet
        #  subset in that dimension has size 1.
        #  NOTE: That this is not always correct as it might be possible that there
        #   are some explicit size 1 dimensions at several places.
        new_strides = []
        for dim_ostride, dim_oinflow in zip(outer_strides, outer_inflow, strict=True):
            if dim_oinflow == 1:
                pass
            else:
                new_strides.append(dim_ostride)
            assert len(new_strides) <= len(inner_shape)
    else:
        # The case that we have more dimensions on the inside than on the outside.
        #  This is currently not supported.
        raise NotImplementedError("NestedSDFGs can not be used to increase the rank.")

    if len(new_strides) != len(inner_shape):
        raise ValueError("Failed to compute the inner strides.")

    # For the strides of the arrays inside the nested SDFG we will create a new unique
    #  symbol which is initialized, through the symbol mapping, to the value of this
    #  stride on the outside. The benefit is that only the mapped container is affected
    #  and nothing else. Consider for example the case where initially two arrays
    #  inside the nested SDFG use the same stride symbol, but only one array is mapped.
    #  The main drawback is that the logical connection is lost, thus if the old
    #  stride symbol is used somewhere inside the nested SDFG, with the expectation
    #  that it corresponds to the stride of the inner container, then this connection
    #  is lost. However, this is probably not much of an issue for the strides, but
    #  more problematic for the shape, whose symbols are likely to appear as loop bounds.
    for i, dim_ostride in enumerate(new_strides):
        if str(dim_ostride).isdigit():
            # A literal stride (e.g. `1`) can be set directly
            new_strides[i] = dim_ostride
        else:
            if dim_ostride.is_symbol:
                # Try reusing the same symbol name as the outer stride, but find a new name if already used.
                dim_istride = nsdfg_node.sdfg.add_symbol(
                    dim_ostride.name, sdfg.symbols[dim_ostride.name], find_new_name=True
                )
            else:
                # Map a symbolic expression such as `value1 - value2` to a new stride symbol.
                dim_istride = nsdfg_node.sdfg.add_symbol(
                    f"__{inner_data}_stride_{i}",
                    gtx_dace_args.FIELD_SYMBOL_DTYPE,
                    find_new_name=True,
                )
            new_strides[i] = dace.symbolic.pystr_to_symbolic(dim_istride)
            nsdfg_node.symbol_mapping[dim_istride] = dim_ostride

    # We have to replace the `strides` attribute of the inner descriptor.
    inner_desc.set_shape(inner_desc.shape, new_strides)


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
    be returned. In addition by seeting `only_arrays` only arrays will be
    returned.

    Args:
        sdfg: The SDFG to process.
        only_transients: If `True` only include transients.
        only_arrays: If `True`, defaults to `False`, only arrays are returned.
        inlclude_views: Also returns Views.

    Returns:
        A `dict` that maps the name of a data container, to a list of tuples
        containing the state where the AccessNode was found and the AccessNode.
    """
    # List of data that is accessed on the top level and all its access node.
    top_level_data: dict[str, list[tuple[dace.SDFGState, dace_nodes.AccessNode]]] = dict()

    # List of all data that were found not on top level.
    not_top_level_data: set[str] = set()

    for state in sdfg.states():
        assert isinstance(state, dace.SDFGState)
        scope_dict = state.scope_dict()
        for dnode in state.data_nodes():
            data: str = dnode.data
            if scope_dict[dnode] is not None:
                # The node was not found on the top level. So we can ignore it.
                # We also check if it was ever found on the top level, this should
                #  not happen, as everything should go through Maps. But some strange
                #  DaCe transformation might do it.
                assert data not in top_level_data, (
                    f"Found {data} on the top level and inside a scope."
                )
                not_top_level_data.add(data)
                continue

            elif data in top_level_data:
                # The data is already known to be in top level data, so we must add the
                #  AccessNode to the list of known nodes. But nothing else.
                top_level_data[data].append((state, dnode))
                continue

            elif gtx_transformations.utils.is_view(dnode, sdfg):
                # The AccessNode refers to a View so we ignore it anyway.
                continue

            # We have found a new data node that is on the top node and is unknown.
            assert data not in not_top_level_data, (
                f"Found {data} on the top level and inside a scope."
            )
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


def _gt_modify_strides_of_views_non_recursive(sdfg: dace.SDFG) -> None:
    """The function determines the strides of Views.

    The function should not be called directly, instead it is called by
    `gt_change_strides()` directly if needed. The function will recursively
    process the SDFG and modifies the strides.

    Todo:
        Once `gt_propagate_strides_from_access_node()` can handle View updates
        evaluate if this function is still needed.
    """
    for state in sdfg.states():
        scope_dict = state.scope_dict()
        propagation_record: set[PropagatedStrideRecord] = set()
        for view_node in state.data_nodes():
            view_desc = view_node.desc(sdfg)
            if not isinstance(view_desc, dace_data.View):
                continue
            viewed_node = gtx_transformations.utils.track_view(view_node, state, sdfg)
            viewed_desc = viewed_node.desc(sdfg)

            # We are not able to handle tower of Views
            if isinstance(viewed_desc, dace_data.View):
                raise NotImplementedError(
                    f"Can not handle the view '{view_node.data}' that views view '{viewed_node.data}'"
                )

            # If both the View and the viewed node are not on the top level then do
            #  nothing. Why? The answer is that this function is only called by
            #  `_gt_change_strides_non_recursive_impl()` which only manipulates
            #  the strides of transients at the top level and leaves the one inside
            #  a Map alone. Thus there was no modification and no change.
            if scope_dict[viewed_node] is not None and scope_dict[view_node] is not None:
                continue

            # If the viewed data is global data, then we do not modify the strides because
            #  we assume that it was set correctly from the beginning and the viewed strides
            #  have not changed.
            # TODO(phimuell): Check interaction with the demote feature of auto optimizer.
            if not viewed_desc.transient:
                continue

            # There is a special case if the View is a scalar or has a shape of `(1,)`
            #  then strides are meaningless.
            if isinstance(view_desc, dace_data.Scalar) or (
                len(view_desc.shape) == 1 and ((view_desc.shape[0] == 1) == True)  # noqa: E712 [true-false-comparison]  # SymPy comparison
            ):
                continue

            # If the dimensionality of the two data is different then we can not handle it.
            #  This is probably the most difficult case to handle. However, instead of
            #  handling the case, we should simply remove the View.
            if len(view_desc.shape) != len(viewed_desc.shape):
                raise NotImplementedError(
                    f"Can not handle the change from {len(viewed_desc.shape)} ({viewed_node.data}) to {len(view_desc.shape)} ({view_node.data})."
                )

            # Even if they have the same dimensionality, we can not simply copy the strides.
            #  Consider the case were `viewed_desc` is a 2D array with the following shape
            #  `(N, M)`. `view_desc` is a vertical slice, but for some reasons has an
            #  additional dummy dimension, i.e. its shape is either `(N, 1)` or `(1, N)`.
            #  So copying the strides around is not gonna work.
            if view_desc.shape != viewed_desc.shape:
                raise NotImplementedError(
                    f"Can not change from shape `{viewed_desc.shape}` ({viewed_node.data})to shape `{view_desc.shape}` ({view_node.data})."
                )

            # In case they have the same shape we are fine.
            view_desc.strides = viewed_desc.strides
            gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=view_node,
                processed_nsdfgs=propagation_record,
            )

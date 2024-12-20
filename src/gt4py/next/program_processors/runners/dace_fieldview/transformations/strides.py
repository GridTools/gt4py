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

from gt4py.next.program_processors.runners.dace_fieldview import (
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


def gt_change_transient_strides(
    sdfg: dace.SDFG,
    gpu: bool,
) -> dace.SDFG:
    """Modifies the strides of transients.

    The function will analyse the access patterns and set the strides of
    transients in the optimal way.
    The function should run after all maps have been created.

    After the strides have been adjusted the function will also propagate
    the strides into nested SDFG. This propagation will happen with
    `ignore_symbol_mapping` set to `True`, see `gt_propagate_strides_of()`
    for more.

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

        # Setting the strides only make sense if we have more than one dimensions
        ndim = len(desc.shape)
        if ndim <= 1:
            continue

        # We assume that everything is in C order initially, to get FORTRAN order
        #  we simply have to reverse the order.
        # TODO(phimuell): Improve this.
        new_stride_order = list(range(ndim))
        desc.set_strides_from_layout(*new_stride_order)

        # Now we have to propagate the changed strides. Because we already have
        #  collected all the AccessNodes we are using the
        #  `gt_propagate_strides_from_access_node()` function, but we have to
        #  create `processed_nsdfg` set already outside here.
        #  Furthermore, the same comment as above applies here, we do not have to
        #  propagate the non-transients, because they either come from outside,
        #  or they were already handled in the levels above, where they were
        #  defined and then propagated down.
        # TODO(phimuell): Updated the functions such that only one scan is needed.
        processed_nsdfgs: set[dace_nodes.NestedSDFG] = set()
        for state, access_node in accesses:
            gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=access_node,
                processed_nsdfgs=processed_nsdfgs,
                ignore_symbol_mapping=True,
            )


def gt_propagate_strides_of(
    sdfg: dace.SDFG,
    data_name: str,
    ignore_symbol_mapping: bool = True,
) -> None:
    """Propagates the strides of `data_name` within the whole SDFG.

    This function will call `gt_propagate_strides_from_access_node()` for every
    AccessNode that refers to `data_name`. It will also make sure that a descriptor
    inside a NestedSDFG is only processed once.

    Args:
        sdfg: The SDFG on which we operate.
        data_name: Name of the data descriptor that should be handled.
        ignore_symbol_mapping: If `False` (default is `True`) try to modify the `symbol_mapping`
            of NestedSDFGs instead of manipulating the data descriptor.
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
                ignore_symbol_mapping=ignore_symbol_mapping,
            )


def gt_propagate_strides_from_access_node(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    outer_node: dace_nodes.AccessNode,
    ignore_symbol_mapping: bool = True,
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
        ignore_symbol_mapping: If `False` (default is `True`), try to modify the `symbol_mapping`
            of NestedSDFGs instead of manipulating the data descriptor.
        processed_nsdfgs: Set of NestedSDFG that were already processed and will be ignored.
            Only specify when you know what your are doing.
    """
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
            ignore_symbol_mapping=ignore_symbol_mapping,
        )
    for out_edge in state.out_edges(outer_node):
        gt_map_strides_to_dst_nested_sdfg(
            sdfg=sdfg,
            state=state,
            edge=out_edge,
            outer_node=outer_node,
            processed_nsdfgs=processed_nsdfgs,
            ignore_symbol_mapping=ignore_symbol_mapping,
        )


def gt_map_strides_to_dst_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.Edge,
    outer_node: dace.nodes.AccessNode,
    ignore_symbol_mapping: bool = True,
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
        ignore_symbol_mapping: If `False`, the default, try to modify the `symbol_mapping`
            of NestedSDFGs instead of manipulating the data descriptor.
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
        ignore_symbol_mapping=ignore_symbol_mapping,
    )


def gt_map_strides_to_src_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.Edge,
    outer_node: dace.nodes.AccessNode,
    ignore_symbol_mapping: bool = False,
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
        ignore_symbol_mapping: If `False`, the default, try to modify the `symbol_mapping`
            of NestedSDFGs instead of manipulating the data descriptor.
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
        ignore_symbol_mapping=ignore_symbol_mapping,
    )


def _gt_map_strides_to_nested_sdfg_src_dst(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace.sdfg.graph.MultiConnectorEdge[dace.Memlet],
    outer_node: dace.nodes.AccessNode,
    processed_nsdfgs: Optional[set[PropagatedStrideRecord]],
    propagate_along_dataflow: bool,
    ignore_symbol_mapping: bool = False,
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
        ignore_symbol_mapping: If `False`, the default, try to modify the `symbol_mapping`
            of NestedSDFGs instead of manipulating the data descriptor.

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
                ignore_symbol_mapping=ignore_symbol_mapping,
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
            ignore_symbol_mapping=ignore_symbol_mapping,
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
                ignore_symbol_mapping=ignore_symbol_mapping,
            )


def _gt_map_strides_into_nested_sdfg(
    sdfg: dace.SDFG,
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_data: str,
    outer_subset: dace.subsets.Subset,
    outer_desc: dace_data.Data,
    ignore_symbol_mapping: bool,
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
        ignore_symbol_mapping: If possible the function will perform the renaming
            through the `symbol_mapping` of the nested SDFG. If `True` then
            the function will always perform the renaming.
            Note that setting this value to `False` might have negative side effects.

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
    inner_strides_init = inner_desc.strides

    outer_strides = outer_desc.strides
    outer_inflow = outer_subset.size()

    new_strides: list = []
    for dim_ostride, dim_oinflow in zip(outer_strides, outer_inflow, strict=True):
        if dim_oinflow == 1:
            # This is the case of implicit slicing along one dimension.
            pass
        else:
            # There is inflow into the SDFG, so we need the stride.
            new_strides.append(dim_ostride)
        assert len(new_strides) <= len(inner_shape)

    # If we have a scalar on the inside, then there is nothing to adjust.
    #  We could have performed the test above, but doing it here, gives us
    #  the chance of validating it.
    if isinstance(inner_desc, dace_data.Scalar):
        if len(new_strides) != 0:
            raise ValueError(f"Dimensional error for '{inner_data}' in '{nsdfg_node.label}'.")
        return

    if not isinstance(inner_desc, dace_data.Array):
        raise TypeError(
            f"Expected that '{inner_data}' is an 'Array' but it is '{type(inner_desc).__name__}'."
        )

    if len(new_strides) != len(inner_shape):
        raise ValueError("Failed to compute the inner strides.")

    # Now we actually replace the strides, there are two ways of doing it.
    #  The first is to create an alias in the `symbol_mapping`, however,
    #  this is only possible if the current strides are singular symbols,
    #  like `__a_strides_1`, but not expressions such as `horizontal_end - horizontal_start`
    #  or literal values. Furthermore, this would change the meaning of the
    #  old stride symbol in any context and not only in the one of the stride
    #  of a single and isolated data descriptor.
    #  The second way would be to replace `strides` attribute of the
    #  inner data descriptor. In case the new stride consists of expressions
    #  such as `value1 - value2` we have to make them available inside the
    #  NestedSDFG. However, it could be that the strides is used somewhere else.
    # We will do the following, if `ignore_symbol_mapping` is `False` and
    #  the strides of the inner descriptors are symbols, we will use the
    #  symbol mapping. Otherwise, we will replace the `strides` attribute
    #  of the inner descriptor, in addition we will install a remapping,
    #  for those values that were a symbol.
    if (not ignore_symbol_mapping) and all(
        isinstance(inner_stride, dace.symbol) for inner_stride in inner_strides_init
    ):
        # Use the symbol
        for inner_stride, outer_stride in zip(inner_desc.strides, new_strides, strict=True):
            nsdfg_node.symbol_mapping[inner_stride.name] = outer_stride
    else:
        # We have to replace the `strides` attribute of the inner descriptor.
        inner_desc.set_shape(inner_desc.shape, new_strides)

        # Now find the free symbols that the new strides need.
        #  Note that usually `free_symbols` returns `set[str]`, but here, because
        #  we fall back on SymPy, we get back symbols. We will keep them, because
        #  then we can use them to extract the type form them, which we need later.
        new_strides_symbols: list[dace.symbol] = []
        for new_stride_dim in new_strides:
            if dace.symbolic.issymbolic(new_stride_dim):
                new_strides_symbols.extend(sym for sym in new_stride_dim.free_symbols)
            else:
                # It is not already a symbol, so we turn it into a symbol.
                #  However, we only add it, if it is also a symbol, for example `1`.
                #  should not be added.
                new_stride_symbol = dace.symbolic.pystr_to_symbolic(new_stride_dim)
                if new_stride_symbol.is_symbol:
                    new_strides_symbols.append(new_stride_symbol)

        # Now we determine the set of symbols that should be mapped inside the NestedSDFG.
        #  We will exclude all that are already inside the `symbol_mapping` (we do not
        #  check if they map to the same value, we just hope it). Furthermore,
        #  we will exclude all symbols that are listed in the `symbols` property
        #  of the SDFG that is nested, and hope that it has the same meaning.
        # TODO(phimuell): Add better checks to avoid overwriting.
        missing_symbol_mappings: set[dace.symbol] = {
            sym
            for sym in new_strides_symbols
            if not (sym.name in nsdfg_node.sdfg.symbols or sym.name in nsdfg_node.symbol_mapping)
        }

        # Now propagate the symbols from the parent SDFG to the NestedSDFG.
        for sym in missing_symbol_mappings:
            assert sym.name in sdfg.symbols, f"Expected that '{sym}' is defined in the parent SDFG."
            nsdfg_node.sdfg.add_symbol(sym.name, sdfg.symbols[sym.name])
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
        A `dict` that maps the name of a data container, to a list of tuples
        containing the state where the AccessNode was found and the AccessNode.
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
                # The AccessNode refers to a View so we ignore it anyway.
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

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from typing import Final, Iterable, Optional, Sequence, TypeAlias

import dace
import sympy
from dace import data as dace_data, subsets as dace_sbs, symbolic as dace_sym
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


InlineSpec: TypeAlias = tuple[
    dace_nodes.MapExit,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
    dace_sbs.Range,
    dict[str, dace_sym.SymbolicType],
]
"""Specify how the inlining has to be performed.

The information is computed by the `find_nodes_to_inline()` function and returned as
its second return value (the first return value is the set of nodes that must be
inlined). The user should see it as an opaque structure.
"""


def inline_dataflow_into_map(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> Optional[tuple[dace_nodes.NestedSDFG, dace_nodes.AccessNode]]:
    """Tries to perform the inlining of `edge`.

    The function replaces the transmission, i.e. reading, of data through `edge`
    with an on-the-fly computation of the same data. The term inlining is used
    because the dataflow, that computes the transmitted data is embedded/inlined
    into the scope of `edge`.

    This function performs inlining in a single step or returns `None` if that
    is not possible. It is also possible to perform these two steps separately,
    if and how inlining is possible, see `find_nodes_to_inline()` and perform the
    actual inlining accordingly, see `perform_dataflow_inlining()`, separately.
    For more information please consult the documentation of these two functions.

    Args:
        sdfg: The SDFG on which we operate.
        state: The state in which `edge` is located.
        edge: The `edge` whose dataflow should be inlined.
    """
    extracted_information = find_nodes_to_inline(sdfg, state, edge)
    if extracted_information is None:
        return None

    return perform_dataflow_inlining(
        sdfg=sdfg,
        state=state,
        edge=edge,
        nodes_to_inline=extracted_information[0],
        inline_spec=extracted_information[1],
    )


def perform_dataflow_inlining(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    nodes_to_inline: set[dace_nodes.Node],
    inline_spec: InlineSpec,
) -> Optional[tuple[dace_nodes.NestedSDFG, dace_nodes.AccessNode]]:
    """Performs the second step, i.e. the actual inlining, of the dataflow.

    The function will use the information of `nodes_to_inline` and `inline_spec`,
    which was computed by `find_nodes_to_inline()`, to perform the inlining. The
    inlining essentially works by creating a nested SDFG inside the Map, where `edge`
    is located and replicate the dataflow into it. The function will also create an
    AccessNode that is used as output of the nested SDFG. Furthermore, the function
    will adjust all dataflow, that was going through `edge` such that it will pass
    through that AccessNode. As a final step the Memlet path of `edge` will be removed.

    Returns:
        The function returns two values. First it will return the nested SDFG that
        contains the dataflow needed for the computation. The second value is an
        AccessNode that is used as output of the nested SDFG.

    Args:
        sdfg: The SDFG on which we operate.
        state: The state in which `edge` is located.
        edge: The `edge` whose dataflow should be inlined.
        nodes_to_inline: The set of nodes that create the dataflow that is needed.
            This is the first return value of `find_nodes_to_inline()`.
        inline_spec: Auxiliary information how the inlining has to happen. This is
            the second return value of `find_nodes_to_inline()`.
    """
    (
        first_map_exit,
        intermediate_node,
        second_map_entry,
        exchange_subset,
        first_map_param_mapping,
    ) = inline_spec
    assert len(nodes_to_inline) > 0
    nsdfg, input_node_map, output_name = _populate_nested_sdfg(
        sdfg=sdfg,
        state=state,
        nodes_to_replicate=nodes_to_inline,
        first_map_exit=first_map_exit,
        exchange_subset=exchange_subset,
        intermediate_node=intermediate_node,
    )

    return _insert_nested_sdfg(
        sdfg=sdfg,
        state=state,
        edge_to_replace=edge,
        second_map_entry=second_map_entry,
        nsdfg=nsdfg,
        input_node_map=input_node_map,
        output_name=output_name,
        first_map_param_mapping=first_map_param_mapping,
        intermediate_node=intermediate_node,
    )


def find_nodes_to_inline(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> Optional[tuple[set[dace_nodes.Node], InlineSpec]]:
    """First step of dataflow inlining, computing the inline specification.

    The inline specification describes how the inlining of dataflow has to be done.
    The values computed by this function must be passed to `perform_dataflow_inlining()`
    for actually inlining dataflow.
    The function will analyze how a read through `edge` can be replaced by direct
    computations. If this is not possible then the function returns `None`.

    Currently the following restrictions (not exhaustive) are imposed on the dataflow:
    - `edge` must be an outgoing edge of a top level MapEntry.
    - The generating dataflow, i.e. the one that is inlined, must be inside a Map.
    - The intermediate between the two Maps can only have one producer, i.e. incoming
        degree of 1. Note that in GT4Py this means that there is no other AccessNode
        that refers to the same data with an incoming edge.
    - All data dependencies must be met. This means that `edge` can only consume data
        that is generated by a single iteration of the first data.

    Return:
        The function returns two values. The first is a `set` that contains all nodes
        that have to be inlined (the two Map nodes of the generating Map are not
        included). The second value are auxiliary data for the inlining in an
        unspecified format.

    Args:
        sdfg: The SDFG on which we operate.
        state: The state in which `edge` is located.
        edge: The `edge` whose dataflow should be inlined.
    """
    scope_dict = state.scope_dict()

    # Must be connected to a top level MapEntry.
    if not isinstance(edge.src, dace_nodes.MapEntry):
        return None
    second_map_entry: dace_nodes.MapEntry = edge.src
    if scope_dict[edge.src] is not None:
        return None

    # Check the Memlet.
    if edge.data.get_src_subset(edge, state) is None:
        return None
    if edge.data.wcr is not None:
        return None

    # We require that every access in the edge is either a literal digit, such as
    #  `1` but also `1:3` or that exactly one symbol that is also map parameter is
    #  used to do the index. Note that this does allow accesses such as
    #  `a[map_index:(map_index + 4)]` but not accesses such as `a[map_index:(map_index + symb)]`
    #  or `a[some_other_symbol]`. Furthermore, we require that all parameters of the
    #  second Map are used for the access (exception see below). These rule are used
    #  to ensure that one iteration of the first Map is enough to satisfy all data
    #  dependencies of the second Map.
    avail_second_map_param = {param for param in second_map_entry.map.params}
    literal_index_access_dims: list[int] = []
    for dim, (start, stop, step) in enumerate(edge.data.src_subset):
        if (step != 1) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return None

        if str(start).isdigit():
            # The start index is a digit, we require that the end is also a digit.
            if not str(stop).isdigit():
                return None

            # There is an access done using a literal.
            literal_index_access_dims.append(dim)

        else:
            # The start index is a symbol. We require that it is an unused parameter
            #  and that the stop is also that parameter.
            start_symbols = {str(sym) for sym in start.free_symbols}
            if len(start_symbols) != 1:
                return None
            elif start_symbols.issubset(avail_second_map_param):
                stop_symbols = {str(sym) for sym in stop.free_symbols}
                if start_symbols != stop_symbols:
                    return None
                avail_second_map_param.remove(next(iter(start_symbols)))

    # Usually we require that all parameters of the second Map are used for the
    #  access. The main reason is to prevent some slicing, i.e. to ensure that
    #  the access is not done later. However, this prevents the second Map to read
    #  from a Map that has less dimensions. Thus as an exception we allow that case
    #  if all accesses are done using parameters, then not all of them have
    #  to be used.
    if (len(literal_index_access_dims) != 0) and (len(avail_second_map_param) != 0):
        return None

    # We restrict the destination nodes to Taklets and AccessNodes because these node
    #  types are simple to modify. However, even the Tasklet could be hard to handle,
    #  in some cases (such as `deref` Tasklets).
    # TODO(phimuell): Allow more node types, especially NestedSDFGs in certain cases.
    for leave_edge in state.memlet_tree(edge).leaves():
        if not isinstance(leave_edge.dst, (dace_nodes.AccessNode, dace_nodes.Tasklet)):
            return None

    # There must be an intermediate that is only generated by a single source.
    #  NOTE: We omit the check that the intermediate is single use data here.
    intermediate_node: dace_nodes.Node = next(
        iter(state.in_edges_by_connector(edge.src, "IN_" + edge.src_conn[4:]))
    ).src
    if not isinstance(intermediate_node, dace_nodes.AccessNode):
        return None
    if state.in_degree(intermediate_node) != 1:
        return None

    # The producer must be a Map.
    producing_edge = next(iter(state.in_edges(intermediate_node)))
    if not isinstance(producing_edge.src, dace_nodes.MapExit):
        return None
    first_map_exit: dace_nodes.MapExit = producing_edge.src

    # Dynamic Map ranges are not yet supported.
    # TODO(phimuell): Lift this restriction.
    first_map_entry: dace_nodes.MapEntry = state.entry_node(first_map_exit)
    if any(
        iedge.dst_conn is None or (not iedge.dst_conn.startswith("IN_"))
        for iedge in state.in_edges(first_map_entry)
    ):
        return None

    # NOTE: On an `OUT_` connector of a MapExit node multiple edges can converge,
    #   we can not handle that thus we have to restrict that case.
    writing_edges = [
        iedge
        for iedge in state.in_edges_by_connector(
            first_map_exit, "IN_" + producing_edge.src_conn[4:]
        )
    ]
    if len(writing_edges) != 1:
        return None

    first_map_param_mapping = _get_first_map_parameter_to_second_map_mapping(
        state=state,
        first_map=first_map_exit.map,
        write_edge=writing_edges[0],
        second_map=edge.src.map,
        read_edge=edge,
    )

    # Check the data dependencies. It is a little bit different compared to classical
    #  MapFusionVertical, where the two iterations are coupled. Here the restriction
    #  is that there exists _one_ iteration of the first Map that, generates the data
    #  that _another_ iteration of the second Map needs. This is a much weaker restriction
    #  but also harder to check. As a first approximation we look at the shape of the
    #  data that is generated and consumed.
    #  To avoid dynamic allocation we require that the exchange set has a constant size.
    consumer_subset = edge.data.src_subset
    consumer_shape = consumer_subset.size()
    producer_subset = writing_edges[0].data.get_dst_subset(writing_edges[0], state)
    producer_shape = producer_subset.size()
    for cshp, pshp in zip(consumer_shape, producer_shape, strict=True):
        if not str(pshp).isdigit():
            return None
        if (cshp <= pshp) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            continue
        return None

    # If a dimension of the intermediate is accessed not by a Map parameter, we require
    #  that that range is known at generation time, see above. In addition to that we
    #  require that the range starts at zero. The only reason for that is that it makes
    #  computing the correction offset much simpler.
    # TODO(phimuell): Allow non-zero start index for intermediate access.
    for literal_index_access_dim in literal_index_access_dims:
        if producer_subset[literal_index_access_dim][0] != 0:
            return None

    # Collect all the nodes that give rise to the data.
    generating_nodes = gtx_transformations.utils.find_upstream_nodes(
        start=first_map_exit,
        state=state,
        start_connector="IN_" + producing_edge.src_conn[4:],
        limit_node=state.entry_node(first_map_exit),
    )
    assert first_map_exit not in generating_nodes
    assert state.entry_node(first_map_exit) not in generating_nodes
    assert len(generating_nodes) >= 1

    # For syntactical reasons we require that all outgoing edges of the nodes in
    #  `generating_nodes` must lead to nodes that are also in that set, or go to
    #  the MapExit node (in which case the outgoing edge must be `edge`). The only
    #  exceptions are AccessNode. Technically we could also handle Tasklets, but
    #  they would require a little bit more work in `_populated_nsdfg()`.
    for generating_node in generating_nodes:
        for oedge in state.out_edges(generating_node):
            if oedge.dst is first_map_exit:
                assert oedge is writing_edges[0]
            elif oedge.dst not in generating_nodes:
                if isinstance(oedge.src, dace_nodes.AccessNode) and (
                    not gtx_transformations.utils.is_view(oedge.src, sdfg)
                ):
                    continue
                return None

    return (
        generating_nodes,
        (
            first_map_exit,
            intermediate_node,
            edge.src,
            producer_shape,
            first_map_param_mapping,
        ),
    )


def _insert_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge_to_replace: dace_graph.MultiConnectorEdge[dace.Memlet],
    second_map_entry: dace_nodes.MapEntry,
    nsdfg: dace.SDFG,
    input_node_map: dict[str, dace_nodes.Node],
    output_name: str,
    first_map_param_mapping: dict[str, dace_sym.SymbolicType],
    intermediate_node: dace_nodes.AccessNode,
) -> tuple[dace_nodes.NestedSDFG, dace_nodes.AccessNode]:
    """Helper function for the `perform_dataflow_inlining()` step.

    After `_populate_nested_sdfg()` has prepared the nested SDFG, this function
    is used to insert it into the second Map scope and to rewire the internal
    dataflow to actually use it. It will also remove the Memlet path of
    `edge_to_replace`.
    """

    # Find all data that is already inside Map.
    nodes_available_in_second_map: Final[dict[dace_nodes.AccessNode, str]] = {
        iedge.src: "OUT_" + iedge.dst_conn[3:]
        for iedge in state.in_edges(second_map_entry)
        # TODO(phimuell): Remove the view check once `_populate_nested_sdfg()` is able to handle them.
        if isinstance(iedge.src, dace_nodes.AccessNode)
        and (not gtx_transformations.utils.is_view(iedge.src, sdfg))
    }

    nsdfg_node = state.add_nested_sdfg(
        sdfg=nsdfg,
        inputs=input_node_map.keys(),
        outputs={output_name},
        symbol_mapping=first_map_param_mapping,
    )

    for input_name, required_node in input_node_map.items():
        if required_node in nodes_available_in_second_map:
            conn_to_use = nodes_available_in_second_map[required_node]

            # Update the range to ensure that everything is mapped into it.
            memlet_path_to_widen = state.memlet_path(
                next(iter(state.in_edges_by_connector(second_map_entry, "IN_" + conn_to_use[4:])))
            )
            for edge in memlet_path_to_widen:
                edge.data.try_initialize(
                    sdfg, state, edge
                )  # There is not `Memlet::set_src_subset()`.
                edge.data.src_subset = dace_sbs.Range.from_array(sdfg.arrays[required_node.data])
                if edge.dst is second_map_entry:
                    break
            else:
                raise RuntimeError("Unreachable code.")

        else:
            conn_to_use = second_map_entry.next_connector(try_name=input_name)
            second_map_entry.add_scope_connectors(conn_to_use)
            state.add_edge(
                required_node,
                None,
                second_map_entry,
                "IN_" + conn_to_use,
                dace.Memlet.from_array(required_node.data, required_node.desc(sdfg)),
            )
            conn_to_use = "OUT_" + conn_to_use

        # Connect to the nested SDFG
        state.add_edge(
            second_map_entry,
            conn_to_use,
            nsdfg_node,
            input_name,
            dace.Memlet.from_array(required_node.data, required_node.desc(sdfg)),
        )

    # In case there is no input use an empty edge to bind the nested SDFG into the scope.
    if len(input_node_map) == 0:
        state.add_edge(
            second_map_entry,
            None,
            nsdfg_node,
            None,
            dace.Memlet(),
        )

    # Now rewire the output.
    #  Instead of reading from the intermediate, we will now read from the output of the
    #  nested SDFG. For which we create a dedicated output.
    outer_output_name = sdfg.add_datadesc(
        output_name,
        nsdfg.arrays[output_name].clone(),
        find_new_name=True,
    )
    sdfg.arrays[outer_output_name].transient = True
    outer_output_node = state.add_access(outer_output_name)

    state.add_edge(
        nsdfg_node,
        output_name,
        outer_output_node,
        None,
        dace.Memlet.from_array(outer_output_name, sdfg.arrays[outer_output_name]),
    )

    offset = _compute_offset(edge_to_replace, second_map_entry.map.params)
    replaced_edge = state.add_edge(
        outer_output_node,
        None,
        edge_to_replace.dst,
        edge_to_replace.dst_conn,
        dace.Memlet(
            data=outer_output_name,
            subset=edge_to_replace.data.src_subset.offset_new(offset, negative=True),
            other_subset=edge_to_replace.data.get_dst_subset(edge_to_replace, state),
        ),
    )

    # Readjust all accesses.
    gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
        is_producer_edge=False,
        new_edge=replaced_edge,
        ss_offset=offset,
        state=state,
        sdfg=sdfg,
        old_node=intermediate_node,
        new_node=outer_output_node,
    )

    # Now removing the edge that we just replaced. We will only replace at most the
    #  Memlet path, the actual source, i.e. the first Map, will still be there.
    #  To remove it, one should call a dead dataflow elimination.
    state.remove_memlet_path(edge_to_replace)

    return nsdfg_node, outer_output_node


def _populate_nested_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    nodes_to_replicate: set[dace_nodes.Node],
    first_map_exit: dace_nodes.MapExit,
    exchange_subset: dace_sbs.Range,
    intermediate_node: dace_nodes.AccessNode,
) -> tuple[dace.SDFG, dict[str, dace_nodes.Node], str]:
    """Helper function for the `perform_dataflow_inlining()` step.

    This is the first step of the inlining. The function will generate an SDFG
    and replicate the dataflow described by `nodes_to_replicate`. It will also
    prepare the mapping of all external data into the nested SDFG.
    """
    nsdfg = dace.SDFG(f"_inline_fused_nested_sdfg_{sdfg.name}_{first_map_exit.map.label}")
    nstate = nsdfg.add_state(is_start_block=True)

    # Maps the name of the input to the node on the outside that supplies it.
    input_node_map: dict[str, dace_nodes.Node] = dict()

    # Replicate the nodes into the nested SDFG.
    replicated_node_map: dict[dace_nodes.Node, dace_nodes.Node] = {}
    for node_to_replicate in nodes_to_replicate:
        if (
            isinstance(node_to_replicate, dace_nodes.AccessNode)
            and node_to_replicate.data not in nsdfg.arrays
        ):
            dname = node_to_replicate.data
            ddesc = sdfg.arrays[dname]
            nsdfg.add_datadesc(dname, ddesc.clone())
        replicated_node = copy.deepcopy(node_to_replicate)
        replicated_node_map[node_to_replicate] = replicated_node
        nstate.add_node(replicated_node)

    # This is the name used to store the output.
    output_name: Optional[str] = None

    # Replicate the connections.
    first_map_entry = state.entry_node(first_map_exit)
    for node_to_replicate in nodes_to_replicate:
        replicated_node = replicated_node_map[node_to_replicate]

        # All outgoing edges, with one exception, lead to a node that also has been
        #  replicated, and we can thus simply copy it. The exception is the final
        #  output edge, which connects the node to the first MapExit node and would
        #  write into the intermediate. Instead of this we create a new output node
        #  node and connect to it.
        for oedge in state.out_edges(node_to_replicate):
            dst_node = oedge.dst

            if dst_node is first_map_exit:
                # A connection to the MapExit node is indicating that we now write to
                #  the output, which is ultimately the intermediate. But since we can
                #  not use it we have to create a replacement.
                if output_name is not None:
                    raise ValueError("Found multiple connection to the MapExit node.")
                assert dst_node not in nodes_to_replicate

                output_name, output_desc = nsdfg.add_array(
                    f"__inline_fuser_replacing_{intermediate_node.data}",
                    shape=tuple(exchange_subset),
                    dtype=intermediate_node.desc(sdfg).dtype,
                    find_new_name=True,
                    transient=False,
                )
                nstate.add_edge(
                    replicated_node_map[oedge.src],
                    oedge.src_conn,
                    nstate.add_access(output_name),
                    oedge.dst_conn,
                    memlet=dace.Memlet(
                        data=output_name,
                        subset=dace_sbs.Range.from_array(output_desc),
                        other_subset=oedge.data.get_src_subset(oedge, state),
                    ),
                )

            elif oedge.src not in nodes_to_replicate:
                # This connection leads to a node that has not been replicated into the
                #  nested SDFG. This is only allowed if `oedge.src` is an AccessNode in
                #  that case we simply do not generate this edge. All other cases
                #  are harder to handle.
                # NOTE: In some cases it would be possible to handle views.
                if (
                    not isinstance(oedge.src, dace_nodes.AccessNode)
                ) or gtx_transformations.utils.is_view(oedge.src, sdfg):
                    raise NotImplementedError(
                        "Connections to a non replicated node are only allowed if the source node is an AccessNode"
                    )

                warnings.warn(
                    "Detected computation of data that might not be needed in inline fuser.",
                    stacklevel=0,
                )

            else:
                # A normal connection to a node that is also mapped into the nested
                #  SDFG. We just replicate it.
                nstate.add_edge(
                    replicated_node_map[oedge.src],
                    oedge.src_conn,
                    replicated_node_map[oedge.dst],
                    oedge.dst_conn,
                    dace.Memlet.from_memlet(oedge.data),
                )

        # We have to look at the input connections separately. There are two types of
        #  them. The first type comes from a node that has been replicated and was
        #  handled by the loop above. The second kind are connections from the
        #  MapEntry node. Such connections will trigger the mapping of the data from
        #  outside the Map into the nested SDFG.
        for iedge in state.in_edges(node_to_replicate):
            src_node = iedge.src
            if src_node is not first_map_entry:
                # This is a connection from a node that is also replicated into the
                #  nested SDFG. We will handle it later during the output nodes, thus
                #  there is nothing to do.
                assert src_node in nodes_to_replicate
                continue

            # If we are here then we have a connection from the MapEntry node. Thus
            #  we have to read from outside data.
            outside_input_node: dace_nodes.AccessNode = state.memlet_path(iedge)[0].src
            assert isinstance(outside_input_node, dace_nodes.AccessNode)

            # If we have not seen this node yet we have to map it into the nested SDFG.
            if outside_input_node not in replicated_node_map:
                if outside_input_node.data not in nsdfg.arrays:
                    # We have not yet mapped this data into the nested SDFG. For that
                    #  we use the same descriptor as on the outside, but we have to
                    #  mark it as non-transient.
                    nested_input_desc = outside_input_node.desc(sdfg).clone()
                    nested_input_desc.transient = False

                    if isinstance(nested_input_desc, dace_data.View):
                        raise NotImplementedError("Mapping of views is not supported.")

                    nsdfg.add_datadesc(outside_input_node.data, nested_input_desc)
                    replicated_node_map[outside_input_node] = nstate.add_access(
                        outside_input_node.data
                    )
                    input_node_map[outside_input_node.data] = outside_input_node

                else:
                    # We have not found this AccessNode yet but the data it refers
                    #  to has been found. In that case we reuse the AccessNode on
                    #  the inside we have already created.
                    # NOTE: That this will slightly change the dataflow as before,
                    #   on the outside the data was accessed through two different
                    #   AccessNodes and now only one is used, also only one of them
                    #   is mapped into the nested SDFG.
                    replicated_node_map[outside_input_node] = replicated_node_map[
                        input_node_map[outside_input_node.data]
                    ]

            # We will use the "last level Memlet" to read from it.
            nstate.add_edge(
                replicated_node_map[outside_input_node],
                None,
                replicated_node_map[iedge.dst],
                iedge.dst_conn,
                dace.Memlet.from_memlet(iedge.data),
            )

    assert len(input_node_map) == len(set(input_node_map.values()))
    assert output_name is not None

    return nsdfg, input_node_map, output_name


def _get_first_map_parameter_to_second_map_mapping(
    state: dace.SDFGState,
    first_map: dace_nodes.Map,
    write_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    second_map: dace_nodes.Map,
    read_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> dict[str, dace_sym.SymbolicType]:
    """Expresses the values of the first Map's parameter name in terms of the second map."""
    assert first_map.get_param_num() == second_map.get_param_num()
    assert isinstance(write_edge.dst, dace_nodes.MapExit)
    assert write_edge.data.get_dst_subset(write_edge, state) is not None
    assert isinstance(read_edge.src, dace_nodes.MapEntry)
    assert read_edge.data.get_src_subset(read_edge, state) is not None

    parameter_mapping: dict[str, str] = {}

    # Currently we only consider the lower bound and nothing else, which is probably
    #  okay for the GT4Py setting.
    #  NOTE: This assumes that all the magic happens in the lower bound.
    first_map_accesses = write_edge.data.get_dst_subset(write_edge, state).min_element()
    second_map_accesses = read_edge.data.get_src_subset(read_edge, state).min_element()

    # We assume that the different dimensions are independent, i.e. each parameter is
    #  used to access exactly one dimension, this rules out accesses such as
    #  `a[i + j, i - j]`. This is okay for GT4Py. However, we have now to find out
    #  which parameter acts in which dimensions.
    first_map_param_mapping = _find_dimensions_access(first_map.params, first_map_accesses)

    # If we read the data then we use the parameters of the second Map. However, we
    #  must now figuring out to which value of the first map this corresponds to.
    for dim in range(first_map.get_param_num()):
        first_map_param = first_map_param_mapping[dim]
        first_map_access = first_map_accesses[dim]
        second_map_access = second_map_accesses[dim]

        first_map_param_uniq = dace_sym.pystr_to_symbolic(
            f"__unique_param_{dim}_{first_map_param!s}_{id(second_map_access)}"
        )
        first_map_access_uniq = first_map_access.subs({first_map_param: first_map_param_uniq})

        res = sympy.solve(sympy.Eq(first_map_access_uniq, second_map_access), first_map_param_uniq)
        if len(res) != 1:
            raise ValueError(
                f"Could not solve access in dimension {dim}: '{first_map_access} == {second_map_access}'"
            )
        parameter_mapping[first_map_param] = res[0]

    return parameter_mapping


def _find_dimensions_access(
    params: Sequence[str], accesses: Iterable[dace_sym.SymbolicType]
) -> dict[int, str]:
    """Determine in which dimension a parameter is used inside `accesses`."""
    result_mapping: dict[int, str] = {}
    aparams: set[str] = set(params)

    for i, access in enumerate(accesses):
        free_symbs = {str(fs) for fs in access.free_symbols}

        if len(free_symbs) == 0:
            # This is the case for an access `a[_i, 1]`, so there is no dimension
            #  to associate it with.
            continue

        for aparam in aparams:
            if aparam in free_symbs:
                break
        else:
            raise ValueError(
                f"Could not associate access '{access}' to any of the parameters '{aparams}'"
            )
        result_mapping[i] = aparam
        aparams.discard(aparam)

    if len(aparams) != 0:
        raise ValueError(f"Could not associate parameters: {aparams}")

    return result_mapping


def _compute_offset(
    edge_to_replace: dace_graph.MultiConnectorEdge[dace.Memlet],
    second_map_params: Sequence[str],
) -> list[dace_sym.SymbolicType]:
    """The offset needed to adjust the reading subsets in the second Map."""
    offset: list[dace_sym.SymbolicType] = []
    avail_second_map_params = set(second_map_params)
    for start, _, _ in edge_to_replace.data.src_subset:
        if str(start).isdigit():
            offset.append(dace_sym.pystr_to_symbolic("0"))
        else:
            start_symbols = {str(sym) for sym in start.free_symbols}
            used_param_symbols = start_symbols.intersection(avail_second_map_params)
            assert len(used_param_symbols) == 1

            # We do not have to use `used_param_symbols[0]` as one might suspect.
            #  Because of the additional offset has been taken care of by
            #  `_get_first_map_parameter_to_second_map_mapping()`.
            offset.append(start)
            avail_second_map_params.discard(next(iter(used_param_symbols)))

    return offset

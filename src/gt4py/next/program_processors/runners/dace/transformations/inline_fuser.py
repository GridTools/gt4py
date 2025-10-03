# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from typing import Final, Iterable, Optional, Sequence

import dace
import sympy
from dace import data as dace_data, subsets as dace_sbs, symbolic as dace_sym
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def inline_dataflow_into_map(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> Optional[tuple[dace_nodes.NestedSDFG, dace_nodes.AccessNode]]:
    extracted_information = _extract_generating_dataflow_for_inlining(sdfg, state, edge)
    if extracted_information is None:
        return None

    (
        nodes_to_replicate,
        first_map_exit,
        intermediate_node,
        second_map_entry,
        exchange_subset,
        first_map_param_mapping,
    ) = extracted_information
    assert len(nodes_to_replicate) > 0
    nsdfg, input_node_map, output_name = _populate_nested_sdfg(
        sdfg=sdfg,
        state=state,
        nodes_to_replicate=nodes_to_replicate,
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
) -> tuple[dace_nodes.NestedSDFG, dace_nodes.AccessNode]:
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

    for input_name, requiered_node in input_node_map.items():
        if requiered_node in nodes_available_in_second_map:
            conn_to_use = nodes_available_in_second_map[requiered_node]

            # Update the range to ensure that everything is mapped into it.
            memlet_path_to_widen = state.memlet_path(
                next(iter(state.in_edges_by_connector(second_map_entry, "IN_" + conn_to_use[4:])))
            )
            for edge in memlet_path_to_widen:
                edge.data.try_initialize(
                    sdfg, state, edge
                )  # There is not `Memlet::set_src_subset()`.
                edge.data.src_subset = dace_sbs.Range.from_array(sdfg.arrays[requiered_node.data])
                if edge.dst is second_map_entry:
                    break
            else:
                raise RuntimeError("Unreachable code.")

        else:
            conn_to_use = second_map_entry.next_connector(try_name=input_name)
            second_map_entry.add_scope_connectors(conn_to_use)
            state.add_edge(
                requiered_node,
                None,
                second_map_entry,
                "IN_" + conn_to_use,
                dace.Memlet.from_array(requiered_node.data, requiered_node.desc(sdfg)),
            )
            conn_to_use = "OUT_" + conn_to_use

        # Connect to the nested SDFG
        state.add_edge(
            second_map_entry,
            conn_to_use,
            nsdfg_node,
            input_name,
            dace.Memlet.from_array(requiered_node.data, requiered_node.desc(sdfg)),
        )

    # Now connect to the output.
    # TODO(phimuell): Update this.
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

    # Now adapt the range from where we read.
    # NOTE: This is currently done in a sub optimal way, and only allows for the
    #   transmission of one element that is immediately consumed. This is currently
    #   guaranteed by the `_extract_generating_dataflow_for_inlining()` function.
    # TODO(phimuell): Lift this requirement and handle subsets and strides. Maybe copy
    #   the idea from MapFusion and create an intermediate here and then use the
    #   correction logic.
    state.add_edge(
        outer_output_node,
        None,
        edge_to_replace.dst,
        edge_to_replace.dst_conn,
        dace.Memlet(
            data=outer_output_name,
            subset=dace_sbs.Range.from_array(sdfg.arrays[outer_output_name]),
            other_subset=edge_to_replace.data.get_dst_subset(edge_to_replace, state),
        ),
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
    """Populate the SDFG.

    The function returns the SDFG which should become the nested SDFG. It will also
    return the set of names that are used/needed as input and the single output name.
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

    # Create the connections between the nodes in the nested SDFG.
    #  This also involves adding the outside data into the nested container and
    #  create the output.
    first_map_entry = state.entry_node(first_map_exit)
    for node_to_replicate in nodes_to_replicate:
        replicated_node = replicated_node_map[node_to_replicate]

        # There are two types of incoming edges. The first kind are edges that come
        #  from another node that is also replicated into the nested SDFG, we just
        #  replicate them.
        #  The second kind are connections from the MapEntry node. That node does not
        #  directly exist in the nested SDFG, but such a connection is used to supply
        #  data into the Map. Thus we have to map that outside data (outside of the
        #  first Map and outside the nested SDFG) into the nested SDFG and connect the
        #  node to it.
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
                        raise NotImplementedError("Mapping in of views is not supported.")

                    nsdfg.add_datadesc(outside_input_node.data, nested_input_desc)
                    replicated_node_map[outside_input_node] = nstate.add_access(
                        outside_input_node.data
                    )
                    input_node_map[outside_input_node.data] = outside_input_node

                else:
                    # We have not found this AccessNode yet but the data it refers
                    #  to has been found. In that case we reuse the AccessNode on
                    #  the inside we have already created.
                    # NOTE: That this will slightly change things as before, on the
                    #   outside the data was accessed through two different
                    #   AccessNodes and now only one is used, also only one of them
                    #   is mapped into the nested SDFG.
                    replicated_node_map[outside_input_node] = replicated_node_map[
                        input_node_map[outside_input_node.data]
                    ]

            # We will use the "last level Memlet" to read from it.
            nstate.add_edge(
                replicated_node_map[outside_input_node],
                iedge.src_conn,
                replicated_node_map[iedge.dst],
                iedge.dst_conn,
                dace.Memlet.from_memlet(iedge.data),
            )

        # Recreate the connections between the nodes that we have replicated.
        #  It is important that above we only handled the connections that were now
        #  because they were coming from the MapEntry node of the first Map.
        #  Furthermore, we also handle the output.
        for oedge in state.out_edges(node_to_replicate):
            dst_node = oedge.dst

            if dst_node is first_map_exit:
                # A connection to the MapExit node is indicating that we now write to
                #  the output, which is ultimately the intermediate. But since we can
                #  not use it we have to create a replacement.
                assert dst_node not in nodes_to_replicate
                assert output_name is None

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
                    memlet=dace.Memlet.from_array(output_name, output_desc),
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

    assert len(input_node_map) == len(set(input_node_map.values()))
    assert output_name is not None

    return nsdfg, input_node_map, output_name


def _extract_generating_dataflow_for_inlining(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> Optional[
    tuple[
        set[dace_nodes.Node],
        dace_nodes.MapExit,
        dace_nodes.AccessNode,
        dace_nodes.MapEntry,
        dace_sbs.Range,
        dict[str, dace_sym.SymbolicType],
    ]
]:
    """Finds the "generating subgraph" of `edge`.

    The function will trace the dataflow back and find all nodes that contribute to
    this single edge. This means all of these nodes are needed to generate the data
    that is transmitted by `edge`. Currently there are the following restrictions:
    - `edge` must be an outgoing edge of a top level MapEntry node.
    - The data of `edge` must come from an intermediate AccessNode, which is
        generated by a Map. Thus the data, that is consumed by `edge` must be generated
        by another Map.
    - The returned subgraph only contains the nodes inside the top Map.

    If any of these requirements are not met the function returns `None`.
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
    #  second Map are used for the access (exception see bellow). These rule are used
    #  to ensure that one iteration of the first Map are enough to satisfy all data
    #  dependencies of the second Map.
    avial_second_map_param = {param for param in second_map_entry.map.params}
    full_symbolic_access = True
    for start, stop, step in edge.data.src_subset:
        if (step != 1) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return None

        if str(start).isdigit():
            # The start index is a digit, we require that the end is also a digit.
            if not str(stop).isdigit():
                return None

            # There is not a full symbolic access.
            full_symbolic_access = False

        else:
            # The start index is a symbol. We require that it is an unused parameter
            #  and that the stop is also that parameter.
            start_symbols = {str(sym) for sym in start.free_symbols}
            if len(start_symbols) != 1:
                return None
            elif start_symbols.issubset(avial_second_map_param):
                stop_symbols = {str(sym) for sym in stop.free_symbols}
                if start_symbols != stop_symbols:
                    return None
                avial_second_map_param.remove(next(iter(start_symbols)))

    # Usually we require that all parameters of the second Map are used for the
    #  access. The main reason is to prevent some slicing, i.e. to ensure that
    #  the access is not done later. However, this prevents the second Map to read
    #  from a Map that has one dimension less. Thus as an exception we allow that
    #  in case all accesses are done using parameters, then not all of them have
    #  to be used.
    if (not full_symbolic_access) and (len(avial_second_map_param) != 0):
        return None

    # We restrict us to them because these node types are kind of simple to modify.
    #  However, even the Tasklet could be hard to handle. The hardest is possible
    #  the nested SDFG. Which is possible to handle in some cases but not in all
    #  cases.
    # TODO(phimuell): Allow more and allow NestedSDFGs in certain cases.
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
    # TODO(phimuell): Improve that.
    consumer_subset = edge.data.src_subset
    consumer_shape = consumer_subset.size()
    producer_subset = writing_edges[0].data.get_dst_subset(writing_edges[0], state)
    producer_shape = producer_subset.size()
    for cshp, pshp in zip(consumer_shape, producer_shape, strict=True):
        if (cshp == pshp) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            continue
        if (cshp < pshp) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            continue
        return None

    # Due to a restriction in `_insert_nested_sdfg()` we can only handle scalars that
    #  are exchanged.
    # TODO(phimuell): Remove as soon as possible.
    if (producer_subset.num_elements() == 1) == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
        return None
    if not isinstance(edge.dst, (dace_nodes.AccessNode, dace_nodes.Tasklet)):
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

    # Fore syntactical reasons we require that all outgoing edges of the nodes in
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
        first_map_exit,
        intermediate_node,
        edge.src,
        consumer_shape,
        first_map_param_mapping,
    )


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

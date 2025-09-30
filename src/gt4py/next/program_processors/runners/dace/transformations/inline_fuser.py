# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
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
        dace.Memlet.from_array(sdfg.arrays[outer_output_name]),
    )

    # Now adapt the range from where we read.
    # NOTE: This is currently done in a sub optimal way, and only allows for the
    #   transmission of one element that is immediately consumed. This is currently
    #   guaranteed by the `_extract_generating_dataflow_for_inlining()` function.
    # TODO(phimuell): Lift this requirement and handle subsets and strides.
    state.add_edge(
        outer_output_node,
        None,
        edge_to_replace.dst,
        edge_to_replace.dst_conn,
        dace.Memlet(
            data=outer_output_name,
            subset=dace_sbs.Range.from_array(sdfg.arrays[outer_output_name]),
            other_subset=edge_to_replace.data.dst_subset,
        ),
    )

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
            outside_input_node: dace_nodes.AccessNode = state.memlet_path(iedge)[0]
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

        # Now handle the output of the node.
        output_name: Optional[str] = None
        for oedge in state.out_edges(node_to_replicate):
            dst_node = oedge.dst

            if dst_node is first_map_exit:
                # A connection to the MapExit node is indicating that we now write to
                #  the output, which is ultimately the intermediate. But since we can
                #  not use it we have to create a replacement.
                assert dst_node not in nodes_to_replicate
                assert output_name is None

                output_name, output_desc = nsdfg.add_array(
                    output_name,
                    shape=exchange_subset.size(),
                    dtype=intermediate_node.dtype,
                    debuginfo=intermediate_node.debuginfo,
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
            else:
                # A normal connection to a node that is also mapped into the nested
                #  SDFG. We just replicate it.
                assert dst_node in nodes_to_replicate
                nstate.add_edge(
                    replicated_node_map[iedge.src],
                    iedge.src_conn,
                    replicated_node_map[iedge.dst],
                    iedge.dst_conn,
                    dace.Memlet.from_memlet(iedge.data),
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
    if scope_dict[edge.src] is not None:
        return None

    # Check the Memlet.
    if edge.data.src_subset is None:
        return None
    if any((step != 1) == True for _, _, step in edge.data.src_subset):  # noqa: E712 [true-false-comparison]  # SymPy comparison
        return None
    if edge.data.wcr is not None:
        return None

    # TODO: Check that only one iteration is needed or the volume.

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
        breakpoint()
        return None

    first_map_param_mapping = _get_first_map_parameter_to_second_map_mapping(
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
    producer_shape = writing_edges[0].data.dst_subset.size()
    for cshp, pshp in zip(consumer_shape, producer_shape, strict=True):
        if (cshp == pshp) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            continue
        if (cshp < pshp) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            continue
        return None

    # Due to a restriction in `_insert_nested_sdfg()` we can only handle scalars
    #  that are immediately consumed.
    # TODO(phimuell): Remove as soon as possible.
    if (consumer_subset.num_elements() == 1) == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
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

    return (
        generating_nodes,
        first_map_exit,
        intermediate_node,
        edge.src,
        consumer_shape,
        first_map_param_mapping,
    )


def _get_first_map_parameter_to_second_map_mapping(
    first_map: dace_nodes.Map,
    write_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    second_map: dace_nodes.Map,
    read_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
) -> dict[str, dace_sym.SymbolicType]:
    """Expresses the values of the first Map's parameter name in terms of the second map."""
    assert first_map.get_param_num() == second_map.get_param_num()
    assert isinstance(write_edge.dst, dace_nodes.MapExit)
    assert write_edge.data.dst_subset is not None
    assert isinstance(read_edge.src, dace_nodes.MapEntry)
    assert read_edge.data.src_subset is not None

    parameter_mapping: dict[str, str] = {}

    # Currently we only consider the lower bound and nothing else, which is probably
    #  okay for the GT4Py setting.
    write_accesses = write_edge.data.dst_subset.min_element()
    read_accesses = read_edge.data.src_subset.min_element()

    # We assume that the different dimensions are independent, i.e. each parameter is
    #  used to access exactly one dimension, this rules out accesses such as
    #  `a[i + j, i - j]`. This is okay for GT4Py. However, we have now to find out
    #  which parameter acts in which dimensions.
    first_mapping = _find_dimensions_access(first_map.params, write_accesses)
    second_mapping = _find_dimensions_access(second_map.params, read_accesses)

    # If we read the data then we use the parameters of the second Map. However, we
    #  must now figuring out to which value of the first map this corresponds to.
    for i in range(first_map.get_param_num()):
        write_param = first_map.params[first_mapping[i]]
        write_access = write_accesses[first_mapping[i]]
        read_access = read_accesses[second_mapping[i]]

        write_param_uniq = dace_sym.pystr_to_symbolic(f"__unique_param_{i}_{write_param!s}")
        write_access_uniq = write_access.subs({write_param: write_param_uniq})

        res = sympy.solve(sympy.Eq(write_access_uniq, read_access), write_access_uniq)
        if len(res) != 1:
            raise ValueError(f"Could not solve '{read_access} == {write_access}'")
        parameter_mapping[write_param] = res[0]

    return parameter_mapping


def _find_dimensions_access(
    params: Sequence[str], accesses: Iterable[dace_sym.SymbolicType]
) -> dict[int, str]:
    result_mapping: dict[int, str] = {}
    aparams: set[str] = set(params)

    for i, access in enumerate(accesses):
        free_symbs = {str(fs) for fs in access.free_symbols}
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

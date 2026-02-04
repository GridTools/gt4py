# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
from typing import Any, Sequence, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, utils as dace_sdutils


@dace_properties.make_properties
class ConcatWhereCopyToMap(dace_transformation.SingleStateTransformation):
    """NA"""

    node_a1 = dace_transformation.PatternNode(dace_nodes.AccessNode)  # Needed to speed up matching.
    concat_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: dict[dace.SDFG, set[str]]

    def __init__(
        self,
        *args: Any,
        single_use_data: dict[dace.SDFG, set[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # ALLOW NONE.
        self._single_use_data = single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.node_a1,
                cls.concat_node,
                cls.map_entry,
            )
        ]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_entry: dace_nodes.MapEntry = self.map_entry
        concat_node: dace_nodes.AccessNode = self.concat_node

        # Must be on the top scope.
        if graph.scope_dict()[map_entry] is not None:
            return False

        # Currently we only allow that there is one consumer, might be lifted but for
        #  sure it must be single use data (check is done at the very end).
        if graph.out_degree(concat_node) != 1:
            return False
        if not concat_node.desc(sdfg).transient:
            return False
        if (
            self._single_use_data is not None
            and concat_node.data not in self._single_use_data[sdfg]
        ):
            return False

        # The concat where node must act as a converging point and no Map can write
        #  into it directly.
        # TODO(phimuell): Consider to lift the restriction that `concat_node` gets
        #   data from other nodes, also allow that Maps write into it by creating
        #   a new intermediate node.
        if graph.in_degree(concat_node) < 2:
            return False
        if any(
            not isinstance(iedge.src, dace_nodes.AccessNode)
            for iedge in graph.in_edges(concat_node)
        ):
            return False

        # Check the consumers, they can only read one element and all must read the same.
        consumer_edges = _find_consumer_edges(graph, concat_node, map_entry)
        if len(consumer_edges) != 1:
            return False  # TODO(phimuell): Lift this restriction.
        read_subsets: list[dace_sbs.Subset] = [
            consumer_edge.data.get_src_subset(consumer_edge, graph)
            for consumer_edge in consumer_edges
        ]
        first_read_subset = read_subsets[0]
        if not all(
            (read_subset.num_elements() == 1) == True and (read_subset == first_read_subset) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            for read_subset in read_subsets
        ):
            return False

        # Check if it is single use data, performed at the end because it is the
        #  most expensive one, if not cached.
        if self._single_use_data is None:
            raise NotImplementedError("`single_use_data` must be passed.")
        else:
            single_use_data = self._single_use_data

        if concat_node.data not in single_use_data[sdfg]:
            return False

        return True


def gt_replace_concat_where_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> list[dace_graph.MultiConnectorEdge[dace.Memlet]]:
    """Performs the replacement

    Todo:
        - Nested scopes.
        - On global scope read of single element.
        - Consumer in multiple scopes.
        - Multiple consumer in different scopes.
        - Nested Maps.
        - Nested SDFG (1 level).
        - Nested SDFG multiple levels.
    """

    assert state.out_degree(concat_node) == 1
    assert all(oedge.dst is map_entry for oedge in state.out_edges(concat_node))

    # Collect all edges that generate data that converges at the concat where node.
    converging_edges = list(state.in_edges(concat_node))
    # TODO(phimuell): Lift this restriction by creating intermediate AccessNodes
    assert all(isinstance(iedge.src, dace_nodes.AccessNode) for iedge in converging_edges)
    converging_edges = sorted(converging_edges, key=lambda iedge: iedge.src.data)

    # These are all the consumers that we need to modify.
    consumer_edges = _find_consumer_edges(state, concat_node, map_entry)
    assert len(consumer_edges) > 0

    # Generate the specification of the producer and ensures that they are
    #  accessible inside the scopes.
    producer_specs = _create_prducer_specs(
        state=state,
        sdfg=sdfg,
        consumer_edges=consumer_edges,
        converging_edges=converging_edges,
        map_entry=map_entry,
    )

    # Now process them.
    new_consumers: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = []
    for consumer_edge in consumer_edges:
        new_consumer = _replace_single_read(
            state=state,
            sdfg=sdfg,
            consumer_edge=consumer_edge,
            producer_specs=producer_specs,
            tag=f"{concat_node.data}_{map_entry.label}",
        )
        new_consumers.append(new_consumer)

    assert state.out_degree(concat_node) == 0
    state.remove_node(concat_node)
    sdfg.remove_data(concat_node.data, validate=False)

    # TODO(phimuell): Run Memlet propagation.

    return new_consumers


@dataclasses.dataclass
class _ProducerSpec:
    """Describes how a `concat_where` converges at a name.

    Args:
        data_name: The name of the data descriptor that provided the original information.
        full_shape: The full shape of the data descriptor.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        data_source: For every final consumer node stores the location where the
            full data of `data_name` can be accessed.

    Note:
        The format of `data_source` seems a bit redundant, however it ensures that
        the information is there even if the SDFG is invalid, i.e. during operation.
        Furthermore, it also allows to handle cases that appears in nested SDFG
        scenarios.
    """

    data_name: str
    full_shape: dace_sbs.Range
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    data_source: dict[dace_nodes.Node, tuple[dace_nodes.Node, str]]


def _create_prducer_specs(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    consumer_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]],
    converging_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]],
    map_entry: dace_nodes.MapEntry,
) -> list[_ProducerSpec]:
    """Helper function that creates the producer specifications.

    In addition this function will also ensure that the data can be accessed where
    it is needed. This function should only be called on the top level state.

    Args:
        state: The state in which we operate.
        sdfg: The sdfg on which we operate.
        consumer_edges: All edges that consume the concat where data.
        converging_edges: The edges that define, the concat where node, i.e.
            all incoming edges to the concat where node.
        map_entry: The top level Map entry.
    """

    # These are the scopes in which (legal) accesses to the `concat_where`. We now have
    #  to make the individual data, i.e. the producer available there.
    scope_dict = state.scope_dict()
    accessed_scopes: set[dace_nodes.Node] = {
        scope_dict[consumer_edge.dst] for consumer_edge in consumer_edges
    }
    assert not any(scope is None for scope in accessed_scopes)

    # Describes where to find data. The list has the same order as `producer_specs`.
    #  The `dict`s map a scope node SDFGState pair to the connector name, at that
    #  scope node where the data can be found. Later it is the basis to compute the
    #  `data_source` attribute.
    scope_sources: list[dict[tuple[dace_nodes.Node, dace.SDFGState], str]] = []

    # Set up the description of the concat where, with exception of the `data_source`
    #  attribute which is populated later. Furthermore, we also map the producer data,
    #  i.e. the data that composes the concat where, into `map_entry`. Nested levels
    #  are handled later.
    producer_specs: list[_ProducerSpec] = []
    already_mapped_data: dict[str, int] = {}
    for converging_edge in converging_edges:
        assert isinstance(converging_edge.src, dace_nodes.AccessNode)
        producer_node = converging_edge.src
        data_name = producer_node.data
        full_shape = dace_sbs.Range.from_array(producer_node.desc(sdfg))
        offset = copy.deepcopy(converging_edge.data.get_src_subset(converging_edge, state))
        subset = copy.deepcopy(converging_edge.data.dst_subset)
        if data_name in already_mapped_data:
            top_level_data_source = scope_sources[already_mapped_data[data_name]][
                (map_entry, state)
            ]
        else:
            # Try to find an edge that maps the producer into the Map.
            edge_that_maps_data_into_map = next(
                (
                    oedge
                    for oedge in state.out_edges(producer_node)
                    if (
                        (not oedge.data.is_empty())
                        and oedge.dst is map_entry
                        and oedge.dst_conn.startswith("IN_")
                        and oedge.data.get_src_subset(oedge, state).covers(full_shape)
                    )
                ),
                None,
            )
            if edge_that_maps_data_into_map is not None:
                top_level_data_source = "OUT_" + edge_that_maps_data_into_map.dst_conn[3:]
            else:
                new_conn_name = map_entry.next_connector(data_name)
                state.add_edge(
                    producer_node,
                    None,
                    map_entry,
                    "IN_" + new_conn_name,
                    dace.Memlet(data=data_name, subset=copy.deepcopy(full_shape)),
                )
                # The out connector is dangling, but we will handle it later.
                map_entry.add_scope_connectors(new_conn_name)
                top_level_data_source = "OUT_" + new_conn_name
            already_mapped_data[data_name] = len(producer_specs)  # Index in the future.
        producer_specs.append(
            _ProducerSpec(
                data_name=data_name,
                full_shape=full_shape,
                offset=offset,
                subset=subset,
                data_source={},  # Filled in later.
            )
        )
        scope_sources.append({(map_entry, state): top_level_data_source})

    # Process the nested scopes.
    handled_scopes: set[dace_nodes.Node] = {map_entry}
    for needed_scope in accessed_scopes:
        _recursive_fill_scope_sources(
            state=state,
            producer_specs=producer_specs,
            scope_sources=scope_sources,
            scope_to_handle=needed_scope,
            handled_scopes=handled_scopes,
        )

    # Now fill in the data sources.
    for consumer_edge in consumer_edges:
        final_consumer = consumer_edge.dst
        consumer_scope = scope_dict[final_consumer]
        for scope_source, producer_spec in zip(scope_sources, producer_specs):
            producer_spec.data_source[final_consumer] = (
                consumer_scope,
                scope_source[(consumer_scope, state)],
            )

    return producer_specs


def _recursive_fill_scope_sources(
    state: dace.SDFGState,
    producer_specs: list[_ProducerSpec],
    scope_sources: list[dict[dace_nodes.Node, str]],
    scope_to_handle: dace_nodes.Node,
    handled_scopes: set[dace_nodes.Node],
) -> None:
    """Helper function of `_create_prducer_specs()` that populate nested scopes."""
    if scope_to_handle in handled_scopes:
        return
    assert scope_to_handle is not None
    assert isinstance(scope_to_handle, dace_nodes.MapEntry)

    # Ensures that the parents are handled
    parent_scope = state.scope_dict()[scope_to_handle]
    if parent_scope not in handled_scopes:
        _recursive_fill_scope_sources(
            state=state,
            producer_specs=producer_specs,
            scope_sources=scope_sources,
            scope_to_handle=parent_scope,
            handled_scopes=handled_scopes,
        )
    assert parent_scope in handled_scopes

    # Now handle the actual scope.
    for producer_spec, scope_source in zip(producer_specs, scope_sources):
        assert (scope_to_handle, state) not in scope_source
        source_conn = scope_source[(parent_scope, state)]
        new_conn_name = scope_to_handle.next_connector(producer_spec.data_name)
        state.add_edge(
            parent_scope,
            source_conn,
            scope_to_handle,
            "IN_" + new_conn_name,
            dace.Memlet(
                data=producer_spec.data_name, subset=copy.deepcopy(producer_spec.full_shape)
            ),
        )
        # The out connector is dangling, but we will handle it later.
        scope_to_handle.add_scope_connectors(new_conn_name)

        # Do we need to spare the pair, the connector name should be enough?
        scope_source[(scope_to_handle, state)] = f"OUT_{new_conn_name}"
    handled_scopes.add(scope_to_handle)


def _replace_single_read(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    consumer_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    producer_specs: Sequence[_ProducerSpec],
    tag: str,
) -> dace_graph.MultiConnectorEdge[dace.Memlet]:
    """Performs the replacement of a single read, defined by `consumer_edge`.

    Essentially the function will replace the access to a converging `concat_where`
    node, i.e. the node where multiple writes converges, with a Tasklet that selects
    the data that should be read. This means only the read of a single element can
    be handled in this way.

    Note that the destination of `consumer_edge` will remain in the SDFG, instead
    the function will create a new data descriptor, into which the result will be
    written that will then be connected to `consumer_edge`, however, `consumer_edge`
    will be removed.

    Args:
        state: The state in which we operate.
        sdfg: The parent SDFG on which we operate.
        consumer_edges: The consumer edge that should be replaced.
        producer_specs: Descriptions of how the `concat_where` converges.
        tag: Used to give newly created elements a unique name.
    """

    assert consumer_edge.data.wcr is None
    assert consumer_edge.data.subset.num_elements() == 1

    tlet_inputs: list[str] = [f"__inp{i}" for i in range(len(producer_specs))]
    tlet_output = "__out"

    select_conds: list[str] = []
    prod_accesses: list[str] = []
    consume_subset = consumer_edge.data.get_src_subset(consumer_edge, state)
    for prod_spec in producer_specs:
        prod_subset = prod_spec.subset
        prod_offsets = prod_spec.offset

        this_select_cond: list[str] = []
        this_prod_access: list[str] = []
        for dim in range(consume_subset.dims()):
            consumer_access = consume_subset[dim][0]
            prod_supply_start = prod_subset[dim][0]
            prod_supply_end = prod_subset[dim][1]  # Inclusive end.
            prod_offset = prod_offsets[dim][0]

            this_select_cond.append(
                f"((({prod_supply_start}) <= ({consumer_access})) and (({consumer_access}) <= ({prod_supply_end})))"
            )
            this_prod_access.append(
                f"(({prod_offset}) + (({consumer_access}) - ({prod_supply_start})))"
            )
        prod_accesses.append(", ".join(this_prod_access))
        select_conds.append(" and ".join(this_select_cond))

    if len(producer_specs) == 2:
        tlet_code = f"{tlet_output} = {tlet_inputs[0]}[{prod_accesses[0]}] if ({select_conds[0]}) else {tlet_inputs[1]}[{prod_accesses[1]}]"
    else:
        tlet_code_lines: list[str] = []
        # TODO(phimuell): Turn this into nested `?:` expressions.
        for i in range(len(producer_specs) - 1):
            tlet_code_lines.append(
                f"{'if' if i == 0 else 'elif'} ({select_conds[i]}):\n\t{tlet_output} = {tlet_inputs[i]}[{prod_accesses[i]}]"
            )
        tlet_code_lines.append(f"else:\n\t{tlet_output} = {tlet_inputs[-1]}[{prod_accesses[-1]}]")
        tlet_code = "\n".join(tlet_code_lines)

    names_of_existing_tasklets = {
        node.label for node in state.nodes() if isinstance(node, dace_nodes.Tasklet)
    }
    tasklet_name = dace.utils.find_new_name(
        f"concat_where_taskelt_{tag}", names_of_existing_tasklets
    )
    concat_where_tasklet = state.add_tasklet(
        tasklet_name,
        inputs=set(tlet_inputs),
        outputs={tlet_output},
        code=tlet_code,
    )

    final_consumer = consumer_edge.dst
    for i in range(len(tlet_inputs)):
        producer_spec = producer_specs[i]
        state.add_edge(
            producer_spec.data_source[final_consumer][0],
            producer_spec.data_source[final_consumer][1],
            concat_where_tasklet,
            tlet_inputs[i],
            dace.Memlet(
                data=producer_spec.data_name,
                subset=copy.deepcopy(producer_spec.full_shape),
            ),
        )

    # Instead of replacing `final_consumer` we create an intermediate output node.
    #  This removes the need to reconfigure the dataflow.
    intermediate_data, _ = sdfg.add_scalar(
        name=f"__gt4py_concat_where_mapper_temp_{tag}",
        dtype=sdfg.arrays[consumer_edge.data.data].dtype,
        transient=True,
        find_new_name=True,
    )
    intermediate_node = state.add_access(intermediate_data)

    state.add_edge(
        concat_where_tasklet,
        tlet_output,
        intermediate_node,
        None,
        dace.Memlet(data=intermediate_data, subset="0"),
    )

    # Find out what `other_subset` of the new Memlet should be.
    if (
        isinstance(consumer_edge.dst, dace_nodes.AccessNode)
        and consumer_edge.dst.data == consumer_edge.data.data
    ):
        other_subset = consumer_edge.data.subset
    else:
        other_subset = consumer_edge.data.other_subset

    # Create the edge between the new intermediate and the old consumer. We ignore
    #  some properties of the Memlet here, such as dynamic (`wcr` was handled above).
    new_consumer_edge = state.add_edge(
        intermediate_node,
        None,
        final_consumer,
        consumer_edge.dst_conn,
        dace.Memlet(
            data=intermediate_data,
            subset="0",
            other_subset=other_subset,
        ),
    )

    # Now remove the old Memlet path.
    dace_sdutils.remove_edge_and_dangling_path(state, consumer_edge)

    return new_consumer_edge


def _find_consumer_edges(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> list[dace_graph.MultiConnectorEdge[dace.Memlet]]:
    """Find all edges that reads from `concat_node` inside the Map defined by `map_entry`."""
    assert state.out_degree(concat_node) == 1
    outer_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = state.edges_between(
        concat_node, map_entry
    )[0]
    assert outer_edge.dst_conn.startswith("IN_")

    consumer_edges: list[dace_graph.MultiConnectorEdge] = []
    for inner_edge in state.out_edges_by_connector(map_entry, "OUT_" + outer_edge.dst_conn[3:]):
        consumer_edges.extend(state.memlet_tree(inner_edge).leaves())

    # Sort the edges according to the destination.
    return sorted(consumer_edges, key=lambda iedge: str(iedge.dst))

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

        # IMPLEMENT ME

        return False


def gt_replace_concat_where_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> None:
    """Performs the replacement

    Todo:
        - On global scope read of single element (needed if it happens in the nested level).
        - Multiple consumer but same element, dedublication.
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
    consumer_specs = _find_consumer_specs(state, concat_node, map_entry)
    assert len(consumer_specs) > 0

    # Generate the specification of the producer and ensures that they are
    #  accessible inside the scopes.
    producer_specs = _create_prducer_specs(
        state=state,
        sdfg=sdfg,
        consumer_specs=consumer_specs,
        converging_edges=converging_edges,
        map_entry=map_entry,
    )

    # Now process them.
    for consumer_spec in consumer_specs:
        _replace_single_read(
            consumer_spec=consumer_spec,
            producer_specs=producer_specs,
            tag=f"{concat_node.data}_{map_entry.label}",
        )

    assert state.out_degree(concat_node) == 0
    state.remove_node(concat_node)
    sdfg.remove_data(concat_node.data, validate=False)

    # TODO(phimuell): Run Memlet propagation.


@dataclasses.dataclass(frozen=True)
class _FinalConsumerSpec:
    """Describes the final consumer.

    Attributes:
        edge:  The final edge.
        consumer: The final consumer node.
        state: The state in which `edge` is.
        sdfg:  The SDFG containing `state`.
    """

    edge: dace_graph.MultiConnectorEdge[dace.Memlet]
    state: dace.SDFGState

    @property
    def sdfg(self) -> dace.SDFG:
        return self.state.sdfg

    @property
    def consumer(self) -> dace_nodes.Node:
        return self.edge.dst


@dataclasses.dataclass(frozen=True)
class _ProducerSpec:
    """Describes how a `concat_where` converges at a name.

    This class only describes a single source of the concat where. To make sense
    one needs a `list` of `_ProducerSpec`, one for each converging edges.

    Args:
        data_name: Name of the data descriptor.
        full_shape: The full shape of the data descriptor.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        data_source: Maps consumer specifications to a `tuple[Node, str]` instance
            that describes where the raw data of that producer can be obtained.

    Note:
        The format of `data_source` seems a bit redundant, however it ensures that
        the information is there even if the SDFG is invalid, and also allows
        to operate in nested SDFG scenario.
    """

    data_name: str
    full_shape: dace_sbs.Range
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    data_source: dict[_FinalConsumerSpec, tuple[dace_nodes.Node, str]]


def _create_prducer_specs(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    consumer_specs: list[_FinalConsumerSpec],
    converging_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]],
    map_entry: dace_nodes.MapEntry,
) -> list[_ProducerSpec]:
    """Helper function that creates the producer specifications.

    In addition this function will also ensure that the data can be accessed where
    it is needed. This function should only be called on the top level state.

    Args:
        state: The state in which we the concat where node is located.
        sdfg: The SDFG that contains `state`.
        consumer_specs: Specification of the consumers.
        converging_edges: The edges that define, the concat where node, i.e.
            all incoming edges to the concat where node.
        map_entry: The top level Map entry.
    """

    # No nested SDFGs scenario supported yet.
    assert all(consumer_spec.state is state for consumer_spec in consumer_specs)

    # These are the scopes in which (legal) accesses to the `concat_where`. We now have
    #  to make the individual data, i.e. the producer available there.
    scope_dict = state.scope_dict()
    accessed_scopes: set[dace_nodes.Node] = {
        scope_dict[consumer_spec.consumer] for consumer_spec in consumer_specs
    }
    assert not any(scope is None for scope in accessed_scopes)

    # Describes where to find data. The list has the same order as `producer_specs`.
    #  The `dict`s map a "scope node"-"SDFGState" pair to the connector name, at that
    #  scope node where the data can be found. Later it is the basis to compute the
    #  `data_source` attribute.
    scope_sources: list[dict[tuple[Union[dace_nodes.Node, None], dace.SDFGState], str]] = []

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
    for consumer_spec in consumer_specs:
        consumer_scope = scope_dict[consumer_spec.consumer]
        for scope_source, producer_spec in zip(scope_sources, producer_specs):
            producer_spec.data_source[consumer_spec] = (
                consumer_scope,
                scope_source[(consumer_scope, consumer_spec.state)],
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
    consumer_spec: _FinalConsumerSpec,
    producer_specs: Sequence[_ProducerSpec],
    tag: str,
) -> None:
    """Performs the replacement of a single read, defined by `consumer_spec`.

    Essentially the function will replace the access to a converging `concat_where`
    node, i.e. the node where multiple writes converges, with a Tasklet that selects
    the data that should be read. This means only the read of a single element can
    be handled in this way.

    Note that the final consumer node will remain in the SDFG, instead the function
    will create a new data descriptor, into which the result will be written that
    will then be connected to the final consumer. However, the edge will be removed.

    Args:
        consumer_spec: Describes a consumer.
        producer_specs: Descriptions of how the `concat_where` converges.
        tag: Used to give newly created elements a unique name.
    """

    assert consumer_spec.edge.data.wcr is None
    assert consumer_spec.edge.data.subset.num_elements() == 1

    tlet_inputs: list[str] = [f"__inp{i}" for i in range(len(producer_specs))]
    tlet_output = "__out"

    select_conds: list[str] = []
    prod_accesses: list[str] = []
    consume_subset = consumer_spec.edge.data.get_src_subset(consumer_spec.edge, consumer_spec.state)
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
        node.label for node in consumer_spec.state.nodes() if isinstance(node, dace_nodes.Tasklet)
    }
    tasklet_name = dace.utils.find_new_name(
        f"concat_where_taskelt_{tag}", names_of_existing_tasklets
    )
    concat_where_tasklet = consumer_spec.state.add_tasklet(
        tasklet_name,
        inputs=set(tlet_inputs),
        outputs={tlet_output},
        code=tlet_code,
    )

    for tlet_input, producer_spec in zip(tlet_inputs, producer_specs):
        consumer_spec.state.add_edge(
            producer_spec.data_source[consumer_spec][0],
            producer_spec.data_source[consumer_spec][1],
            concat_where_tasklet,
            tlet_input,
            dace.Memlet(
                data=producer_spec.data_name,
                subset=copy.deepcopy(producer_spec.full_shape),
            ),
        )

    # Instead of replacing `final_consumer` we create an intermediate output node.
    #  This removes the need to reconfigure the dataflow.
    intermediate_data, _ = consumer_spec.sdfg.add_scalar(
        name=f"__gt4py_concat_where_mapper_temp_{tag}",
        dtype=consumer_spec.sdfg.arrays[consumer_spec.edge.data.data].dtype,
        transient=True,
        find_new_name=True,
    )
    intermediate_node = consumer_spec.state.add_access(intermediate_data)

    consumer_spec.state.add_edge(
        concat_where_tasklet,
        tlet_output,
        intermediate_node,
        None,
        dace.Memlet(data=intermediate_data, subset="0"),
    )

    # Find out what `other_subset` of the new Memlet should be.
    if (
        isinstance(consumer_spec.consumer, dace_nodes.AccessNode)
        and consumer_spec.edge.dst.data == consumer_spec.edge.data.data
    ):
        other_subset = consumer_spec.edge.data.subset
    else:
        other_subset = consumer_spec.edge.data.other_subset

    # Create the edge between the new intermediate and the old consumer. We ignore
    #  some properties of the Memlet here, such as dynamic (`wcr` was handled above).
    consumer_spec.state.add_edge(
        intermediate_node,
        None,
        consumer_spec.consumer,
        consumer_spec.edge.dst_conn,
        dace.Memlet(
            data=intermediate_data,
            subset="0",
            other_subset=other_subset,
        ),
    )

    # Now remove the old Memlet path.
    dace_sdutils.remove_edge_and_dangling_path(consumer_spec.state, consumer_spec.edge)


def _find_consumer_specs(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> list[_FinalConsumerSpec]:
    """Find all edges that reads from `concat_node` inside the Map defined by `map_entry`."""
    assert state.out_degree(concat_node) == 1
    outer_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = state.edges_between(
        concat_node, map_entry
    )[0]
    assert outer_edge.dst_conn.startswith("IN_")

    consumer_specs: list[_FinalConsumerSpec] = []
    for inner_edge in state.out_edges_by_connector(map_entry, "OUT_" + outer_edge.dst_conn[3:]):
        consumer_specs.extend(
            (
                _FinalConsumerSpec(edge=edge, state=state)
                for edge in state.memlet_tree(inner_edge).leaves()
            )
        )

    # Sort the edges according to the destination.
    return sorted(consumer_specs, key=lambda cspec: str(cspec.consumer))

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
from typing import Any, Optional, Sequence, TypeAlias, Union

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
    tag: Optional[str] = None,
) -> None:
    """Performs the replacement

    User must make sure that `concat_where` is single use data.

    Todo:
        - Symbol mapping.
        - On global scope read of single element (needed if it happens in the nested level).
        - Multiple consumer but same element, dedublication.
        - Nested Maps.
        - Nested SDFG (1 level).
        - Nested SDFG multiple levels.
    """

    # These are all the consumers that we need to modify.
    find_consumer_result = _find_consumer_specs_single_level(state, concat_node)
    assert find_consumer_result is not None
    consumer_specs, descening_points = find_consumer_result
    assert len(consumer_specs) > 0
    assert len(descening_points) == 0

    scope_dict = state.scope_dict()
    initial_producer_specs, base_scope_sources = _setup_initial_producer_description_for(
        sdfg=sdfg,
        state=state,
        concat_node=concat_node,
        scope_dict=scope_dict,
    )

    # We have to ensures that the "descending points" have access to the data.
    producer_specs = _map_data_into_nested_scopes(
        state=state,
        scope_dict=scope_dict,
        initial_producer_specs=initial_producer_specs,
        base_scope_sources=base_scope_sources,
        consumer_specs=consumer_specs + descening_points,
    )

    # Now process them.
    for consumer_spec in consumer_specs:
        _replace_single_read(
            state=state,
            sdfg=sdfg,
            consumer_spec=consumer_spec,
            producer_specs=producer_specs,
            tag=f"{concat_node.data}_{'' if tag is None else tag}",
        )

    assert state.out_degree(concat_node) == 0
    state.remove_node(concat_node)
    sdfg.remove_data(concat_node.data, validate=False)

    # TODO(phimuell): Run Memlet propagation.


_ScopeLocation: TypeAlias = Union[dace_nodes.MapEntry, None]
"""Defines a scope in the DaCe term.

This is either a `MapEntry` if nested or `None` if something is at the global scope.
"""


@dataclasses.dataclass(frozen=True)
class _DataSource:
    """Describes where (full producer) data can be found.

    Essentially a pair consisting of a Node and the required connector that must be
    used to access it.
    """

    node: Union[dace_nodes.AccessNode, dace_nodes.MapEntry]
    conn: Union[str, None]

    def __post_init__(self) -> None:
        assert isinstance(self.node, (dace_nodes.AccessNode, dace_nodes.MapEntry))

    def __hash__(self) -> int:
        return hash((self.node, self.conn))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DataSource):
            return self.node == other.node and self.conn == other.conn
        return NotImplemented


@dataclasses.dataclass(frozen=True)
class _FinalConsumerSpec:
    """Describes the final consumer."""

    edge: dace_graph.MultiConnectorEdge[dace.Memlet]

    def consumed_subset(self, state: dace.SDFGState) -> dace_sbs.Range:
        return self.edge.data.get_src_subset(self.edge, state)

    @property
    def consumer(self) -> dace_nodes.Node:
        return self.edge.dst

    def __hash__(self) -> int:
        return hash(self.edge)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dace_graph.MultiConnectorEdge):
            return self.edge == other
        elif isinstance(other, _FinalConsumerSpec):
            return self.edge == other.edge
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _FinalConsumerSpec):
            return (str(self.consumer), str(self.edge)) < (str(self.consumer), str(other.edge))
        return NotImplemented


@dataclasses.dataclass(frozen=True)
class _ProducerSpec:
    """Describes how a `concat_where` converges at a name.

    This class only describes a single source of the concat where. To make sense
    one needs a `list` of `_ProducerSpec`, one for each converging edges.

    Args:
        data_name: Name of the data descriptor.
        full_shape: The full shape of the data descriptor.
        dtype: The Type of the data.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        data_source: Maps each consumer to the data source location.

    Note:
        The format of `data_source` seems a bit redundant, however, it ensures that
        the information is there even if the SDFG is invalid, and also allows
        to operate in nested SDFG scenario.
    """

    data_name: str
    full_shape: dace_sbs.Range
    dtype: dace.dtypes.typeclass
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    data_source: dict[_FinalConsumerSpec, _DataSource]

    def __copy__(self) -> "_ProducerSpec":
        return _ProducerSpec(
            data_name=self.data_name,
            full_shape=copy.copy(self.full_shape),
            dtype=self.dtype,
            offset=copy.copy(self.offset),
            subset=copy.copy(self.subset),
            data_source=self.data_source.copy(),
        )

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _ProducerSpec):
            return (self.data_name, str(self.subset)) < (other.data_name, str(other.subset))
        return NotImplemented

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ProducerSpec):
            return id(self) == id(other)
        return NotImplemented


def _setup_initial_producer_description_for(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    scope_dict: Any,
) -> tuple[list[_ProducerSpec], list[_DataSource]]:
    """Sets up the producer specs.

    The function expects that `concat_node` is the result of a `concat_where`
    expression. Furthermore, it is assumed that it is single use data, i.e. not
    used anywhere else. This function will then create an intial specification
    for the replacement.

    It also creates an initial description of the scope source, which is a helper
    construct to compute `_ProducerSpec.data_source`. However it is only valid
    for the state (and SDFG level) `concat_node` resides in. To process different
    states (which means inside nested SDFGs) it is not suited and
    `_setup_base_scope_sources_in_state()` has to be used first.

    To fully populate the production specification it has to be passed to
    `_map_data_into_nested_scopes()`.
    """

    # First collect the edges that define the concat where expression. We currently
    #  assume that all of them came from AccessNodes, but this is a restriction that
    #  could be lifted.
    converging_edges = list(state.in_edges(concat_node))
    assert all(isinstance(iedge.src, dace_nodes.AccessNode) for iedge in converging_edges)

    assert len(converging_edges) >= 2
    assert isinstance(converging_edges[0].dst, dace_nodes.AccessNode)
    assert scope_dict[converging_edges[0].dst] is None
    concat_node = converging_edges[0].dst
    concat_dtype = concat_node.desc(sdfg).dtype

    base_scope_sources_unordered: dict[_ProducerSpec, _DataSource] = {}
    producer_specs: list[_ProducerSpec] = []
    for converging_edge in converging_edges:
        assert isinstance(converging_edge.src, dace_nodes.AccessNode)
        assert converging_edge.dst is concat_node
        producer_node = converging_edge.src
        producer_specs.append(
            _ProducerSpec(
                data_name=producer_node.data,
                full_shape=dace_sbs.Range.from_array(producer_node.desc(sdfg)),
                dtype=concat_dtype,
                offset=copy.deepcopy(converging_edge.data.get_src_subset(converging_edge, state)),
                subset=copy.deepcopy(converging_edge.data.dst_subset),
                data_source={},
            )
        )
        base_scope_sources_unordered[producer_specs[-1]] = _DataSource(producer_node, None)

    # Now sort the producer specification and then apply the same order to the scope sources.
    producer_specs = sorted(producer_specs)
    base_scope_sources = [base_scope_sources_unordered[prod_spec] for prod_spec in producer_specs]

    return producer_specs, base_scope_sources


def _map_data_into_nested_scopes(
    state: dace.SDFGState,
    scope_dict: Any,
    base_scope_sources: Sequence[_DataSource],
    initial_producer_specs: Sequence[_ProducerSpec],
    consumer_specs: Sequence[_FinalConsumerSpec],
) -> list[_ProducerSpec]:
    """Complete the production specifications and ensures that the data is available in nested scopes.

    The function will use the information passed by `base_scope_sources` and
    `initial_producer_specs` (which were obtained by a previous call to
    `_setup_initial_producer_description_for()` or `_setup_base_scope_sources_in_state()`)
    to compute a full producer description, which is also returned. In addition it will
    make sure that the data is available in the nested scopes, such that the
    replacement, performed by `_replace_single_read()` can be performed.
    """
    # These are the scopes in which (legal) accesses to the `concat_where`. We now have
    #  to make the individual data, i.e. the producer available there.
    accessed_scopes: set[_ScopeLocation] = {
        scope_dict[consumer_spec.consumer] for consumer_spec in consumer_specs
    }

    producer_specs = [copy.copy(producer_spec) for producer_spec in initial_producer_specs]
    scope_sources = [
        {scope_dict[base_scope_source.node]: base_scope_source}
        for base_scope_source in base_scope_sources
    ]

    # Process the nested scopes.
    for needed_scope in accessed_scopes:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_dict=scope_dict,
            scope_to_handle=needed_scope,
            producer_specs=producer_specs,
            scope_sources=scope_sources,
        )

    # Now fill in the data sources, i.e. expand `scope_source` into `_ProducerSpec.data_source`.
    for consumer_spec in consumer_specs:
        consumer_scope = scope_dict[consumer_spec.consumer]
        for scope_source, producer_spec in zip(scope_sources, producer_specs):
            producer_spec.data_source[consumer_spec] = scope_source[consumer_scope]

    return producer_specs


def _map_data_into_nested_scopes_impl(
    state: dace.SDFGState,
    scope_dict: Any,
    scope_to_handle: _ScopeLocation,
    producer_specs: list[_ProducerSpec],
    scope_sources: list[dict[_ScopeLocation, _DataSource]],
) -> None:
    """Helper function of `_map_data_into_nested_scopes_impl()` that populate nested scopes."""
    if scope_to_handle in scope_sources[0]:
        return

    # Check if the parent scope was handled before, if not handle it.
    parent_scope = scope_dict[scope_to_handle]
    if parent_scope not in scope_sources[0]:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_dict=scope_dict,
            scope_to_handle=parent_scope,
            producer_specs=producer_specs,
            scope_sources=scope_sources,
        )
    assert parent_scope in scope_sources[0]

    # On the top level perform dedublication of inputs.
    already_mapped_data: dict[str, int] = {}
    for i, (producer_spec, scope_source) in enumerate(zip(producer_specs, scope_sources)):
        if parent_scope is not None:
            # Nested scopes, just pipe them through.
            assert parent_scope in scope_source
            assert scope_to_handle not in scope_source
            parent_source = scope_source[parent_scope]
            assert parent_source.node is not None and scope_to_handle is not None
            new_conn_name = scope_to_handle.next_connector(producer_spec.data_name)
            state.add_edge(
                parent_source.node,
                parent_source.conn,
                scope_to_handle,
                "IN_" + new_conn_name,
                dace.Memlet(
                    data=producer_spec.data_name, subset=copy.deepcopy(producer_spec.full_shape)
                ),
            )
            scope_to_handle.add_scope_connectors(new_conn_name)
            data_source = _DataSource(scope_to_handle, f"OUT_{new_conn_name}")

        elif producer_spec.data_name in already_mapped_data:
            data_source = scope_sources[already_mapped_data[producer_spec.data_name]][
                scope_to_handle
            ]
        else:
            # Try to find an edge that maps the producer into the Map.
            parent_source = scope_source[parent_scope]
            producer_node = parent_source.node
            assert isinstance(producer_node, dace_nodes.AccessNode)
            edge_that_goes_into_scope = next(
                (
                    oedge
                    for oedge in state.out_edges(producer_node)
                    if (
                        (not oedge.data.is_empty())
                        and oedge.dst is scope_to_handle
                        and oedge.dst_conn.startswith("IN_")
                        and oedge.data.get_src_subset(oedge, state).covers(producer_spec.full_shape)
                    )
                ),
                None,
            )
            if edge_that_goes_into_scope is not None:
                scope_connector = "OUT_" + edge_that_goes_into_scope.dst_conn[3:]
            else:
                new_conn_name = scope_to_handle.next_connector(producer_spec.data_name)  # type: ignore[union-attr]
                state.add_edge(
                    producer_node,
                    None,
                    scope_to_handle,
                    "IN_" + new_conn_name,
                    dace.Memlet(
                        data=producer_spec.data_name, subset=copy.deepcopy(producer_spec.full_shape)
                    ),
                )
                scope_to_handle.add_scope_connectors(new_conn_name)  # type: ignore[union-attr]
                scope_connector = "OUT_" + new_conn_name
            already_mapped_data[producer_spec.data_name] = i
            data_source = _DataSource(node=scope_to_handle, conn=scope_connector)
        scope_source[scope_to_handle] = data_source

    return


def _setup_producer_description_for_nested_state(
    nested_state: dace.SDFGState,
    nested_sdfg: dace.SDFG,
    top_level_initial_producer_specs: list[_ProducerSpec],
    rename_spec: dict[str, str],
    resuse_exiting_access_nodes: bool,
) -> tuple[list[_ProducerSpec], list[_DataSource]]:
    """Analogue operation to `_setup_initial_producer_description_for()` but on nested levels."""
    nested_initial_producer_specs: list[_ProducerSpec] = []
    nested_base_source_scopes: list[_DataSource] = []
    created_access_nodes: dict[str, dace_nodes.AccessNode] = (
        {}
        if not resuse_exiting_access_nodes
        else {
            s.data: s for s in nested_state.source_nodes() if isinstance(s, dace_nodes.AccessNode)
        }
    )

    for tl_prod_spec in top_level_initial_producer_specs:
        nested_data_name = rename_spec.get(tl_prod_spec.data_name, tl_prod_spec.data_name)

        # Ensure that the data is registered in the sdfg.
        if nested_data_name not in nested_sdfg.arrays:
            nested_data_name, _ = nested_sdfg.add_array(
                name=nested_data_name,
                shape=tuple(s for s, _, _ in tl_prod_spec.full_shape),
                dtype=tl_prod_spec.dtype,
                transient=False,
                find_new_name=True,
            )
        assert not nested_sdfg.arrays[nested_data_name].transient

        # Created the access node.
        if nested_data_name not in created_access_nodes:
            created_access_nodes[nested_data_name] = nested_state.add_access(nested_data_name)
        access_node = created_access_nodes[nested_data_name]

        nested_initial_producer_specs.append(
            _ProducerSpec(
                data_name=nested_data_name,
                full_shape=copy.copy(tl_prod_spec.full_shape),
                dtype=tl_prod_spec.dtype,
                offset=copy.copy(tl_prod_spec.offset),
                subset=copy.copy(tl_prod_spec.subset),
                data_source={},
            )
        )
        nested_base_source_scopes.append(_DataSource(access_node, None))

    return nested_initial_producer_specs, nested_base_source_scopes


def _replace_single_read(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
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
        state: The state on which we operate.
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
    consume_subset = consumer_spec.consumed_subset(state)
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

    for tlet_input, producer_spec in zip(tlet_inputs, producer_specs):
        state.add_edge(
            producer_spec.data_source[consumer_spec].node,
            producer_spec.data_source[consumer_spec].conn,
            concat_where_tasklet,
            tlet_input,
            dace.Memlet(
                data=producer_spec.data_name,
                subset=copy.deepcopy(producer_spec.full_shape),
            ),
        )

    # Instead of replacing `final_consumer` we create an intermediate output node.
    #  This removes the need to reconfigure the dataflow.
    intermediate_data, _ = sdfg.add_scalar(
        name=f"__gt4py_concat_where_mapper_temp_{tag}",
        dtype=producer_specs[0].dtype,
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
        isinstance(consumer_spec.consumer, dace_nodes.AccessNode)
        and consumer_spec.consumer.data == consumer_spec.edge.data.data
    ):
        other_subset = consumer_spec.edge.data.subset
    else:
        other_subset = consumer_spec.edge.data.other_subset

    # Create the edge between the new intermediate and the old consumer. We ignore
    #  some properties of the Memlet here, such as dynamic (`wcr` was handled above).
    state.add_edge(
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
    dace_sdutils.remove_edge_and_dangling_path(state, consumer_spec.edge)


def _find_consumer_specs_single_level(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
) -> Optional[tuple[list[_FinalConsumerSpec], list[_FinalConsumerSpec]]]:
    """Find all consumers of `concat_node` in this state and on this level.

    The consumers are partitioned into two groups. The first group are the "genuine
    consumers", this are the consumers that only read one element from `concat_node`.
    These are the consumers that are handled by `_replace_single_read()`.
    The second group are "descending points", these are essentially nested SDFGs,
    that consume the entire concat where node (a nested SDFG that just consumes
    a single element is still classified as a "genuine consumer"). These are the
    nodes that needs further processing.
    `None` is returned in cases consumers are found that can not be handled.

    Args:
        state: The state in which we operate.
        concat_node: The node representing the `concat_where` result.

    Note:
        - This function only process a single SDFG level, i.e. does not decent into
            nested SDFGs.
        - It only operates in state `state` starting from `concat_node`, i.e. if
            there are other nodes that refers to the same data as `concat_node`
            then they are ignored, even in the same state.
        - The "genuine consumers" are in a sorted order, but the "descending
            points" are in an unspecific order.
    """

    if state.out_degree(concat_node) == 0:
        return [], []

    # First collect all consumer, we will wet and partition them afterwards.
    all_consumers: list[_FinalConsumerSpec] = []
    for oedge in state.out_edges(concat_node):
        consumer = oedge.dst
        if oedge.data.is_empty():
            # Possible, but hard to handle.
            return None
        if isinstance(consumer, dace_nodes.MapEntry):
            if not oedge.dst_conn.startswith("IN_"):
                return None  # We do not handle dynamic map ranges.
            for edge_inside_map in state.out_edges_by_connector(
                consumer, "OUT_" + oedge.dst_conn[3:]
            ):
                all_consumers.extend(
                    (
                        _FinalConsumerSpec(edge=edge)
                        for edge in state.memlet_tree(edge_inside_map).leaves()
                    )
                )
        elif isinstance(consumer, (dace_nodes.AccessNode, dace_nodes.Tasklet)):
            all_consumers.append(_FinalConsumerSpec(edge=oedge))
        else:
            return None  # These kind of consumers can not be handled.

    # Now wet and partition the consumer. Most importantly test if they only read
    #  a single element, in which case they are genuine consumers this applies to
    #  _all_ nodes, even nested SDFGs. Otherwise it might open up a "new level".
    #  For this the consumer must be a nested SDFG that reads the _whole_ concat
    #  where node.
    concat_where_shape = dace_sbs.Range.from_array(concat_node.desc(state.sdfg))
    stairs_to_deeper_levels: list[_FinalConsumerSpec] = []  # Better name appreciated.
    consumer_specs: list[_FinalConsumerSpec] = []
    for consumer_spec in all_consumers:
        consumed_subset = consumer_spec.consumed_subset(state)

        if consumed_subset.num_elements() == 1:  # Real consumer
            consumer_specs.append(consumer_spec)
        elif isinstance(consumer_spec.consumer, dace_nodes.NestedSDFG) and consumed_subset.covers(
            concat_where_shape
        ):
            stairs_to_deeper_levels.append(consumer_spec)  # Nested SDFG that opens a "new level".
        else:
            return None  # We can not handle this consumer.

    return sorted(consumer_specs), stairs_to_deeper_levels

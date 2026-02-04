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
        - Symbol mapping.
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

    producer_specs, scope_sources = _setup_initial_producer_description(
        sdfg=sdfg,
        state=state,
        converging_edges=converging_edges,
    )

    _map_data_into_nested_scopes(
        state=state,
        producer_specs=producer_specs,  # Will be modified
        scope_sources=scope_sources,  # Will be modified
        consumer_specs=consumer_specs,
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
class _ScopeLocation:
    """Denotes a scope.

    Usually in DaCe a scope is either `None`, for the global scope or a Map node.

    Essentially a pair consisting of the actual scope node or `None` if on global
    scope and the SDFG state where the node is located. The reason for not just
    using the node is, because we might have multiple global scopes, which are
    `None`, that we have to distinguish.
    """

    scope_node: Union[dace_nodes.AccessNode, dace_nodes.MapEntry, None]
    state: dace.SDFGState

    @property
    def sdfg(self) -> dace.SDFG:
        return self.state.sdfg

    def __post_init__(self) -> None:
        assert self.scope_node is None or isinstance(
            self.scope_node, (dace_nodes.AccessNode, dace_nodes.MapEntry)
        )

    def __hash__(self) -> int:
        return hash((self.scope_node, self.state))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ScopeLocation):
            return self.scope_node == other.scope_node and self.state == other.state
        elif isinstance(other, tuple):
            assert len(other) == 2
            return self == _ScopeLocation(other[0], other[1])
        return NotImplemented


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

    def __hash__(self) -> int:
        return hash(self.edge)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dace_graph.MultiConnectorEdge):
            return self.edge == other
        elif isinstance(other, _FinalConsumerSpec):
            return self.edge == other.edge
        return NotImplemented


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
        data_source: Maps each consumer to the data source location.

    Note:
        The format of `data_source` seems a bit redundant, however, it ensures that
        the information is there even if the SDFG is invalid, and also allows
        to operate in nested SDFG scenario.
    """

    data_name: str
    full_shape: dace_sbs.Range
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    data_source: dict[_FinalConsumerSpec, _DataSource]

    def __copy__(self) -> "_ProducerSpec":
        return _ProducerSpec(
            data_name=copy.copy(self.data_name),
            full_shape=copy.copy(self.full_shape),
            offset=copy.copy(self.offset),
            subset=copy.copy(self.subset),
            data_source=self.data_source.copy(),
        )


def _setup_initial_producer_description(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    converging_edges: Sequence[dace_graph.MultiConnectorEdge[dace.Memlet]],
) -> tuple[list[_ProducerSpec], list[dict[_ScopeLocation, _DataSource]]]:
    """Sets up the producer specs.

    The function essentially populates all attribute of `_ProducerSpec` with the
    exception of `.data_source` which is only partially populated.
    Note that the function expects that `converting_edges` are AccessNode to AccessNode
    connections and have the same target, the concat where node.
    Furthermore, the order in which producer specifications are generated is the same
    as in `converging_edges`, the sources do not need to be different.
    In addition it is also important to realize that multiple producer specs might
    refer to the same data.

    It also sets up the scope source, like the returned producer specs it is also a
    list with the same ordering. Each element is a `dict` that maps a scope to
    where the full data of that scope can be found.
    In the end it is a compressed version of `_ProducerSpec.data_source`, that however
    needs that `scope_dict` is valid at every point. It will later be expanded.

    Note:
        This function can only be called on a valid SDFG. Furthermore, it only works
        for a single state.
    """
    assert len(converging_edges) >= 2
    assert isinstance(converging_edges[0].dst, dace_nodes.AccessNode)
    assert state.scope_dict()[converging_edges[0].dst] is None
    concat_node = converging_edges[0].dst

    scope_sources: list[dict[_ScopeLocation, _DataSource]] = []
    producer_specs: list[_ProducerSpec] = []
    for converging_edge in converging_edges:
        assert isinstance(converging_edge.src, dace_nodes.AccessNode)
        assert converging_edge.dst is concat_node
        producer_node = converging_edge.src
        producer_specs.append(
            _ProducerSpec(
                data_name=producer_node.data,
                full_shape=dace_sbs.Range.from_array(producer_node.desc(sdfg)),
                offset=copy.deepcopy(converging_edge.data.get_src_subset(converging_edge, state)),
                subset=copy.deepcopy(converging_edge.data.dst_subset),
                data_source={},
            )
        )
        scope_sources.append({_ScopeLocation(None, state): _DataSource(producer_node, None)})

    return producer_specs, scope_sources


def _map_data_into_nested_scopes(
    state: dace.SDFGState,
    scope_sources: list[dict[_ScopeLocation, _DataSource]],
    producer_specs: list[_ProducerSpec],
    consumer_specs: list[_FinalConsumerSpec],
) -> None:
    """Ensures that the producer data is available in every nested scope.

    It is important that this function only processes the scope defined by the
    consumer, `consumer_specs`. Which are further required to only reside inside
    `state` (which includes nested SDFGs).

    This function essentially finalizes, `producer_specs` which was obtained from
    a previous call from `_setup_initial_producer_description()` as was `scope_sources`
    which is modified, in the sense of a scratch pad.
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

    # Process the nested scopes.
    for needed_scope in accessed_scopes:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_to_handle=_ScopeLocation(needed_scope, state),
            producer_specs=producer_specs,
            scope_sources=scope_sources,
        )

    # Now fill in the data sources, i.e. expand `scope_source` into `_ProducerSpec.data_source`.
    for consumer_spec in consumer_specs:
        consumer_scope = scope_dict[consumer_spec.consumer]
        for scope_source, producer_spec in zip(scope_sources, producer_specs):
            producer_spec.data_source[consumer_spec] = scope_source[
                _ScopeLocation(consumer_scope, consumer_spec.state)
            ]

    return None


def _map_data_into_nested_scopes_impl(
    state: dace.SDFGState,
    scope_to_handle: _ScopeLocation,
    producer_specs: list[_ProducerSpec],
    scope_sources: list[dict[_ScopeLocation, _DataSource]],
) -> None:
    """Helper function of `_map_data_into_nested_scopes_impl()` that populate nested scopes."""
    if scope_to_handle in scope_sources[0]:
        return

    # Check if the parent scope was handled before, if not handle it.
    parent_scope = _ScopeLocation(
        scope_to_handle.state.scope_dict()[scope_to_handle.scope_node], scope_to_handle.state
    )
    if parent_scope not in scope_sources[0]:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_to_handle=parent_scope,
            producer_specs=producer_specs,
            scope_sources=scope_sources,
        )
    assert parent_scope in scope_sources[0]

    # On the top level perform dedublication of inputs.
    already_mapped_data: dict[str, int] = {}
    for i, (producer_spec, scope_source) in enumerate(zip(producer_specs, scope_sources)):
        if parent_scope.scope_node is not None:
            # Nested scopes, just pipe them through.
            assert parent_scope in scope_source
            assert scope_to_handle not in scope_source
            parent_source = scope_source[parent_scope]
            assert parent_source.node is not None and scope_to_handle.scope_node is not None
            new_conn_name = scope_to_handle.scope_node.next_connector(producer_spec.data_name)
            scope_to_handle.state.add_edge(
                parent_source.node,
                parent_source.conn,
                scope_to_handle.scope_node,
                "IN_" + new_conn_name,
                dace.Memlet(
                    data=producer_spec.data_name, subset=copy.deepcopy(producer_spec.full_shape)
                ),
            )
            scope_to_handle.scope_node.add_scope_connectors(new_conn_name)
            data_source = _DataSource(scope_to_handle.scope_node, f"OUT_{new_conn_name}")

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
                        and oedge.dst is scope_to_handle.scope_node
                        and oedge.dst_conn.startswith("IN_")
                        and oedge.data.get_src_subset(oedge, state).covers(producer_spec.full_shape)
                    )
                ),
                None,
            )
            if edge_that_goes_into_scope is not None:
                scope_connector = "OUT_" + edge_that_goes_into_scope.dst_conn[3:]
            else:
                new_conn_name = scope_to_handle.scope_node.next_connector(producer_spec.data_name)  # type: ignore[union-attr]
                state.add_edge(
                    producer_node,
                    None,
                    scope_to_handle.scope_node,
                    "IN_" + new_conn_name,
                    dace.Memlet(
                        data=producer_spec.data_name, subset=copy.deepcopy(producer_spec.full_shape)
                    ),
                )
                scope_to_handle.scope_node.add_scope_connectors(new_conn_name)  # type: ignore[union-attr]
                scope_connector = "OUT_" + new_conn_name
            already_mapped_data[producer_spec.data_name] = i
            data_source = _DataSource(node=scope_to_handle.scope_node, conn=scope_connector)
        scope_source[scope_to_handle] = data_source

    return


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

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
from typing import Any, Literal, Optional, Sequence, TypeAlias, Union, overload

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, utils as dace_sdutils

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


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
    genuine_consumer_specs, descending_points = _find_consumer_specs_single_source_single_level(
        state, concat_node, for_check=False
    )
    assert all(_check_descending_point(descening_point) for descening_point in descending_points)

    scope_dict = state.scope_dict()
    initial_producer_specs, base_scope_sources = _setup_initial_producer_description_on_top_level(
        sdfg=sdfg,
        state=state,
        concat_node=concat_node,
        scope_dict=scope_dict,
    )

    # Not only the genuine consumers but also the "descending points" needs access to the data.
    producer_specs = _map_data_into_nested_scopes(
        state=state,
        scope_dict=scope_dict,
        initial_producer_specs=initial_producer_specs,
        base_scope_sources=base_scope_sources,
        consumer_specs=genuine_consumer_specs + descending_points,
    )

    # Now process the genuine consumers.
    for consumer_spec in genuine_consumer_specs:
        _replace_single_read(
            state=state,
            sdfg=sdfg,
            consumer_spec=consumer_spec,
            producer_specs=producer_specs,
            tag=f"{concat_node.data}_{'' if tag is None else tag}",
        )

    _process_descending_points_of_state(
        state=state,
        descending_points=descending_points,
        initial_producer_specs=initial_producer_specs,
        producer_specs=producer_specs,
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
            return (self.consumer, str(self.edge)) < (self.consumer, str(other.edge))
        return NotImplemented


_ConsumerPartition: TypeAlias = tuple[list[_FinalConsumerSpec], list[_FinalConsumerSpec]]
"""Represents a partition of consumer specifications.

It is a pair of `list`s. The first `list` contains the "genuine consumers", this are
the consumers that only read one element from the concat where node. Essentially,
these consumers can be handled by `_replace_single_read()`.
The second `list` contains the "descending points", these are essentially nested SDFGs,
that consume the entire data from the concat where node (a nested SDFG that just
consumes a single element is classified as a "genuine consumer"). These are the nodes
that needs further processing.
"""


@dataclasses.dataclass(frozen=True)
class _ProducerSpec:
    """Describes how a `concat_where` converges at a name.

    This class only describes a single source of the concat where. To make sense
    one needs a `list` of `_ProducerSpec`, one for each converging edges.

    Args:
        data_name: Name of the data descriptor.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        desc: The data descriptor.
        free_symb_types: Contains the type of every free symbol of the data descriptor.
        data_source: Maps each consumer to the data source location.

    Note:
        The format of `data_source` seems a bit redundant, however, it ensures that
        the information is there even if the SDFG is invalid, and also allows
        to operate in nested SDFG scenario.
    """

    data_name: str
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    desc: dace_data.Data
    free_symb_types: dict[str, dace.dtypes.typeclass]  # Find out if redundant.
    data_source: dict[_FinalConsumerSpec, _DataSource]

    def __copy__(self) -> "_ProducerSpec":
        return _ProducerSpec(
            data_name=self.data_name,
            offset=copy.copy(self.offset),
            subset=copy.copy(self.subset),
            desc=self.desc.clone(),
            free_symb_types=self.free_symb_types.copy(),
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


def _setup_initial_producer_description_on_top_level(
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

    base_scope_sources_unordered: dict[_ProducerSpec, _DataSource] = {}
    initial_producer_specs: list[_ProducerSpec] = []
    sdfg_symbols = sdfg.symbols
    for converging_edge in converging_edges:
        assert isinstance(converging_edge.src, dace_nodes.AccessNode)
        assert converging_edge.dst is concat_node
        converging_edge.data.try_initialize(sdfg, state, converging_edge)
        producer_node = converging_edge.src
        producer_desc = producer_node.desc(sdfg)
        free_symb_types = {str(fs): sdfg_symbols[str(fs)] for fs in producer_desc.free_symbols}
        offset = copy.deepcopy(converging_edge.data.src_subset)
        subset = copy.deepcopy(converging_edge.data.dst_subset)
        assert offset is not None and subset is not None

        initial_producer_specs.append(
            _ProducerSpec(
                data_name=producer_node.data,
                offset=offset,
                subset=subset,
                desc=producer_desc.clone(),
                free_symb_types=free_symb_types,
                data_source={},
            )
        )
        base_scope_sources_unordered[initial_producer_specs[-1]] = _DataSource(producer_node, None)

    # Now sort the producer specification and then apply the same order to the scope sources.
    initial_producer_specs = sorted(initial_producer_specs)
    base_scope_sources = [
        base_scope_sources_unordered[prod_spec] for prod_spec in initial_producer_specs
    ]

    return initial_producer_specs, base_scope_sources


def _process_descending_points_of_state(
    state: dace.SDFGState,
    descending_points: Sequence[_FinalConsumerSpec],
    initial_producer_specs: Sequence[_ProducerSpec],
    producer_specs: Sequence[_ProducerSpec],
) -> None:
    """Do some preprocessing."""
    configured_descending_points: dict[dace_nodes.NestedSDFG, list[_ProducerSpec]] = {}
    for descending_point in descending_points:
        if descending_point.consumer not in configured_descending_points:
            configured_descending_points[descending_point.consumer] = _configure_descending_point(
                state=state,
                descending_point=descending_point,
                parent_initial_producer_specs=initial_producer_specs,
                full_parent_producer_specs=producer_specs,
            )

        _process_descending_point_impl(
            this_descending_point=descending_point,
            transformed_initial_producer_specs=configured_descending_points[
                descending_point.consumer
            ],
        )
        dace_sdutils.remove_edge_and_dangling_path(state, descending_point.edge)


def _process_descending_point_impl(
    this_descending_point: _FinalConsumerSpec,
    transformed_initial_producer_specs: list[_ProducerSpec],  # UPPER LEVEL
) -> None:
    """Perform preprocessing and then call the actual implementation."""
    consumers = _find_consumer_specs_in_descending_point(this_descending_point)
    for state, (concat_node, (genuine_consumer_specs, descending_points)) in consumers.items():
        scope_dict = state.scope_dict()
        initial_producer_specs, base_scope_sources = (
            _setup_initial_producer_description_in_nested_state(
                descending_point=this_descending_point,
                nested_state=state,
                transformed_initial_producer_specs=transformed_initial_producer_specs,
            )
        )
        producer_specs = _map_data_into_nested_scopes(
            state=state,
            scope_dict=scope_dict,
            initial_producer_specs=initial_producer_specs,
            base_scope_sources=base_scope_sources,
            consumer_specs=genuine_consumer_specs + descending_points,
        )

        # Replace the consumer on this level.
        for consumer_spec in genuine_consumer_specs:
            _replace_single_read(
                state=state,
                sdfg=state.sdfg,
                consumer_spec=consumer_spec,
                producer_specs=producer_specs,
                tag=f"{this_descending_point.edge.dst_conn}",
            )

        # Now recursively descend.
        _process_descending_points_of_state(
            state=state,
            descending_points=descending_points,
            initial_producer_specs=initial_producer_specs,
            producer_specs=producer_specs,
        )

        assert state.degree(concat_node) == 0
        state.remove_node(concat_node)


def _setup_initial_producer_description_in_nested_state(
    descending_point: _FinalConsumerSpec,
    nested_state: dace.SDFGState,
    transformed_initial_producer_specs: list[_ProducerSpec],
) -> tuple[list[_ProducerSpec], list[_DataSource]]:
    """Analogue operation to `_setup_initial_producer_description_on_top_level()` but on nested levels.

    The function expects that `initial_producer_specs` was generated by a previous
    call to `_configure_descending_point()`, that got the same `descending_point`.
    Furthermore, the function expects that `nested_state` is a state inside the
    descending point.

    In principle the function will do the same as `_setup_initial_producer_description_on_top_level()`,
    i.e. setting up the producer and scope source description. However, it will also
    generate create AccessNodes for the mapped data, if it can not find any data
    inside.

    Args:
        descending_point: The descending point in which we operate.
        nested_state: A state inside the nested SDFG.
        transformed_initial_producer_specs: The transformed initial producer
            specification, generated by `_configure_descending_point()`.
    """
    nested_sdfg: dace.SDFG = descending_point.consumer.sdfg
    assert nested_state in nested_sdfg.states()

    # Look for access nodes that already exist.
    source_access_nodes: dict[str, dace_nodes.AccessNode] = {}
    needed_data: set[str] = {
        producer_spec.data_name for producer_spec in transformed_initial_producer_specs
    }
    for dnode in nested_state.data_nodes():
        if dnode.data in needed_data:
            needed_data.remove(dnode.data)
            assert not nested_sdfg.arrays[dnode.data].transient
            assert nested_state.scope_dict()[dnode] is None
            source_access_nodes[dnode.data] = dnode
            if len(needed_data) == 0:
                break

    # Now create the ones that are missing.
    for missing_data in needed_data:
        assert not nested_sdfg.arrays[missing_data].transient
        source_access_nodes[missing_data] = nested_state.add_access(missing_data)

    # Now setup the data.
    nested_initial_producer_specs: list[_ProducerSpec] = []
    nested_base_source_scopes: list[_DataSource] = []
    for prod_spec in transformed_initial_producer_specs:
        nested_initial_producer_specs.append(
            _ProducerSpec(
                data_name=prod_spec.data_name,
                offset=copy.deepcopy(prod_spec.offset),
                subset=copy.deepcopy(prod_spec.subset),
                desc=prod_spec.desc.clone(),
                free_symb_types=prod_spec.free_symb_types.copy(),
                data_source={},
            )
        )
        nested_base_source_scopes.append(
            _DataSource(source_access_nodes[prod_spec.data_name], None)
        )

    return nested_initial_producer_specs, nested_base_source_scopes


def _configure_descending_point(
    state: dace.SDFGState,
    descending_point: _FinalConsumerSpec,
    parent_initial_producer_specs: Sequence[_ProducerSpec],
    full_parent_producer_specs: Sequence[_ProducerSpec],
) -> list[_ProducerSpec]:
    """Handles a descending point.

    The function computes the initial producer specification for the nested SDFG.
    In addition it ensures:
    - That the nested SDFG has the necessary data descriptors and symbols:
    - That the nested SDFG has an updated symbol mapping.
    - That the nested SDFG was connected to the data source.
    - Remove the edge that maps in the top level concat where data.
    - Applies the symbol remapping to the returned producer specifications.

    The function returns a transformed version of the initial producer specification
    of the parent that is can be used inside the descending point. It can then later
    be passed to `_setup_initial_producer_description_in_nested_state()` to obtain
    producer specification and scope information inside a state.

    It is important that this function can only be called once per descending point.

    Args:
        state: The state containing the descending point.
        descending_point: The consumer that represents a descending point.
        parent_initial_producer_specs: The initial producer specification on the top level.
        full_parent_producer_specs: The fully processed producer specification of the parent.
    """
    assert isinstance(descending_point.consumer, dace_nodes.NestedSDFG)
    nsdfg_node: dace_nodes.NestedSDFG = descending_point.consumer
    nsdfg: dace.SDFG = nsdfg_node.sdfg
    symbol_mapping: dict[str, dace_sym.SymExpr] = nsdfg_node.symbol_mapping

    # Name of the data at the top that is used for the concat where.
    top_concat_data = descending_point.edge.data.data

    # Now we need to map the data into the nested SDFG. For that we will look at
    #  what is already there.
    nested_to_top_desc_mapping = gtx_transformations.utils.gt_data_descriptor_mapping(
        state=state,
        nsdfg=nsdfg_node,
        only_inputs=True,
    )
    top_to_nested_desc_mapping = {top: nested for nested, top in nested_to_top_desc_mapping.items()}

    transformed_initial_producer_specs: list[_ProducerSpec] = []
    handled_data: dict[str, int] = {}
    for i, parent_prod_spec in enumerate(parent_initial_producer_specs):
        parent_data_name = parent_prod_spec.data_name

        # This data is already handled so we can simply copy it.
        if parent_data_name in handled_data:
            transformed_initial_producer_specs.append(
                copy.copy(transformed_initial_producer_specs[handled_data[parent_data_name]])
            )
            continue

        nested_offset = copy.deepcopy(parent_prod_spec.offset)
        nested_subset = copy.deepcopy(parent_prod_spec.subset)
        nested_desc = parent_prod_spec.desc.clone()
        nested_desc.transient = False  # Inputs of nested SDFGs are always global.

        # Handle the symbols
        nested_free_symb_types: dict[str, dace.dtypes.typeclass] = {}
        repl_dict: dict[str, str] = {}
        for psym, ptype in parent_prod_spec.free_symb_types.items():
            if (
                psym in symbol_mapping
                and psym == str(symbol_mapping[psym])
                and psym in nsdfg.symbols
                and nsdfg.symbols[psym] == ptype
            ):
                # The symbol is already mapped 1:1 into the nested sdfg, so we do not have to do anything.
                nested_symbol = psym
            else:
                # The symbol is either not known or it is not mapped 1:1 or has the wrong type,
                #  in either case we have to create a new symbol inside the nested SDFG.
                nested_symbol = nsdfg.add_symbol(psym, ptype, find_new_name=True)
                assert nested_symbol not in symbol_mapping
                symbol_mapping[nested_symbol] = psym
                repl_dict[psym] = nested_symbol
            nested_free_symb_types[nested_symbol] = ptype

        # If needed apply the renaming of symbols in the
        if len(repl_dict) != 0:
            nested_offset.replace(repl_dict)
            nested_subset(repl_dict)
            dace.sdfg.replace_properties_dict(nested_desc, repl=repl_dict)

        # Now make sure that `parent_data_name` is available in the nested SDFG.
        if parent_data_name in top_to_nested_desc_mapping:
            # The data is already available inside the nested SDFG.
            nested_data_name = top_to_nested_desc_mapping[parent_data_name]
        else:
            # The data is not available inside the nested SDFG or the name as a different meaning.
            nested_data_name, _ = nsdfg.add_datadesc(
                name=nested_data_name,
                datadesc=nested_desc.clone(),
                find_new_name=True,
            )

        # Now create a connection to fully map the data into the nested SDFG.
        state.add_edge(
            full_parent_producer_specs[i].data_source[descending_point].node,
            full_parent_producer_specs[i].data_source[descending_point].conn,
            descending_point.consumer,
            nested_data_name,
            dace.Memlet(
                data=full_parent_producer_specs[i].data_name,
                subset=dace_sbs.Range.from_array(full_parent_producer_specs[i].desc),
            ),
        )
        assert nested_data_name not in descending_point.consumer.in_connectors
        descending_point.consumer.add_in_connector(nested_data_name, dtype=nested_desc.dtype)

        transformed_initial_producer_specs.append(
            _ProducerSpec(
                data_name=nested_data_name,
                offset=nested_offset,
                subset=nested_subset,
                desc=nested_desc.clone(),
                free_symb_types=nested_free_symb_types,
                data_source={},
            )
        )
        handled_data[parent_data_name] = len(transformed_initial_producer_specs)

    # Restore the proper format of the symbol mapping.
    nsdfg_node.symbol_mapping = symbol_mapping
    nested_concat_data = top_to_nested_desc_mapping[top_concat_data]

    # Remove the consumer
    dace_sdutils.remove_edge_and_dangling_path(state, descending_point.edge)
    assert nested_concat_data not in descending_point.consumer.out_connectors

    return transformed_initial_producer_specs


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
    `_setup_initial_producer_description_on_top_level()` or
    `_setup_base_scope_sources_in_state()`) to compute a full producer description
    which is also returned. In addition it will make sure that the data is available
    in the nested scopes, such that the replacement, performed by `_replace_single_read()`
    can be performed.
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
                    data=producer_spec.data_name,
                    subset=dace_sbs.Range.from_array(producer_spec.desc),
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
            full_shape = dace_sbs.Range.from_array(producer_spec.desc)
            assert isinstance(producer_node, dace_nodes.AccessNode)
            edge_that_goes_into_scope = next(
                (
                    oedge
                    for oedge in state.out_edges(producer_node)
                    if (
                        (not oedge.data.is_empty())
                        and oedge.dst is scope_to_handle
                        and oedge.dst_conn.startswith("IN_")
                        and oedge.data.get_src_subset(oedge, state).covers(full_shape)
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
                        data=producer_spec.data_name,
                        subset=full_shape,  # No need to copy.
                    ),
                )
                scope_to_handle.add_scope_connectors(new_conn_name)  # type: ignore[union-attr]
                scope_connector = "OUT_" + new_conn_name
            already_mapped_data[producer_spec.data_name] = i
            data_source = _DataSource(node=scope_to_handle, conn=scope_connector)
        scope_source[scope_to_handle] = data_source

    return


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
                subset=dace_sbs.Range.from_array(producer_spec.desc),
            ),
        )

    # Instead of replacing `final_consumer` we create an intermediate output node.
    #  This removes the need to reconfigure the dataflow.
    intermediate_data, _ = sdfg.add_scalar(
        name=f"__gt4py_concat_where_mapper_temp_{tag}",
        dtype=sdfg.arrays[consumer_spec.edge.data.data].dtype,
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


def _find_consumer_specs_in_descending_point(
    descening_point: _FinalConsumerSpec,
) -> dict[dace.SDFGState, tuple[dace_nodes.AccessNode, _ConsumerPartition]]:
    """Scans the descending point for consumer (non recursive).

    The function will scan every state for occurrences of the concat where data
    and return the consumer partition for each state. If there are multiple
    AccessNode referring to the concat where data then the nodes are unified,
    i.e. this function mutates the state.

    Furthermore, this function assumes that the partition exists, i.e.
    `_check_descending_point()` has returned `True`.
    """
    sdfg: dace.SDFG = descening_point.consumer.sdfg
    concat_data = descening_point.edge.dst_conn

    consumers: dict[dace.SDFG, tuple[dace_nodes.AccessNode, _ConsumerPartition]] = {}
    for state in sdfg.states():
        concat_node: Union[None, dace_nodes.AccessNode] = None
        for dnode in state.data_nodes():
            if dnode.data != concat_data:
                continue
            assert state.in_edges(dnode) == 0

            if concat_node is not None:
                # We found a second node referring to the concat where data,
                #  we have to merge them.
                for oedge in state.out_edges(dnode):
                    dace_transformation.helpers.redirect_edge(
                        state=state,
                        edge=oedge,
                        new_src=concat_node,
                    )
                assert state.degree(dnode) == 0
                state.remove_node(dnode)

        if concat_node is None:
            continue

        consumer_partition = _find_consumer_specs_single_source_single_level(
            state, concat_node, for_check=False
        )
        consumers[state] = (concat_node, consumer_partition)
    return consumers


def _check_descending_point(
    descening_point: _FinalConsumerSpec,
) -> bool:
    """Check if this descending point has valid consumers, consumer are checked recursively."""
    sdfg: dace.SDFG = descening_point.consumer.sdfg
    concat_data = descening_point.edge.dst_conn
    assert not sdfg.arrays[concat_data].transient

    for state in sdfg.states():
        for dnode in state.data_nodes():
            if dnode.data != concat_data:
                continue
            if state.in_edges(dnode) != 0:
                return False
            scan_for_nested_descending_points = _find_consumer_specs_single_source_single_level(
                state,
                dnode,
                for_check=True,
            )
            if scan_for_nested_descending_points is None:
                return False
            for nested_descending_point in scan_for_nested_descending_points:
                if not _check_descending_point(nested_descending_point):
                    return False
    return True


@overload
def _find_consumer_specs_single_source_single_level(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    for_check: Literal[False],
) -> _ConsumerPartition: ...


@overload
def _find_consumer_specs_single_source_single_level(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    for_check: Literal[True],
) -> Optional[list[_FinalConsumerSpec]]: ...


def _find_consumer_specs_single_source_single_level(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    for_check: bool,
) -> Union[Optional[list[_FinalConsumerSpec]], _ConsumerPartition]:
    """Find all consumers of `concat_node` in state `state`.

    Of `for_check` is `False` then the function returns the consumer partition for
    `concat_node`. Note if the consumer partition does not exists then an error
    is returned in this mode.
    If `for_check` is `True` then the function only checks if the partition exists
    to indicate this it will only return the descending points or `None` if the
    partition does not exist.

    Important the function does not:
    - Considers any other state.
    - Considers any other AccessNode in `state` that also refers to the same
        data as `concat_node` does.
    - Does not descend into nested SDFGs.

    Args:
        state: The state in which we operate.
        concat_node: The node representing the `concat_where` result.
        for_check: Only perform checks.

    Note:
        If `for_check` is `False` then both kind of consumers were sorted. For `True`
        the order of the returned descending points is unspecific.
    """

    if state.out_degree(concat_node) == 0:
        return ([], []) if not for_check else []

    # First collect all consumer, we will check and partition them afterwards.
    all_consumers: list[_FinalConsumerSpec] = []
    edges_to_scan_next: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = list(
        state.out_edges(concat_node)
    )
    partition_does_not_exists = False
    while len(edges_to_scan_next) != 0:
        edges_to_scan, edges_to_scan_next = edges_to_scan_next, []
        for oedge in edges_to_scan:
            consumer = oedge.dst
            if oedge.data.is_empty():
                partition_does_not_exists = True  # We should probably handle it.
                break
            if isinstance(consumer, dace_nodes.MapEntry):
                if not oedge.dst_conn.startswith("IN_"):
                    partition_does_not_exists = True  # Dynamic map ranges are not supported.
                    break
                for edge_inside_map in state.out_edges_by_connector(
                    consumer, "OUT_" + oedge.dst_conn[3:]
                ):
                    edges_to_scan_next.extend(
                        edge for edge in state.memlet_tree(edge_inside_map).leaves()
                    )
            elif isinstance(
                consumer, (dace_nodes.AccessNode, dace_nodes.Tasklet, dace_nodes.NestedSDFG)
            ):
                all_consumers.append(_FinalConsumerSpec(edge=oedge))
            else:
                partition_does_not_exists = True  # These kind of consumers can not be handled.
                break
        if partition_does_not_exists:
            if for_check:
                return None
            raise ValueError(f"Expected that partition for {concat_node} exists but it does not.")

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
            if not for_check:
                consumer_specs.append(consumer_spec)
        elif isinstance(consumer_spec.consumer, dace_nodes.NestedSDFG) and consumed_subset.covers(
            concat_where_shape
        ):
            stairs_to_deeper_levels.append(consumer_spec)  # Nested SDFG that opens a "new level".
        else:
            if for_check:
                return None
            raise ValueError(f"Expected that partition for {concat_node} exists but it does not.")

    if for_check:
        return stairs_to_deeper_levels
    return sorted(consumer_specs), sorted(stairs_to_deeper_levels)

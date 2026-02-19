# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import functools
import warnings
from typing import Any, Collection, Literal, Mapping, Optional, Sequence, TypeAlias, Union, overload

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next import config as gtx_config
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class ConcatWhereCopyToMap(dace_transformation.SingleStateTransformation):
    """Replaces `concat_where` nodes with Tasklets that access the source data directly.

    This is essentially a wrapper around the `gt_replace_concat_where_node()`, for
    more information see there.
    There is also `gt_apply_concat_where_replacement_on_sdfg()`, it is recommended
    to use, as it guarantees stable ordering.

    Args:
        single_use_data: Single use data, if not provided a scan will be performed.
        tag: Used to mangle names of data descriptors, see `gt_replace_concat_where_node()`
            for more.
    """

    node_a1 = dace_transformation.PatternNode(dace_nodes.AccessNode)  # Needed to speed up matching.
    concat_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    tag = dace_properties.Property(dtype=str, default=None, allow_none=True)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[Mapping[dace.SDFG, set[str]]]

    def __init__(
        self,
        tag: Optional[str] = None,
        single_use_data: Optional[Mapping[dace.SDFG, set[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._single_use_data = single_use_data
        self.tag = tag

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

        # If we have the single use data check, perform it here to safe some scanning.
        if (
            self._single_use_data is not None
            and concat_node.data not in self._single_use_data[sdfg]
        ):
            return False

        # We can not replace global data.
        if not sdfg.arrays[concat_node.data].transient:
            return False

        # Must be on the top scope.
        if graph.scope_dict()[map_entry] is not None:
            return False

        # Check if the accesses are valid.
        if not gt_check_if_concat_where_node_is_replaceable(
            state=graph,
            concat_node=concat_node,
        ):
            return False

        # Check if single use data if not yet done.
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
            if concat_node.data not in single_use_data[sdfg]:
                return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        concat_node: dace_nodes.AccessNode = self.concat_node
        gt_replace_concat_where_node(
            sdfg=sdfg,
            state=graph,
            concat_node=concat_node,
            tag=self.tag,
        )


def gt_replace_concat_where_node(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    tag: Optional[str] = None,
) -> int:
    """Replaces all accesses to `concat_node`, result of a `concat_where`, with Tasklets.

    Instead of accessing the AccessNode, in which the result of the `concat_where`
    is written, directly, the function replaces accesses with Tasklets, that based
    on original access select what data from the producers are accessed.
    The function is also able to handle nested SDFGs, i.e. the `concat_where` data
    is accessed inside nested SDFGs.

    The function has currently the following limitations:
    - `concat_node` must be single use data.
    - All producers of `concat_node` must be AccessNodes (might be lifted).
    - It is only possible to replace accesses of a single element that happens on
        a Memlet, i.e. no neighborhood accesses.

    The function returns the number of edges that were replaced.

    In order to check if a `concat_where` AccessNode can be replaced use
    `gt_check_if_concat_where_node_is_replaceable()` (it is a pre condition of this
    function that `gt_check_if_concat_where_node_is_replaceable()` returned `True`).
    To replace all `concat_where` nodes in an entire SDFG
    `gt_apply_concat_where_replacement_on_sdfg()` is provided.

    Args:
        sdfg: The SDFG in which we operate.
        state: The state that contains the `concat_where` node.
        concat_node: The node that represents the result of the `concat_where` expression.
        tag: Use this to mangle the name of data descriptors that should be created.
            If not given defaults to `concat_node.data`.
    """
    assert sdfg.arrays[concat_node.data].transient

    # These are all the consumers that we need to modify.
    genuine_consumer_specs, descending_points = _find_consumer_specs_single_source_single_level(
        state, concat_node, for_check=False
    )

    scope_dict = state.scope_dict()
    initial_producer_specs = _setup_initial_producer_description_on_top_level(
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
        consumer_specs=genuine_consumer_specs + descending_points,
    )

    # Now process the genuine consumers.
    nb_applies = 0
    for consumer_spec in genuine_consumer_specs:
        _replace_single_read(
            sdfg=sdfg,
            state=state,
            scope_dict=scope_dict,
            consumer_spec=consumer_spec,
            producer_specs=producer_specs,
            tag=f"{concat_node.data}_{'' if tag is None else tag}",
        )
        nb_applies += 1

    nb_applies += _process_descending_points_of_state(
        state=state,
        scope_dict=scope_dict,
        descending_points=descending_points,
        producer_specs=producer_specs,
    )

    assert state.out_degree(concat_node) == 0
    state.remove_node(concat_node)
    sdfg.remove_data(concat_node.data, validate=False)

    return nb_applies


def gt_check_if_concat_where_node_is_replaceable(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
) -> bool:
    """Check if `concat_node` is a suitable candidate for replacement by `gt_replace_concat_where_node()`.

    The function checks if the producer are valid, i.e. if they are AccessNodes and
    if the consumer of `concat_where` data can be replaced. It does not check if
    `concat_node` is single use data.

    Args:
        state: The state that contains the `concat_where` node.
        concat_node: The node that represents the result of the `concat_where` expression.
    """
    # Check the producers
    if state.in_degree(concat_node) < 2:
        # Otherwise it would be a redundant copy.
        return False
    for iedge in state.in_edges(concat_node):
        if not isinstance(iedge.src, dace_nodes.AccessNode):
            return False

    descending_points = _find_consumer_specs_single_source_single_level(
        state, concat_node, for_check=True
    )
    if descending_points is None:
        return False

    for descending_point in descending_points:
        if not _check_descending_point(descending_point):
            return False

    return True


def gt_apply_concat_where_replacement_on_sdfg(
    sdfg: dace.SDFG,
    single_use_data: Optional[Mapping[dace.SDFG, Collection[str]]] = None,
    validate: bool = False,
    validate_all: bool = False,
) -> int:
    """Applies `gt_apply_concat_where_replacement_on_sdfg()` on the entire SDFG.

    The function scans the SDFG and calls `gt_apply_concat_where_replacement_on_sdfg()`
    for all suitable nodes, including nested SDFGs. The function guarantees a stable
    ordering, that is based on the labels.
    The function returns the number of replacements.

    Args:
        sdfg: The sdfg to process.
        single_use_data: The result of the `FindSingleUseData` analysis pass. If `None`,
            the default, the function will perform the scan automatically.
        validate: Perform validation at the end.
        validate_all: Perform validation also at intermediate steps.
    """

    if single_use_data is None or sdfg not in single_use_data:
        find_single_use_data = dace_analysis.FindSingleUseData()
        single_use_data = find_single_use_data.apply_pass(sdfg, None)

    found_nsdfgs: list[tuple[dace.SDFGState, dace_nodes.NestedSDFG]] = []
    suitable_concat_nodes: list[tuple[dace.SDFGState, dace_nodes.AccessNode]] = []
    nb_applies = 0
    for state in sdfg.states():
        scope_dict = state.scope_dict()
        for node in state.nodes():
            if (
                isinstance(node, dace_nodes.AccessNode)
                and scope_dict[node] is None
                and sdfg.arrays[node.data].transient
            ):
                if node.data in single_use_data[
                    sdfg
                ] and gt_check_if_concat_where_node_is_replaceable(state, node):
                    suitable_concat_nodes.append((state, node))
            elif isinstance(node, dace_nodes.NestedSDFG):
                found_nsdfgs.append((state, node))

    if len(suitable_concat_nodes) > 0:
        suitable_concat_nodes = sorted(suitable_concat_nodes, key=lambda x: (repr(x[0]), x[1].data))
        for state, concat_node in suitable_concat_nodes:
            nb_applies += gt_replace_concat_where_node(
                sdfg=sdfg,
                state=state,
                concat_node=concat_node,
            )
            if validate_all:
                # TODO(phimuell): Limit validation to the state.
                sdfg.validate()

    if len(found_nsdfgs) > 0:
        found_nsdfgs = sorted(found_nsdfgs, key=lambda x: (repr(x[0]), str(x[1])))
        for _, nsdfg in found_nsdfgs:
            nb_applies += gt_apply_concat_where_replacement_on_sdfg(
                sdfg=nsdfg.sdfg,
                single_use_data=single_use_data,
                validate=False,
                validate_all=validate_all,
            )

    if validate:
        sdfg.validate()

    return nb_applies


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


@dataclasses.dataclass(frozen=True)
class _FinalConsumerSpec:
    """Describes the final consumer."""

    edge: dace_graph.MultiConnectorEdge[dace.Memlet]

    def consumed_subset(self, state: dace.SDFGState) -> dace_sbs.Range:
        return self.edge.data.get_src_subset(self.edge, state)

    @property
    def consumer(self) -> dace_nodes.Node:
        return self.edge.dst

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
    """Describes how a `concat_where` converges.

    This class only describes a single source of the concat where. To make sense
    one needs a `list` of `_ProducerSpec`, one for each converging edges.

    Args:
        data_name: Name of the data descriptor.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        desc: The data descriptor.
        data_source: Maps a scope location to a data source, i.e. the node and the
            connector where the full producer data, `self` represents, can be accessed.
            You should not access it directly but instead use `get_data_source()`.
    """

    data_name: str
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    desc: dace_data.Data
    data_source: dict[_ScopeLocation, _DataSource]

    def get_data_source(
        self,
        consumer_spec: _FinalConsumerSpec,
        scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
    ) -> _DataSource:
        """Maps a consumer to the data source it should use.

        Note that `scope_dict` should be obtained by `state.scope_dict()` to prevent
        issues when the state is temporary invalid, this dict should be obtained in
        the beginning.
        """
        return self.data_source[scope_dict[consumer_spec.consumer]]

    @property
    def free_symbols(self) -> set[str]:
        # Sometimes `.free_symbols` does not returns `str` but symbols, so we have to
        #  convert them.
        return {
            str(fs)
            for fs in (self.desc.free_symbols | self.offset.free_symbols | self.subset.free_symbols)
        }

    def __copy__(self) -> "_ProducerSpec":
        return _ProducerSpec(
            data_name=self.data_name,
            offset=copy.deepcopy(self.offset),
            subset=copy.deepcopy(self.subset),
            desc=self.desc.clone(),
            data_source=self.data_source.copy(),
        )

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _ProducerSpec):
            return (self.data_name, str(self.subset)) < (other.data_name, str(other.subset))
        return NotImplemented


def _setup_initial_producer_description_on_top_level(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
) -> list[_ProducerSpec]:
    """Sets up the producer specifications at the top level.

    To fully populate the production specification it has to be passed to
    `_map_data_into_nested_scopes()`.

    This function should be called on the top level, i.e. not in nested SDFGs.
    For these cases you should use `_setup_initial_producer_description_in_nested_state()`.

    Args:
        sdfg: The top level SDFG, in which `state` is located.
        state: The state in which `concat_node` is located.
        concat_node: The node that is the result of a concat where expression.
        scope_dict: The scope dictionary of the state.
    """
    assert scope_dict[concat_node] is None
    # First collect the edges that define the concat where expression. We currently
    #  assume that all of them came from AccessNodes, but this is a restriction that
    #  could be lifted.
    converging_edges = list(state.in_edges(concat_node))
    assert all(isinstance(iedge.src, dace_nodes.AccessNode) for iedge in converging_edges)
    assert len(converging_edges) >= 2

    initial_producer_specs: list[_ProducerSpec] = []
    for converging_edge in converging_edges:
        converging_edge.data.try_initialize(sdfg, state, converging_edge)
        producer_node = converging_edge.src
        producer_desc = producer_node.desc(sdfg)
        offset = copy.deepcopy(converging_edge.data.src_subset)
        subset = copy.deepcopy(converging_edge.data.dst_subset)
        assert offset is not None and subset is not None

        initial_producer_specs.append(
            _ProducerSpec(
                data_name=producer_node.data,
                offset=offset,
                subset=subset,
                desc=producer_desc.clone(),
                data_source={None: _DataSource(producer_node, None)},
            )
        )
        assert all(str(fs) in sdfg.symbols for fs in initial_producer_specs[-1].free_symbols)

    return sorted(initial_producer_specs)


def _process_descending_points_of_state(
    state: dace.SDFGState,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
    descending_points: Sequence[_FinalConsumerSpec],
    producer_specs: Sequence[_ProducerSpec],
) -> int:
    """Processes all descending points inside a single state.

    Processes all descending points in the given state. It will configure them if
    needed. However, the function assumes that the producer data has already been
    made available inside the scopes of the descending points, which is the case
    if `_map_data_into_nested_scopes()` has been run on the state.

    Args:
        state: The state containing the descending points.
        scope_dict: The scope dict of `state`.
        descending_points: All descending points inside `state`.
        producer_specs: The producer specification of the parent.
    """

    configured_descending_points: dict[dace_nodes.NestedSDFG, list[_ProducerSpec]] = {}
    nb_applies = 0
    for descending_point in descending_points:
        # If needed configure the descending point.
        nsdfg: dace_nodes.NestedSDFG = descending_point.consumer
        if nsdfg not in configured_descending_points:
            configured_descending_points[nsdfg] = _configure_descending_point(
                state=state,
                scope_dict=scope_dict,
                descending_point=descending_point,
                parent_producer_specs=producer_specs,
            )

        # Process this descending point.
        # NOTE: A nested SDFG might be processed multiple times, but each time
        #   different data is processed.
        nb_applies += _process_descending_points_of_state_impl(
            this_descending_point=descending_point,
            nested_initial_producer_specs=configured_descending_points[nsdfg],
        )

        # Now remove the Memlet "path" that mapped the concat where data into the
        #  nested SDFG and also delete its alias inside it.
        _cleanup_memlet_path(state, descending_point)
        descending_point.consumer.remove_in_connector(descending_point.edge.dst_conn)
        nsdfg.sdfg.remove_data(descending_point.edge.dst_conn, validate=gtx_config.DEBUG)

    return nb_applies


def _process_descending_points_of_state_impl(
    this_descending_point: _FinalConsumerSpec,
    nested_initial_producer_specs: list[_ProducerSpec],
) -> int:
    """Process the interior of a descending point.

    Essentially iterate through all states of the descending point and perform
    the replacement of concat where accesses. The function assumes that
    `this_descending_point` has been configured.
    Furthermore, the function will handled descending points that are located within
    `this_descending_point` by passing them to `_process_descending_points_of_state()`.

    Args:
        this_descending_point: The descending point that is currently processed.
        nested_initial_producer_specs: The producer specification that describes
            the concat where convergence in the context of this descending point.
            It is obtained by a call to `_configure_descending_point()`.

    Note:
        - This function should not be called directly.
        - It is possible that a nested SDFG is handled multiple times, but each time
            it is handled through a different descending point, i.e. the concat where
            data is aliased to multiple data.
    """
    assert all(len(pspec.data_source) == 0 for pspec in nested_initial_producer_specs)

    consumers = _find_consumer_specs_in_descending_point(this_descending_point)
    nb_applies = 0
    for state, (_concat_node, (genuine_consumer_specs, descending_points)) in consumers.items():
        initial_producer_specs = _setup_initial_producer_description_in_nested_state(
            descending_point=this_descending_point,
            nested_state=state,
            nested_initial_producer_specs=nested_initial_producer_specs,
        )
        # `state.scope_dict()` must be called after `_setup_initial_producer_description_in_nested_state()`.
        scope_dict = state.scope_dict()
        producer_specs = _map_data_into_nested_scopes(
            state=state,
            scope_dict=scope_dict,
            initial_producer_specs=initial_producer_specs,
            consumer_specs=genuine_consumer_specs + descending_points,
        )

        # Replace the consumer on this level.
        for consumer_spec in genuine_consumer_specs:
            _replace_single_read(
                sdfg=state.sdfg,
                state=state,
                scope_dict=scope_dict,
                consumer_spec=consumer_spec,
                producer_specs=producer_specs,
                tag=f"{this_descending_point.edge.dst_conn}",
            )
            nb_applies += 1

        # Now recursively descend.
        if len(descending_points) > 0:
            nb_applies += _process_descending_points_of_state(
                state=state,
                scope_dict=scope_dict,
                descending_points=descending_points,
                producer_specs=producer_specs,
            )

        # NOTE: `_concat_node` has been removed already by `_configure_descending_point()`.

    # NOTE: We can not remove the concat where data here, because a descending point
    #   does not equals a nested SDFG. Thus there could be other descending points
    #   that refers to the same nested SDFG. The data is removed in
    #   `_process_descending_points_of_state()`.

    return nb_applies


def _setup_initial_producer_description_in_nested_state(
    descending_point: _FinalConsumerSpec,
    nested_state: dace.SDFGState,
    nested_initial_producer_specs: Sequence[_ProducerSpec],
) -> list[_ProducerSpec]:
    """Analogue operation to `_setup_initial_producer_description_on_top_level()` but on nested levels.

    The function expects that `initial_producer_specs` was generated by a previous
    call to `_configure_descending_point()`, that got the same `descending_point`.
    Furthermore, the function expects that `nested_state` is a state inside the
    descending point.

    In principle the function will do the same as `_setup_initial_producer_description_on_top_level()`,
    i.e. setting up the producer and scope source description. However, it will also
    generate AccessNodes for the mapped data in the state if they are not yet there.

    Args:
        descending_point: The descending point in which we operate.
        nested_state: A state inside the nested SDFG.
        nested_initial_producer_specs: The transformed initial producer
            specification, generated by `_configure_descending_point()`.
    """
    nested_sdfg: dace.SDFG = descending_point.consumer.sdfg
    assert nested_state in nested_sdfg.states()
    assert all(len(prod_spec.data_source) == 0 for prod_spec in nested_initial_producer_specs)

    # Look for access nodes that already exist.
    source_access_nodes: dict[str, dace_nodes.AccessNode] = {}
    needed_data: set[str] = {
        producer_spec.data_name for producer_spec in nested_initial_producer_specs
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

    return [
        _ProducerSpec(
            data_name=prod_spec.data_name,
            offset=copy.deepcopy(prod_spec.offset),
            subset=copy.deepcopy(prod_spec.subset),
            desc=prod_spec.desc.clone(),
            data_source={None: _DataSource(source_access_nodes[prod_spec.data_name], None)},
        )
        for prod_spec in nested_initial_producer_specs
    ]


def _configure_descending_point(
    state: dace.SDFGState,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
    descending_point: _FinalConsumerSpec,
    parent_producer_specs: Sequence[_ProducerSpec],
) -> list[_ProducerSpec]:
    """Handles a descending point.

    The function computes the initial producer specification for the nested SDFG.
    In addition it ensures:
    - That the nested SDFG has the necessary data descriptors and symbols:
    - That the nested SDFG has an updated symbol mapping.
    - That the nested SDFG was connected to the data source.
    - Applies the symbol remapping to the returned producer specifications.

    The function returns a transformed version of the initial producer specification
    that can be used inside the descending point. It must later be passed to
    `_setup_initial_producer_description_in_nested_state()` to obtain a full
    producer specification for a particular state inside the nested SDFG.

    It is important that this function can only be called once per descending point.
    Furthermore, it will not remove the connection between the concat where node
    and the nested SDFG, i.e. `descending_point.edge`.

    Args:
        state: The state containing the descending point.
        scope_dict: The scope dict of `state`.
        descending_point: The consumer that represents a descending point, located in `scope`.
        parent_producer_specs: The initial producer specification on the top level.

    Note:
        You should not use this function directly, instead use `_process_descending_points_of_state()`.
    """
    assert isinstance(descending_point.consumer, dace_nodes.NestedSDFG)
    nsdfg_node: dace_nodes.NestedSDFG = descending_point.consumer
    nsdfg: dace.SDFG = nsdfg_node.sdfg
    symbol_mapping: dict[str, dace_sym.SymExpr] = nsdfg_node.symbol_mapping

    # Now we need to map the data into the nested SDFG. The mapping function only
    #  gives us `inter -> top` so we have to invert it. Inversion here should be
    #  stable because the data is internally ordered.
    nested_to_parent_name_mapping = gtx_transformations.utils.gt_data_descriptor_mapping(
        state=state,
        nsdfg=nsdfg_node,
        only_fully_mapped=True,
        only_inputs=True,
    )
    parent_to_nested_name_mapping = {
        parent: nested for nested, parent in nested_to_parent_name_mapping.items()
    }

    nested_initial_producer_specs: list[_ProducerSpec] = []
    handled_data: dict[str, _ProducerSpec] = {}
    full_repl_dict: dict[str, str] = {}  # All known replacements.
    for parent_prod_spec in parent_producer_specs:
        parent_data_name = parent_prod_spec.data_name

        # Compute the replacement mapping for symbols.
        repl_dict: dict[str, str] = {}  # For this producer.
        for psym in sorted(parent_prod_spec.free_symbols):
            ptype = state.sdfg.symbols[psym]
            if psym in full_repl_dict:
                # The symbol was already handled in a previous producer, so we reuse it.
                nested_symbol = full_repl_dict[psym]
                assert nested_symbol in nsdfg.symbols and nsdfg.symbols[nested_symbol] == ptype

            elif (
                psym in symbol_mapping
                and psym == str(symbol_mapping[psym])
                and psym in nsdfg.symbols
                and nsdfg.symbols[psym] == ptype
            ):
                # The symbol is already mapped 1:1 into the nested sdfg -> nothing to do.
                nested_symbol = psym
                assert nested_symbol not in repl_dict

            else:
                # The symbol is either not known or it is not mapped 1:1 or has the wrong type,
                #  in either case we have to create a new symbol inside the nested SDFG.
                nested_symbol = nsdfg.add_symbol(psym, ptype, find_new_name=True)
                assert nested_symbol not in symbol_mapping
                symbol_mapping[nested_symbol] = psym

            if psym != nested_symbol:
                repl_dict[psym] = nested_symbol
                if psym not in full_repl_dict:
                    full_repl_dict[psym] = nested_symbol

        # Compute the nested data descriptor.
        if parent_data_name in handled_data:
            # The data has already been handled before. So we can simply reuse it.
            #  Symbol renaming has already been handled.
            already_handled_prod_spec = handled_data[parent_data_name]
            nested_desc = already_handled_prod_spec.desc.clone()

        elif parent_data_name in parent_to_nested_name_mapping:
            # We have not handled the data, but it is already mapped into the nested
            #  SDFG, for other reasons. Thus we will reuse this data descriptor.
            #  The symbol renaming is already done and not necessarily described
            #  through `repl_dict`.
            nested_data_name = parent_to_nested_name_mapping[parent_data_name]
            nested_desc = nsdfg.arrays[nested_data_name].clone()
            assert not nested_desc.transient

        else:
            # The data is not yet mapped into the nested SDFG, so we have to create
            #  a new one. And we have to apply symbol renaming on it. In addition
            #  we have to make sure that it is a global.
            nested_desc = parent_prod_spec.desc.clone()
            nested_desc.transient = False
            if repl_dict:
                dace_sym.safe_replace(
                    mapping=repl_dict,
                    replace_callback=functools.partial(
                        dace.sdfg.replace_properties_dict, nested_desc
                    ),
                )
            nested_data_name = nsdfg.add_datadesc(parent_data_name, nested_desc, find_new_name=True)

            # We also need to map it into the nested SDFG.
            source_loc = parent_prod_spec.get_data_source(descending_point, scope_dict)
            state.add_edge(
                source_loc.node,
                source_loc.conn,
                descending_point.consumer,
                nested_data_name,
                dace.Memlet(
                    data=parent_prod_spec.data_name,
                    subset=dace_sbs.Range.from_array(parent_prod_spec.desc),
                ),
            )
            assert nested_data_name not in descending_point.consumer.in_connectors
            descending_point.consumer.add_in_connector(
                nested_data_name, dtype=dace.pointer(nested_desc.dtype)
            )

        # Apply symbol renaming on offset and subset; descriptor was handled above.
        nested_offset = copy.deepcopy(parent_prod_spec.offset)
        nested_subset = copy.deepcopy(parent_prod_spec.subset)
        if repl_dict:
            for nested_set in [nested_offset, nested_subset]:
                dace_sym.safe_replace(  # Performs inplace replacement
                    mapping=repl_dict,
                    replace_callback=nested_set.replace,
                )

        nested_initial_producer_specs.append(
            _ProducerSpec(
                data_name=nested_data_name,
                offset=nested_offset,
                subset=nested_subset,
                desc=nested_desc,
                data_source={},  # Intentionally empty.
            )
        )

    # Restore the proper format of the symbol mapping.
    nsdfg_node.symbol_mapping = symbol_mapping

    return nested_initial_producer_specs


def _map_data_into_nested_scopes(
    state: dace.SDFGState,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
    initial_producer_specs: Sequence[_ProducerSpec],
    consumer_specs: Sequence[_FinalConsumerSpec],
) -> list[_ProducerSpec]:
    """Complete the scope information and ensure that they are available.

    The function returns a finalized version of the production specifications passed
    through `initial_producer_specs`, which were obtained by a previous call to
    `_setup_initial_producer_description_on_top_level()` or
    `_setup_initial_producer_description_in_nested_state()`.
    The main differences between the return value and `initial_producer_specs` is
    that `_ProducerSpec::data_source` has been fully populated and the scope nodes
    are prepared.

    Args:
        state: The state in which we operate.
        scope_dict: The scope dict of `state`.
        initial_producer_specs: The initial producers.
        consumer_specs: All consumers that should be prepared.

    Note:
        In most cases you should pass both the genuine consumer and the descending
        points to this function.
    """
    # Pre-Condition: The top level AccessNodes, which constitute the producers of the
    #  `concat_where` expression must be handled already.
    assert all(None in prod_spec.data_source for prod_spec in initial_producer_specs)
    producer_specs = [copy.copy(producer_spec) for producer_spec in initial_producer_specs]

    # These are the scopes in which _accesses_ to the `concat_where` happens, It does
    #  not (necessarily) contains their parent scopes as well. Note that in order to
    #  make the data accessible in a scope, the data needs to be accessible in all of
    #  its ancestors scopes too. We will bring them in a deterministic order before
    #  process them, handling missing parent scopes on the fly, see
    #  `_map_data_into_nested_scopes_impl()`.
    scopes_containing_consumers: list[_ScopeLocation] = sorted(
        {scope_dict[consumer_spec.consumer] for consumer_spec in consumer_specs},
        key=lambda scope: "NONE" if scope is None else str(scope),
    )
    for scope in scopes_containing_consumers:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_dict=scope_dict,
            scope_to_handle=scope,
            producer_specs=producer_specs,
        )

    return producer_specs


def _map_data_into_nested_scopes_impl(
    state: dace.SDFGState,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
    scope_to_handle: _ScopeLocation,
    producer_specs: Sequence[_ProducerSpec],
) -> None:
    """Helper function of `_map_data_into_nested_scopes()`.

    This function will ensure that the producers of the `concat_where` expression are
    accessible inside `scope_to_handle`. The information is returned inside the
    `data_source` attribute of the passed `producer_specs` objects.
    This is either done by reusing an already existing suitable Memlet or creating
    a new one.

    If the data is not accessible inside the parent scope of `scope_to_handle`, all
    missing ancestor scopes will be handled first.
    Furthermore, it is not an error if `scope_to_handle` was already handled. This
    might happen in case `scope_to_handle` was the parent of a scope previously
    handled (think of nested scopes each containing proper consumers and they are
    processing starting from the most deeply nested one).

    The only condition this function has is that the top level, i.e. `scope_to_handle`
    is `None`, has already been handled.
    """

    # Test if the scope is already handled. It is enough to check the first producer
    #  because if a scope was handled for one producer it was handled for all.
    if scope_to_handle in producer_specs[0].data_source:
        assert all(scope_to_handle in producer_spec.data_source for producer_spec in producer_specs)
        return

    # The global scope, i.e. the AccessNodes referring to the data, has already been
    #  handled outside, thus `scope_to_handle` (at this point) can only be a `MapEntry`.
    assert isinstance(scope_to_handle, dace_nodes.EntryNode)

    # Check if we have to handle the parent scope first. Again it is enough to check
    #  the first producer.
    parent_scope = scope_dict[scope_to_handle]
    if parent_scope not in producer_specs[0].data_source:
        _map_data_into_nested_scopes_impl(
            state=state,
            scope_dict=scope_dict,
            scope_to_handle=parent_scope,
            producer_specs=producer_specs,
        )

    # Now connect the data from `parent_scope` to `scope_to_handle`.
    already_mapped_data: dict[str, _DataSource] = {}
    for producer_spec in producer_specs:
        assert parent_scope in producer_spec.data_source
        assert scope_to_handle not in producer_spec.data_source

        if producer_spec.data_name in already_mapped_data:
            # This data was already seen so reuse the connector, no need to create
            #  an additional edge.
            data_source = already_mapped_data[producer_spec.data_name]

        else:
            # This data has never been seen. Check if there is a already a connection
            #  between scope and its parent (for some other reason). We have to make
            #  sure that the full data is fully transferred.
            parent_source = producer_spec.data_source[parent_scope]
            full_shape = dace_sbs.Range.from_array(producer_spec.desc)  # Might be consumed later.
            potential_reusable_edge = next(
                (
                    oedge
                    for oedge in state.out_edges_by_connector(
                        parent_source.node, parent_source.conn
                    )
                    if (
                        (not oedge.data.is_empty())
                        and oedge.dst is scope_to_handle
                        and oedge.dst_conn.startswith("IN_")
                        # For syntactical reasons one has to use `src_subset` here.
                        and oedge.data.get_src_subset(oedge, state).covers(full_shape)
                    )
                ),
                None,
            )

            if potential_reusable_edge is not None:
                # There was an edge that we can reuse.
                scope_connector = "OUT_" + potential_reusable_edge.dst_conn[3:]
            else:
                # Create a new edge between the `parent_scope` and `scope_to_handle` nodes.
                new_conn_name = scope_to_handle.next_connector(producer_spec.data_name)
                state.add_edge(
                    parent_source.node,
                    parent_source.conn,
                    scope_to_handle,
                    "IN_" + new_conn_name,
                    dace.Memlet(
                        data=producer_spec.data_name,
                        subset=full_shape,  # Intentionally not copied.
                    ),
                )
                scope_to_handle.add_scope_connectors(new_conn_name)
                scope_connector = f"OUT_{new_conn_name}"
            data_source = _DataSource(scope_to_handle, scope_connector)

        producer_spec.data_source[scope_to_handle] = data_source
        if producer_spec.data_name not in already_mapped_data:
            already_mapped_data[producer_spec.data_name] = data_source


def _replace_single_read(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    scope_dict: Mapping[dace_nodes.Node, _ScopeLocation],
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
        sdfg: The sdfg that we process.
        state: The state on which we operate.
        scope_dict: The scope dict of `state`.
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
    consumed_subset = consumer_spec.consumed_subset(state)
    for prod_spec in producer_specs:
        prod_subset = prod_spec.subset
        prod_offsets = prod_spec.offset

        this_select_cond: list[str] = []
        this_prod_access: list[str] = []
        for dim in range(consumed_subset.dims()):
            consumer_access = consumed_subset[dim][0]
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
        f"concat_where_tasklet_{tag}", names_of_existing_tasklets
    )
    concat_where_tasklet = state.add_tasklet(
        tasklet_name,
        inputs=set(tlet_inputs),
        outputs={tlet_output},
        code=tlet_code,
    )

    for tlet_input, producer_spec in zip(tlet_inputs, producer_specs):
        data_source = producer_spec.get_data_source(consumer_spec, scope_dict)
        state.add_edge(
            data_source.node,
            data_source.conn,
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
    _cleanup_memlet_path(state, consumer_spec)


def _find_consumer_specs_in_descending_point(
    descending_point: _FinalConsumerSpec,
) -> dict[dace.SDFGState, tuple[dace_nodes.AccessNode, _ConsumerPartition]]:
    """Scans the descending point for consumer (non recursive).

    The function will scan every state for occurrences of the concat where data
    and return the consumer partition for each state. If there are multiple
    AccessNode referring to the concat where data then the nodes are unified,
    i.e. this function mutates the state.

    Furthermore, this function assumes that the partition exists, i.e.
    `_check_descending_point()` has returned `True`.
    """
    sdfg: dace.SDFG = descending_point.consumer.sdfg
    concat_data = descending_point.edge.dst_conn

    consumers: dict[dace.SDFG, tuple[dace_nodes.AccessNode, _ConsumerPartition]] = {}
    for state in sdfg.states():
        concat_node: Union[None, dace_nodes.AccessNode] = None
        for dnode in state.data_nodes():
            if dnode.data != concat_data:
                continue

            assert state.in_degree(dnode) == 0
            if concat_node is None:
                concat_node = dnode
            else:
                # We found a second node referring to the concat where data,
                #  we have to merge them.
                for oedge in state.out_edges(dnode):
                    dace_transformation.helpers.redirect_edge(
                        state=state, edge=oedge, new_src=concat_node
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
    descending_point: _FinalConsumerSpec,
) -> bool:
    """Check if this descending point has valid consumers, consumer are checked recursively."""
    sdfg: dace.SDFG = descending_point.consumer.sdfg
    concat_data = descending_point.edge.dst_conn
    assert not sdfg.arrays[concat_data].transient

    for state in sdfg.states():
        for dnode in state.data_nodes():
            if dnode.data != concat_data:
                continue
            if state.in_degree(dnode) != 0:
                return False
            scan_for_nested_descending_points = _find_consumer_specs_single_source_single_level(
                state, dnode, for_check=True
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

    If `for_check` is `False` then the function returns the consumer partition for
    `concat_node`. Note if the consumer partition does not exists an error is generated.
    If `for_check` is `True` then the function only checks if the partition exists,
    which is indicated by a list containing all descending points of this state,
    can be empty. `None` is returned if the partition does not exist.
    To further check the returned descending points you can use `_check_descending_point()`.

    Important this function does not:
    - Considering any other state.
    - Considering any other AccessNode in `state` that also refers to the same
        data as `concat_node` does.
    - Does not descend into nested SDFGs.

    Args:
        state: The state in which we operate.
        concat_node: The node representing the `concat_where` result.
        for_check: Only perform checks.

    Note:
        If `for_check` is `False` then both kind of consumers are sorted. For `True`
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
        elif isinstance(consumer_spec.consumer, dace_nodes.NestedSDFG) and (
            consumed_subset.covers(concat_where_shape)
            or _handle_special_case_of_gt4py_scan_point(
                state, consumer_spec, concat_node, consumed_subset
            )
        ):
            # The whole array is mapped into the nested SDFG.
            stairs_to_deeper_levels.append(consumer_spec)
        else:
            if for_check:
                return None
            raise ValueError(f"Expected that partition for {concat_node} exists but it does not.")

    if for_check:
        return stairs_to_deeper_levels
    return sorted(consumer_specs), sorted(stairs_to_deeper_levels)


def _handle_special_case_of_gt4py_scan_point(
    state: dace.SDFGState,
    descending_point: _FinalConsumerSpec,
    concat_node: dace_nodes.AccessNode,
    consumed_subset: dace_sbs.Range,
) -> bool:
    """Performs special checking for cases where data is not properly mapped into a nested SDFG.

    In certain cases, especially for scans, DaCe Memlet propagation (or related) will
    fail to properly annotate the consumer Memlet that maps the data inside a nested
    SDFG. This function applies some heuristic checks if the nested SDFG can be
    processed or not.

    The pathological case is a scan that is inside a Map. In that case it can happen
    that `consumed_subset` is not the full shape of `concat_node` but #  something
    like `[0:i_Cell_gtx_horizontal + 1, 0:3]` this function applies #  some empirical
    checks to see if this is the case and then concludes "it is safe to do".
    """
    assert isinstance(descending_point.consumer, dace_nodes.NestedSDFG)
    nsdfg: dace_nodes.NestedSDFG = descending_point.consumer

    # There must be one node in the nested SDFG that is not a state.
    # TODO(phimuell): Find out if too restrictive.
    if all(isinstance(node, dace.SDFGState) for node in nsdfg.sdfg.nodes()):
        return False

    # At this point `state` should be in a valid state so that it is safe to
    #  compute the scope dict.
    scope_dict = state.scope_dict()
    if scope_dict[nsdfg] is None:
        return False

    parent_desc = concat_node.desc(state.sdfg)
    nested_desc = nsdfg.sdfg.arrays[descending_point.edge.dst_conn]

    if len(parent_desc.shape) != len(nested_desc.shape):
        return False
    if type(parent_desc) is not type(nested_desc):
        return False

    if not all((start == 0) == True for start in consumed_subset.min_element()):  # noqa: E712 [true-false-comparison]  # SymPy comparison
        return False

    # Find all map parameters.
    edge = descending_point.edge
    map_params: set[str] = set()
    while isinstance(edge.src, dace_nodes.MapEntry):
        assert edge.src_conn.startswith("OUT")
        map_params.update(edge.src.map.params)
        edge = next(
            iter(e for e in state.in_edges_by_connector(edge.src, "IN_" + edge.src_conn[4:]))
        )

    assert scope_dict[edge.dst] is None  # really strange case.

    for i, end in enumerate(consumed_subset.max_element()):
        if str(end).isdigit():
            if (end + 1) != parent_desc.shape[i]:  # `+1` because of storage format.
                return False
        else:
            # It is a symbol.
            # TODO(phimuell): For the pathological case one could check
            #  `str(end) not in map_params`. However, I think we could also write
            #  `not end.free_symbols.intersection(map_params)`, which allows DaCe a bit
            #  more slack. However, it allows DaCe more slack. We should check if that
            #  is okay.
            if not map_params.intersection((str(fs) for fs in end.free_symbols)):
                return False

    warnings.warn(
        f"Special rule applied to `concat_where`-inline `{concat_node.data}` into `{nsdfg.label}`.",
        stacklevel=1,
    )
    return True


def _cleanup_memlet_path(state: dace.SDFGState, consumer_spec: _FinalConsumerSpec) -> None:
    """Similar to `remove_edge_and_dangling_path()` but special to our case.

    Removes the full path that leads to `consumer_spec` and removing the ultimate
    source if it has become empty. The connector name of the original consumer
    will never be removed.
    """

    edge: dace_graph.MultiConnectorEdge[dace.Memlet] = consumer_spec.edge
    while True:
        state.remove_edge(edge)

        # We go up as long as `edge.src` is a scoping node.
        if not isinstance(edge.src, dace_nodes.EntryNode):
            break

        # If `edge.src_conn` is in use we are done, otherwise we remove it.
        if len(list(state.out_edges_by_connector(edge.src, edge.src_conn))) != 0:
            break

        # We now find the next edge, which must be done before we remove the connectors.
        other_conn = "IN_" + edge.src_conn[4:]
        next_edges = list(state.in_edges_by_connector(edge.src, other_conn))
        assert len(next_edges) == 1

        # Now remove the connectors.
        edge.src.remove_out_connector(edge.src_conn)
        edge.src.remove_in_connector(other_conn)

        edge = next_edges[0]

    # Test for isolated node and remove it in that case
    if (state.in_degree(edge.src) == 0) and (state.out_degree(edge.src) == 0):
        state.remove_node(edge.src)

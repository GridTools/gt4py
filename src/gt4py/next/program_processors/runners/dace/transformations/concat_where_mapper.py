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


def replace_concat_where_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> list[dace_graph.MultiDiConnectorGraph[dace.Memlet]]:
    """Performs the replacement"""

    consumer_edges = _find_consumer_edges(state, concat_node, map_entry)
    assert len(consumer_edges) > 0

    producer_specs: list[_ProducerDesc] = []
    already_mapped_data: dict[str, int] = {}
    for iedge in state.in_edges(concat_node):
        assert isinstance(iedge.src, dace_nodes.AccessNode)
        producer_node = iedge.src
        data_name = producer_node.data
        full_shape = dace_sbs.Range.from_array(producer_node.desc(sdfg))
        offset = copy.deepcopy(iedge.data.get_src_subset(iedge, state))
        subset = copy.deepcopy(iedge.data.dst_subset)
        if data_name in already_mapped_data:
            prod_source = producer_specs[already_mapped_data[data_name]].prod_source
        else:
            possible_edges_between_producer_and_map = state.edges_between(producer_node, map_entry)
            if len(possible_edges_between_producer_and_map) != 0:
                reuse_this_connection = next(
                    e
                    for e in possible_edges_between_producer_and_map
                    if e.dst_conn and e.dst_conn.startswith("IN_")
                )
                prod_source = ("OUT_" + reuse_this_connection.dst_conn[3:], map_entry)
            else:
                new_conn_name = map_entry.next_connector(data_name)
                state.add_edge(
                    producer_node,
                    None,
                    map_entry,
                    new_conn_name,
                    dace.Memlet(data=data_name, subset=copy.deepcopy(full_shape)),
                )
                # The out connector is dangling, but we will handle it later.
                map_entry.add_scope_connectors(new_conn_name)
            already_mapped_data[data_name] = prod_source

        producer_specs.append(
            _ProducerDesc(
                data_name=data_name,
                full_shape=full_shape,
                offset=offset,
                subset=subset,
                prod_source=prod_source,
            )
        )

    new_consumers: list[dace_graph.MultiDiConnectorGraph[dace.Memlet]] = []
    for consumer_edge in consumer_edges:
        new_consumer = _replace_single_read(
            state=state,
            sdfg=sdfg,
            concat_where_data=concat_node.data,
            consumer_edge=consumer_edge,
            producer_specs=producer_specs,
        )
        new_consumers.append(new_consumer)

    assert state.out_degree(concat_node) == 0
    state.remove_node(concat_node)

    return new_consumers


@dataclasses.dataclass
class _ProducerDesc:
    """Describes a data producer over multiple levels.

    It has the following attributes:

    Args:
        data_name: The name of the data descriptor that provided the original information.
        full_shape: The full shape of the data descriptor.
        offset: The range that is read from the data descriptor.
        subset: The range that is written into the `concat_where` node.
        prod_source: A pair consisting of the connector name and the node where the
            original data can be loaded from.
    """

    data_name: str
    full_shape: dace_sbs.Range
    offset: dace_sbs.Range
    subset: dace_sbs.Range
    prod_source: tuple[str, dace_nodes.Node]


def _replace_single_read(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    concat_where_data: str,
    consumer_edge: dace_graph.MultiDiConnectorGraph[dace.Memlet],
    producer_specs: Sequence[_ProducerDesc],
) -> dace_graph.MultiDiConnectorGraph[dace.Memlet]:
    """Performs the replacement a single read.

    The function will examine the read defined by `consumer_edge`, which is the last
    edge that performs the read from `concat_where_data` (which is the name of the
    data where the concat where converges) with a controlled access, i.e. replaces
    the edge with a Tasklet.
    Note that the destination of `consumer_edge` will remain in the SDFG, instead
    the function will create a new data descriptor, into which the result will be
    written that will then be connected to `consumer_edge`, however, `consumer_edge`
    will be removed.

    Current limitations:
    - The function assumes that the dataflow is in a canonical state.
    - Only one element can be read.
    """

    assert consumer_edge.data.wcr is None
    assert (
        consumer_edge.data.data == concat_where_data
    )  # Avoids some nasty updates; ensured by canonicalization.
    assert consumer_edge.data.subset.num_elements() == 1

    tlet_inputs: list[str] = [f"__inp{i}" for i in range(len(producer_specs))]
    tlet_output = "__out"

    # Start to compose the Tasklets body.
    select_conds: list[str] = []
    prod_accesses: list[str] = []
    consume_subset = consumer_edge.data.src_subset
    for prod_spec in producer_specs:
        prod_subset = prod_spec.subset
        prod_offsets = prod_spec.offset

        this_select_cond: list[str] = []
        this_prod_access: list[str] = []
        for dim in range(consume_subset.dims()):
            consumer_access = consume_subset[dim][0]
            prod_supply_start = prod_subset[dim][0]
            prod_supply_end = prod_subset[dim][1]
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
        for i in range(len(producer_specs) - 1):
            tlet_code_lines.append(
                f"if ({select_conds[i]}):\n\t{tlet_output} = {tlet_inputs[i]}[{prod_accesses[i]}]"
            )
        tlet_code_lines.append(f"else:\n\t{tlet_output} = {tlet_inputs[-1]}[{prod_accesses[-1]}]")
        tlet_code = "\n".join(tlet_code_lines)

    concat_where_tasklet = state.add_tasklet(
        "concat_where_taskelt_that_needs_a_unique_name",
        inputs=set(tlet_inputs),
        outputs={tlet_output},
        code=tlet_code,
    )

    scope_dict = state.scope_dict()
    final_consumer = consumer_edge.dst
    for i in range(len(tlet_inputs)):
        producer_spec = producer_specs[i]
        assert scope_dict[producer_spec.prod_source[1]] is scope_dict[final_consumer]
        state.add_edge(
            producer_spec.prod_source[1],
            producer_spec.prod_source[0],
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
        name=sdfg.temp_data_name(),
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

    # Now connect the intermediate access with the final consumer.
    #  The Memlet definition inherently assumes canonicalized Memlet trees.
    new_consumer_edge = state.add_edge(
        intermediate_node,
        None,
        final_consumer,
        consumer_edge.dst_conn,
        dace.Memlet(
            data=intermediate_data,
            subset="0",
            other_subset=consumer_edge.data.other_subset,
        ),
    )

    # Now remove the old Memlet path.
    dace_sdutils.remove_edge_and_dangling_path(state, consumer_edge)

    return new_consumer_edge


def _find_consumer_edges(
    state: dace.SDFGState,
    concat_node: dace_nodes.AccessNode,
    map_entry: dace_nodes.MapEntry,
) -> list[dace_graph.MultiDiConnectorGraph[dace.Memlet]]:
    """Find all edges that reads from `concat_node` inside the Map defined by `map_entry`."""
    assert state.out_degree(concat_node) == 1
    outer_edge: dace_graph.MultiDiConnectorGraph[dace.Memlet] = next(
        state.edges_between(concat_node, map_entry)
    )
    assert outer_edge.dst_conn.startswith("IN_")
    inner_edge: dace_graph.MultiDiConnectorGraph[dace.Memlet] = state.out_edges_by_connector(
        map_entry, "OUT_" + outer_edge.dst_conn[3:]
    )

    return state.memlet_tree(inner_edge).leaves()

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import functools
from typing import Any, Union

import dace
from dace import properties, subsets, transformation
from dace.sdfg import SDFG, SDFGState, graph as dace_graph, nodes
from dace.transformation import helpers

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util


@properties.make_properties
class KBlocking(transformation.SingleStateTransformation):
    """Performs K blocking.

    This transformation takes a multidimensional map and performs k blocking on
    one particular dimension, which is identified by `block_dim`, which we also
    call `k`.
    All dimensions except `k` are unaffected and the transformation replaces it
    with `kk` and the range `0:N:B`, where `N` is the original end of the
    transformation and `B` is the block size, passed as `blocking_size`.
    The transformation will then add an inner sequential map with one
    dimension `k = kk:(kk+B)`.

    Furthermore, the map will inspect the neighbours of the old, or outer map.
    If the node does not depend on the blocked dimension, the node will be put
    between the two maps, thus its content will only be computed once.
    """

    blocking_size = properties.Property(
        dtype=int,
        allow_none=True,
        desc="Size of the inner k Block.",
    )
    block_dim = properties.Property(
        dtype=str,
        allow_none=True,
        desc="Which dimension should be blocked.",
    )

    map_entry = transformation.transformation.PatternNode(nodes.MapEntry)

    def __init__(
        self,
        blocking_size: int,
        block_dim: Union[gtx_common.Dimension, str],
    ) -> None:
        super().__init__()
        self.blocking_size = blocking_size
        if isinstance(block_dim, str):
            pass
        elif isinstance(block_dim, gtx_common.Dimension):
            block_dim = dace_fieldview_util.get_map_variable(block_dim)
        self.block_dim = block_dim

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Test if the map can be blocked.

        For this the map:
        - Must contain the block dimension.
        - Must not be serial.
        """
        map_entry: nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        map_range: subsets.Range = map_entry.map.range
        block_var: str = self.block_dim

        scope = graph.scope_dict()
        if scope[map_entry] is not None:
            return False
        if block_var not in map_entry.map.params:
            return False
        if map_entry.map.schedule == dace.dtypes.ScheduleType.Sequential:
            return False
        if self.partition_map_output(map_entry, block_var, graph, sdfg) is None:
            return False
        if map_range[map_params.index(block_var)][2] != 1:
            return False

        return True

    def apply(
        self,
        graph: Union[SDFGState, SDFG],
        sdfg: SDFG,
    ) -> None:
        """Performs the blocking transformation."""
        outer_entry: nodes.MapEntry = self.map_entry
        outer_exit: nodes.MapExit = graph.exit_node(outer_entry)
        outer_map: nodes.Map = outer_entry.map
        map_range: subsets.Range = outer_entry.map.range
        map_params: list[str] = outer_entry.map.params

        # This is the name of the iterator we coarsen
        block_var: str = self.block_dim
        block_idx = map_params.index(block_var)

        # This is the name of the iterator that we use in the outer map for the
        #  blocked dimension
        coarse_block_var = "__coarse_" + block_var

        # Now compute the partitions of the nodes.
        independent_nodes, dependent_nodes = self.partition_map_output(  # type: ignore[misc]  # Guaranteed to be not `None`.
            outer_entry, block_var, graph, sdfg
        )

        # Now generate the sequential inner map
        rng_start = map_range[block_idx][0]
        rng_stop = map_range[block_idx][1]
        inner_label = f"inner_{outer_map.label}"
        inner_range = {
            block_var: subsets.Range.from_string(
                f"({coarse_block_var} * {self.blocking_size} + {rng_start}):min(({rng_start} + {coarse_block_var} + 1) * {self.blocking_size}, {rng_stop} + 1)"
            )
        }
        inner_entry, inner_exit = graph.add_map(
            name=inner_label,
            ndrange=inner_range,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        # Now we modify the properties of the outer map.
        coarse_block_range = subsets.Range.from_string(
            f"0:int_ceil(({rng_stop} + 1) - {rng_start}, {self.blocking_size})"
        ).ranges[0]
        outer_map.params[block_idx] = coarse_block_var
        outer_map.range[block_idx] = coarse_block_range

        # Contains the independent nodes that are already relocated.
        relocated_nodes: set[nodes.Node] = set()

        # Now we iterate over all the output edges of the outer map and rewire them.
        #  Note that this only handles the map entry.
        for out_edge in list(graph.out_edges(outer_entry)):
            edge_dst: nodes.Node = out_edge.dst
            edge_conn: str = out_edge.src_conn[4:]

            if edge_dst in dependent_nodes:
                # This is the simple case as we just ave to rewire the edge
                #  and make a connection between the outer and inner map.
                helpers.redirect_edge(
                    state=graph,
                    edge=out_edge,
                    new_src=inner_entry,
                    new_src_conn="OUT_" + edge_conn,
                )
                graph.add_edge(
                    outer_entry,
                    "OUT_" + edge_conn,
                    inner_entry,
                    "IN_" + edge_conn,
                    copy.deepcopy(out_edge.data),
                )
                inner_entry.add_in_connector("IN_" + edge_conn)
                inner_entry.add_out_connector("OUT_" + edge_conn)
                continue

            elif edge_dst in relocated_nodes:
                # See `else` case
                continue

            else:
                # Relocate the node and make the reconnection.
                #  Different from the dependent case we will handle the node fully,
                #  i.e. all of its edges will be processed in one go.
                relocated_nodes.add(edge_dst)

                # This is the node serving as the storage to store the independent
                #  data, and is used within the inner loop.
                #  This prevents the reloading of data.
                assert graph.out_degree(edge_dst) == 1
                caching_node: nodes.AccessNode = next(iter(graph.out_edges(edge_dst)))

                for consumer_edge in list(graph.out_edges(caching_node)):
                    new_map_conn = inner_exit.next_connector()
                    helpers.redirect_edge(
                        state=graph,
                        edge=consumer_edge,
                        new_dst=inner_entry,
                        new_dst_conn="IN_" + new_map_conn,
                    )
                    graph.add_edge(
                        inner_entry,
                        "OUT_" + new_map_conn,
                        consumer_edge.dst,
                        consumer_edge.dst_conn,
                        copy.deepcopy(consumer_edge.data),
                    )
                    inner_entry.add_in_connector("IN_" + new_map_conn)
                    inner_entry.add_out_connector("OUT_" + new_map_conn)
                continue

        # Now we have to handle the output of the map.
        #  There is not many to do just reconnect some edges.
        for out_edge in list(graph.in_edges(outer_exit)):
            edge_conn = out_edge.dst_conn[3:]
            helpers.redirect_edge(
                state=graph,
                edge=out_edge,
                new_dst=inner_exit,
                new_dst_conn="IN_" + edge_conn,
            )
            graph.add_edge(
                inner_exit,
                "OUT_" + edge_conn,
                outer_exit,
                out_edge.dst_conn,
                copy.deepcopy(out_edge.data),
            )
            inner_exit.add_in_connector("IN_" + edge_conn)
            inner_exit.add_out_connector("OUT_" + edge_conn)

        # TODO(phimuell): Use a less expensive method.
        dace.sdfg.propagation.propagate_memlets_state(sdfg, graph)

    def partition_map_output(
        self,
        map_entry: nodes.MapEntry,
        block_param: str,
        state: SDFGState,
        sdfg: SDFG,
    ) -> tuple[set[nodes.Node], set[nodes.Node]] | None:
        """Partition the outputs

        This function computes the partition of the intermediate outputs of the map.
        It will compute two set:
        - The independent outputs `\mathcal{I}`:
            These are output nodes, whose output does not depend on the blocked
            dimension. These nodes will be relocated between the outer and
            inner map.
        - The dependent output `\mathcal{D}`:
            These are the output nodes, whose output depend on the blocked dimension.
            Thus they will not be relocated between the two maps, but remain in the
            inner most scope.

        In case the function fails to compute the partition `None` is returned.

        Args:
            map_entry: The map entry node.
            block_param: The Map variable that should be blocked.
            state: The state on which we operate.
            sdfg: The SDFG in which we operate on.

        Note:
            - Currently this function only considers the input Memlets and the
                `used_symbol` properties of a Tasklet.
            - Furthermore only the first level is inspected.
        """
        block_independent: set[nodes.Node] = set()  # `\mathcal{I}`
        block_dependent: set[nodes.Node] = set()  # `\mathcal{D}`

        # Find all nodes that are adjacent to the map entry.
        nodes_to_partition: set[nodes.Node] = {edge.dst for edge in state.out_edges(map_entry)}

        # Now we examine every node and assign them to a set.
        #  Note that this is only tentative and we will later inspect the
        #  outputs of the independent node and reevaluate the classification.
        for node in nodes_to_partition:
            # Filter out all nodes that we can not (yet) handle.
            if not isinstance(node, (nodes.Tasklet, nodes.AccessNode)):
                return None

            # Check if we have a strange Tasklet or if it uses the symbol inside it.
            if isinstance(node, nodes.Tasklet):
                if node.side_effects:
                    return None
                if block_param in node.free_symbols:
                    block_dependent.add(node)
                    continue

            # Only one output is allowed
            #  Might be less important for Tasklets but for AccessNodes.
            # TODO(phimuell): Lift this restriction.
            if state.out_degree(node) != 1:
                block_dependent.add(node)
                continue

            # Now we have to understand how the node generates its information.
            #  for this we have to look at all the edges that feed information to it.
            edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = list(state.in_edges(node))

            # If the node gets information from other nodes than the map entry
            #  we classify it as a dependent node, although there can be situations
            #  were it could still be an independent node, but figuring this out
            #  is too complicated.
            if any(edge.src is not map_entry for edge in edges):
                block_dependent.add(node)
                continue

            # If all edges are empty, i.e. they are only needed to keep the
            #  node inside the scope, consider it as independent.
            if all(edge.data.is_empty() for edge in edges):
                block_independent.add(node)
                continue

            # Now we have to look at the edges individually.
            #  If this loop ends normally, i.e. it goes into its `else`
            #  clause then we classify the node as independent.
            for edge in edges:
                memlet: dace.Memlet = edge.data
                src_subset: subsets.Subset = memlet.src_subset
                dst_subset: subsets.Subset | None = memlet.dst_subset
                edge_desc: dace.data.Data = sdfg.arrays[memlet.data]
                edge_desc_size = functools.reduce(lambda a, b: a * b, edge_desc.shape)

                if memlet.is_empty():
                    # Empty Memlets do not impose any restrictions.
                    continue
                if src_subset.num_elements() == edge_desc_size:
                    # The whole source array is consumed, which is not a problem.
                    continue

                # Now we have to look at the source and destination set of the Memlet.
                subsets_to_inspect: list[subsets.Subset] = [src_subset]
                if dst_subset is not None:
                    subsets_to_inspect.append(dst_subset)

                # If a subset needs the block variable then the node is not
                #  independent from the block variable.
                if any(block_param in subset.free_symbols for subset in subsets_to_inspect):
                    break
            else:
                # The loop ended normally, thus we did not found anything that made us
                #  believe that the node is _not_ an independent node. We will later
                #  also inspect the output, which might reclassify the node
                block_independent.add(node)

            # If the node is not independent then it must be dependent, my dear Watson.
            if node not in block_independent:
                block_dependent.add(node)

        # We now make a last screening of the independent nodes.
        for independent_node in list(block_independent):
            for out_edge in state.out_edges(independent_node):
                if (
                    (not isinstance(out_edge.dst, nodes.AccessNode))
                    or (state.in_degree(out_edge.dst) != 1)
                    or (sdfg.desc(out_edge.dst).lifetime != dace.dtypes.AllocationLifetime.Scope)
                ):
                    block_independent.discard(independent_node)
                    block_dependent.add(independent_node)
                    break

        assert not block_dependent.intersection(block_independent)
        assert (len(block_independent) + len(block_dependent)) == len(nodes_to_partition)

        return (block_independent, block_dependent)

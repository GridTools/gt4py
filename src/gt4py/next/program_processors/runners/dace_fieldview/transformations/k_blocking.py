# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Optional, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation import helpers as dace_helpers

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import utility as gtx_dace_fieldview_util


@dace_properties.make_properties
class KBlocking(dace_transformation.SingleStateTransformation):
    """Applies k-Blocking with separation on a Map.

    This transformation takes a multidimensional Map and performs blocking on a
    dimension, that is commonly called "k", but identified with `block_dim`.

    All dimensions except `k` are unaffected by this transformation. In the outer
    Map will be replace the `k` range, currently `k = 0:N`, with
    `__coarse_k = 0:N:B`, where `N` is the original size of the range and `B`
    is the block size, passed as `blocking_size`. The transformation also handles the
    case if `N % B != 0`.
    The transformation will then create an inner sequential map with
    `k = __coarse_k:(__coarse_k + B)`.

    However, before the split the transformation examines all adjacent nodes of
    the original Map. If a node does not depend on `k`, then the  node will be
    put between the two maps, thus its content will only be computed once.

    The function will also change the name of the outer map, it will append
    `_blocked` to it.

    Todo:
        - Allow that independent nodes does not need to have direct connection
            to the map, instead it is enough that they only have connections to
            independent nodes.
        - Investigate if the iteration should always start at zero.
    """

    blocking_size = dace_properties.Property(
        dtype=int,
        allow_none=True,
        desc="Size of the inner k Block.",
    )
    block_dim = dace_properties.Property(
        dtype=str,
        allow_none=True,
        desc="Which dimension should be blocked (must be an exact match).",
    )

    map_entry = dace_transformation.transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        blocking_size: Optional[int] = None,
        block_dim: Optional[Union[gtx_common.Dimension, str]] = None,
    ) -> None:
        super().__init__()
        if isinstance(block_dim, gtx_common.Dimension):
            block_dim = gtx_dace_fieldview_util.get_map_variable(block_dim)
        if block_dim is not None:
            self.block_dim = block_dim
        if blocking_size is not None:
            self.blocking_size = blocking_size

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Test if the map can be blocked.

        The test involves:
        - The map must be at the top level.
        - The map shall not be serial.
        - The block dimension must be present (exact string match).
        - The map range must have step size of 1.
        - The partition must exists (see `partition_map_output()`).
        """
        if self.block_dim is None:
            raise ValueError("The blocking dimension was not specified.")
        elif self.blocking_size is None:
            raise ValueError("The blocking size was not specified.")

        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        map_range: dace_subsets.Range = map_entry.map.range
        block_var: str = self.block_dim

        scope = graph.scope_dict()
        if scope[map_entry] is not None:
            return False
        if block_var not in map_entry.map.params:
            return False
        if map_entry.map.schedule == dace.dtypes.ScheduleType.Sequential:
            return False
        if map_range[map_params.index(block_var)][2] != 1:
            return False
        if self.partition_map_output(map_entry, block_var, graph, sdfg) is None:
            return False

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        """Creates a blocking map.

        Performs the operation described in the doc string.
        """
        outer_entry: dace_nodes.MapEntry = self.map_entry
        outer_exit: dace_nodes.MapExit = graph.exit_node(outer_entry)
        outer_map: dace_nodes.Map = outer_entry.map
        map_range: dace_subsets.Range = outer_entry.map.range
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

        # Generate the sequential inner map
        rng_start = map_range[block_idx][0]
        rng_stop = map_range[block_idx][1]
        inner_label = f"inner_{outer_map.label}"
        inner_range = {
            block_var: dace_subsets.Range.from_string(
                f"({coarse_block_var} * {self.blocking_size} + {rng_start}):min(({rng_start} + {coarse_block_var} + 1) * {self.blocking_size}, {rng_stop} + 1)"
            )
        }
        inner_entry, inner_exit = graph.add_map(
            name=inner_label,
            ndrange=inner_range,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        # TODO(phimuell): Investigate if we want to prevent unrolling here

        # Now we modify the properties of the outer map.
        coarse_block_range = dace_subsets.Range.from_string(
            f"0:int_ceil(({rng_stop} + 1) - {rng_start}, {self.blocking_size})"
        ).ranges[0]
        outer_map.params[block_idx] = coarse_block_var
        outer_map.range[block_idx] = coarse_block_range
        outer_map.label = f"{outer_map.label}_blocked"

        # Contains the nodes that are already have been handled.
        relocated_nodes: set[dace_nodes.Node] = set()

        # We now handle all independent nodes, this means that all of their
        #  _output_ edges have to go through the new inner map and the Memlets need
        #  modifications, because of the block parameter.
        for independent_node in independent_nodes:
            for out_edge in graph.out_edges(independent_node):
                edge_dst: dace_nodes.Node = out_edge.dst
                relocated_nodes.add(edge_dst)

                # If destination of this edge is also independent we do not need
                #  to handle it, because that node will also be before the new
                #  inner serial map.
                if edge_dst in independent_nodes:
                    continue

                # Now split `out_edge` such that it passes through the new inner entry.
                #  We do not need to modify the subsets, i.e. replacing the variable
                #  on which we block, because the node is independent and the outgoing
                #  new inner map entry iterate over the blocked variable.
                new_map_conn = inner_entry.next_connector()
                dace_helpers.redirect_edge(
                    state=graph,
                    edge=out_edge,
                    new_dst=inner_entry,
                    new_dst_conn="IN_" + new_map_conn,
                )
                # TODO(phimuell): Check if there might be a subset error.
                graph.add_edge(
                    inner_entry,
                    "OUT_" + new_map_conn,
                    out_edge.dst,
                    out_edge.dst_conn,
                    copy.deepcopy(out_edge.data),
                )
                inner_entry.add_in_connector("IN_" + new_map_conn)
                inner_entry.add_out_connector("OUT_" + new_map_conn)

        # Now we handle the dependent nodes, they differ from the independent nodes
        #  in that they _after_ the new inner map entry. Thus, we will modify incoming edges.
        for dependent_node in dependent_nodes:
            for in_edge in graph.in_edges(dependent_node):
                edge_src: dace_nodes.Node = in_edge.src

                # Since the independent nodes were already processed, and they process
                #  their output we have to check for this. We do this by checking if
                #  the source of the edge is the new inner map entry.
                if edge_src is inner_entry:
                    assert dependent_node in relocated_nodes
                    continue

                # A dependent node has at least one connection to the outer map entry.
                #  And these are the only connections that we must handle, since other
                #  connections come from independent nodes, and were already handled
                #  or are inner nodes.
                if edge_src is not outer_entry:
                    continue

                # If we encounter an empty Memlet we just just attach it to the
                #  new inner map entry. Note the partition function ensures that
                #  either all edges are empty or non.
                if in_edge.data.is_empty():
                    assert (
                        edge_src is outer_entry
                    ), f"Found an empty edge that does not go to the outer map entry, but to '{edge_src}'."
                    dace_helpers.redirect_edge(state=graph, edge=in_edge, new_src=inner_entry)
                    continue

                # Because of the definition of a dependent node and the processing
                #  order, their incoming edges either point to the outer map or
                #  are already handled.
                if edge_src is not outer_entry:
                    sdfg.view()
                assert (
                    edge_src is outer_entry
                ), f"Expected to find source '{outer_entry}' but found '{edge_src}' || {dependent_node} // {edge_src in independent_nodes}."
                edge_conn: str = in_edge.src_conn[4:]

                # Must be before the handling of the modification below
                #  Note that this will remove the original edge from the SDFG.
                dace_helpers.redirect_edge(
                    state=graph,
                    edge=in_edge,
                    new_src=inner_entry,
                    new_src_conn="OUT_" + edge_conn,
                )

                # In a valid SDFG only one edge can go into an input connector of a Map.
                if "IN_" + edge_conn in inner_entry.in_connectors:
                    # We have found this edge multiple times already.
                    #  To ensure that there is no error, we will create a new
                    #  Memlet that reads the whole array.
                    piping_edge = next(graph.in_edges_by_connector(inner_entry, "IN_" + edge_conn))
                    data_name = piping_edge.data.data
                    piping_edge.data = dace.Memlet.from_array(
                        data_name, sdfg.arrays[data_name], piping_edge.data.wcr
                    )

                else:
                    # This is the first time we found this connection.
                    #  so we just create the edge.
                    graph.add_edge(
                        outer_entry,
                        "OUT_" + edge_conn,
                        inner_entry,
                        "IN_" + edge_conn,
                        copy.deepcopy(in_edge.data),
                    )
                    inner_entry.add_in_connector("IN_" + edge_conn)
                    inner_entry.add_out_connector("OUT_" + edge_conn)

        # In certain cases it might happen that we need to create an empty
        #  Memlet between the outer map entry and the inner one.
        if graph.in_degree(inner_entry) == 0:
            graph.add_edge(
                outer_entry,
                None,
                inner_entry,
                None,
                dace.Memlet(),
            )

        # Handle the Map exits
        #  This is simple reconnecting, there would be possibilities for improvements
        #  but we do not use them for now.
        for in_edge in graph.in_edges(outer_exit):
            edge_conn = in_edge.dst_conn[3:]
            dace_helpers.redirect_edge(
                state=graph,
                edge=in_edge,
                new_dst=inner_exit,
                new_dst_conn="IN_" + edge_conn,
            )
            graph.add_edge(
                inner_exit,
                "OUT_" + edge_conn,
                outer_exit,
                in_edge.dst_conn,
                copy.deepcopy(in_edge.data),
            )
            inner_exit.add_in_connector("IN_" + edge_conn)
            inner_exit.add_out_connector("OUT_" + edge_conn)

        # TODO(phimuell): Use a less expensive method.
        dace.sdfg.propagation.propagate_memlets_state(sdfg, graph)

    def partition_map_output(
        self,
        map_entry: dace_nodes.MapEntry,
        block_param: str,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> tuple[set[dace_nodes.Node], set[dace_nodes.Node]] | None:
        """Partition the of the nodes of the Map.

        The outputs will be two sets, defined as:
        - The independent nodes `\mathcal{I}`:
            These are the nodes, whose output does not depend on the blocked
            dimension. These nodes can be relocated between the outer and inner map.
            Nodes in these set does not necessarily have a direct edge to `map_entry`.
            However, the exists a path from `map_entry` to any node in this set.
        - The dependent nodes `\mathcal{D}`:
            These are the nodes, whose output depend on the blocked dimension.
            Thus they can not be relocated between the two maps, but will remain
            inside the inner scope. All these nodes have at least one edge to `map_entry`.

        The function uses an iterative method to compute the set of independent nodes.
        In each iteration the function will look classify all nodes that have an
        incoming edge originating either at `map_entry` or from a node that was
        already classified as independent. This is repeated until no new independent
        nodes are found. This means that independent nodes does not necessarily have
        a direct connection to `map_entry`.

        The dependent nodes on the other side always have a direct edge to `map_entry`.
        As they are the set of nodes that are adjacent to `map_entry` but are not
        independent.

        For the sake of arguments, all nodes, except the map entry and exit nodes,
        that are inside the scope and are not classified as dependent or independent
        are known as "inner nodes".

        In case the function fails to compute the partition `None` is returned.

        Args:
            map_entry: The map entry node.
            block_param: The Map variable that should be blocked.
            state: The state on which we operate.
            sdfg: The SDFG in which we operate on.

        Note:
            - The function will only inspect the direct children of the map entry.
            - Currently this function only considers the input Memlets and the
                `used_symbol` properties of a Tasklet to determine if a Tasklet is dependent.
        """
        independent_nodes: set[dace_nodes.Node] = set()  # `\mathcal{I}`

        while True:
            # Find all the nodes that we have to classify in this iteration.
            #  - All nodes adjacent to `map_entry`
            #  - All nodes adjacent to independent nodes.
            nodes_to_classify: set[dace_nodes.Node] = {
                edge.dst for edge in state.out_edges(map_entry)
            }
            for independent_node in independent_nodes:
                nodes_to_classify.update({edge.dst for edge in state.out_edges(independent_node)})
            nodes_to_classify.difference_update(independent_nodes)

            # Now classify each node
            found_new_independent_node = False
            for node_to_classify in nodes_to_classify:
                class_res = self.classify_node(
                    node_to_classify=node_to_classify,
                    map_entry=map_entry,
                    block_param=block_param,
                    independent_nodes=independent_nodes,
                    state=state,
                    sdfg=sdfg,
                )

                # Check if the partition exists.
                if class_res is None:
                    return None
                if class_res is True:
                    found_new_independent_node = True

            # If we found a new independent node then we have to continue.
            if not found_new_independent_node:
                break

        # After the independent set is computed compute the set of dependent nodes
        #  as the set of all nodes adjacent to `map_entry` that are not dependent.
        dependent_nodes: set[dace_nodes.Node] = {
            edge.dst for edge in state.out_edges(map_entry) if edge.dst not in independent_nodes
        }

        return (independent_nodes, dependent_nodes)

    def classify_node(
        self,
        node_to_classify: dace_nodes.Node,
        map_entry: dace_nodes.MapEntry,
        block_param: str,
        independent_nodes: set[dace_nodes.Node],
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool | None:
        """Internal function for classifying a single node.

        The general rule to classify if a node is independent are:
        - The node must be a Tasklet or an AccessNode, in all other cases the
            partition does not exist.
        - `free_symbols` of the nodes shall not contain the `block_param`.
        - All incoming _empty_ edges must be connected to the map entry.
        - A node either has only empty Memlets or none of them.
        - Incoming Memlets does not depend on the `block_param`.
        - All incoming edges must start either at `map_entry` or at dependent nodes.
        - All output Memlets are non empty.

        Returns:
            The function returns `True` if `node_to_classify` is considered independent.
            In this case the function will add the node to `independent_nodes`.
            If the function returns `False` the node was classified as a dependent node.
            The function will return `None` if the node can not be classified, in this
            case the partition does not exist.

        Args:
            node_to_classify: The node that should be classified.
            map_entry: The entry of the map that should be partitioned.
            block_param: The iteration parameter that should be blocked.
            independent_nodes: The set of nodes that was already classified as
                independent, in case the node is classified as independent the set is
                updated.
            state: The state containing the map.
            sdfg: The SDFG that is processed.
        """

        # We are only able to handle certain kind of nodes, so screening them.
        if isinstance(node_to_classify, dace_nodes.Tasklet):
            if node_to_classify.side_effects:
                # TODO(phimuell): Think of handling it.
                return None
        elif isinstance(node_to_classify, dace_nodes.AccessNode):
            # AccessNodes need to have some special properties.
            node_desc: dace.data.Data = node_to_classify.desc(sdfg)

            if isinstance(node_desc, dace.data.View):
                # Views are forbidden.
                return None
            if node_desc.lifetime != dace.dtypes.AllocationLifetime.Scope:
                # The access node has to life fully within the scope.
                return None
        else:
            # Any other node type we can not handle, so the partition can not exist.
            # TODO(phimuell): Try to handle certain kind of library nodes.
            return None

        # Now we have to understand how the node generates its data. For this we have
        #  to look at all the incoming edges.
        in_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = list(
            state.in_edges(node_to_classify)
        )

        # In a first phase we will only look if the partition exists or not.
        #  We will therefore not check if the node is independent or not, since
        #  for these classification to make sense the partition has to exist in the
        #  first place.

        # Either all incoming edges of a node are empty or none of them. If it has
        #  empty edges, they are only allowed to come from the map entry.
        found_empty_edges, found_nonempty_edges = False, False
        for in_edge in in_edges:
            if in_edge.data.is_empty():
                found_empty_edges = True
                if in_edge.src is not map_entry:
                    # TODO(phimuell): Lift this restriction.
                    return None
            else:
                found_nonempty_edges = True

        # Test if we found a mixture of empty and nonempty edges.
        if found_empty_edges and found_nonempty_edges:
            return None
        assert (
            found_empty_edges or found_nonempty_edges
        ), f"Node '{node_to_classify}' inside '{map_entry}' without an input connection."

        # Requiring that all output Memlets are non empty implies, because we are
        #  inside a scope, that there exists an output.
        if any(out_edge.data.is_empty() for out_edge in state.out_edges(node_to_classify)):
            return None

        # Now we have ensured that the partition exists, thus we will now evaluate
        #  if the node is independent or dependent.

        # Test if the body of the Tasklet depends on the block variable.
        if (
            isinstance(node_to_classify, dace_nodes.Tasklet)
            and block_param in node_to_classify.free_symbols
        ):
            return False

        # Now we have to look at incoming edges individually.
        #  We will inspect the subset of the Memlet to see if they depend on the
        #  block variable. If this loop ends normally, then we classify the node
        #  as independent and the node is added to `independent_nodes`.
        for in_edge in in_edges:
            memlet: dace.Memlet = in_edge.data
            src_subset: dace_subsets.Subset | None = memlet.src_subset
            dst_subset: dace_subsets.Subset | None = memlet.dst_subset

            if memlet.is_empty():  # Empty Memlets do not impose any restrictions.
                continue

            # Now we have to look at the source and destination set of the Memlet.
            subsets_to_inspect: list[dace_subsets.Subset] = []
            if dst_subset is not None:
                subsets_to_inspect.append(dst_subset)
            if src_subset is not None:
                subsets_to_inspect.append(src_subset)

            # If a subset needs the block variable then the node is not independent
            #  but dependent.
            if any(block_param in subset.free_symbols for subset in subsets_to_inspect):
                return False

            # The edge must either originate from `map_entry` or from an independent
            #  node if not it is dependent.
            if not (in_edge.src is map_entry or in_edge.src in independent_nodes):
                return False

        # Loop ended normally, thus we classify the node as independent.
        independent_nodes.add(node_to_classify)
        return True

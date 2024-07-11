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

"""Implements Helper functionaliyies for map fusion"""

import functools
import itertools
from typing import Any, Optional, Sequence, Union

import dace
from dace import data, properties, subsets, transformation
from dace.sdfg import SDFG, SDFGState, graph as dace_graph, nodes
from dace.transformation import helpers

from . import util


@properties.make_properties
class MapFusionHelper(transformation.SingleStateTransformation):
    """
    Contains common part of the map fusion for parallel and serial map fusion.

    See also [this HackMD document](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG)
    about the underlying assumption this transformation makes.

    After every transformation that manipulates the state machine, you shouls recreate
    the transformation.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
    """

    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are on the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    shared_transients = properties.DictProperty(
        key_type=SDFG,
        value_type=set[str],
        default=None,
        allow_none=True,
        desc="Maps SDFGs to the set of array transients that can not be removed. "
        "The variable acts as a cache, and is managed by 'is_interstate_transient()'.",
    )

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        self.shared_transients = {}

    @classmethod
    def expressions(cls) -> bool:
        raise RuntimeError("The `_MapFusionHelper` is not a transformation on its own.")

    def can_be_fused(
        self,
        map_entry_1: nodes.MapEntry,
        map_entry_2: nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Performs some checks if the maps can be fused.

        Essentially, this function only checks constrains that does not depend if
        a serial or a parallel map fusion happens. Thus it tests:
        - The scope of the maps.
        - The scheduling of the maps.
        - The map parameters.

        However, for performance reasons, the function does not check if the node
        decomposition exists.
        """

        if self.only_inner_maps and self.only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")

        # ensure that both have the same schedule
        if map_entry_1.map.schedule != map_entry_2.map.schedule:
            return False

        # Fusing is only possible if our two entries are in the same scope.
        scope = graph.scope_dict()
        if scope[map_entry_1] != scope[map_entry_2]:
            return False
        elif self.only_inner_maps:
            if scope[map_entry_1] is None:
                return False
        elif self.only_toplevel_maps:
            if scope[map_entry_1] is not None:
                return False
            elif util.is_nested_sdfg(sdfg):
                return False

        # We will now check if there exists a remapping that we can use.
        if not self.map_parameter_compatible(
            map_1=map_entry_1.map, map_2=map_entry_2.map, state=graph, sdfg=sdfg
        ):
            return False

        return True

    @staticmethod
    def relocate_nodes(
        from_node: Union[nodes.MapExit, nodes.MapEntry],
        to_node: Union[nodes.MapExit, nodes.MapEntry],
        state: SDFGState,
        sdfg: SDFG,
    ) -> None:
        """Move the connectors and edges from `from_node` to `to_nodes` node.

        Note:
            - This function dos not remove the `from_node` but it will have degree
                zero and have no connectors.
            - If this function fails, the SDFG is in an invalid state.
            - Usually this function should be called twice per Map scope, once for the
                entry node and once for the exit node.
        """

        # Now we relocate empty Memlets, from the `from_node` to the `to_node`
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.out_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_src=to_node)
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.in_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_dst=to_node)

        # We now ensure that there is only one empty Memlet from the `to_node` to any other node.
        #  Although it is allowed, we try to prevent it.
        empty_targets: set[nodes.Node] = set()
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
            if empty_edge.dst in empty_targets:
                state.remove_edge(empty_edge)
            empty_targets.add(empty_edge.dst)

        # We now determine if which connections we have to migrate
        #  We only consider the in edges, for Map exits it does not matter, but for
        #  Map entries, we need it for the dynamic map range feature.
        for edge_to_move in list(state.in_edges(from_node)):
            assert isinstance(edge_to_move.dst_conn, str)

            if not edge_to_move.dst_conn.startswith("IN_"):
                # Dynamic Map Range
                #  The connector name simply defines a variable name that is used,
                #  inside the Map scope to define a variable. We handle it directly.
                assert isinstance(from_node, nodes.MapEntry)
                dmr_symbol = edge_to_move.dst_conn

                # TODO(phimuell): Check if the symbol is really unused.
                if dmr_symbol in to_node.in_connectors:
                    raise NotImplementedError(
                        f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                        f" to '{to_node}', but the symbol is already known there, but the"
                        " renaming is not implemented."
                    )
                if not to_node.add_in_connector(dmr_symbol, force=False):
                    raise RuntimeError(  # Might fail because of out connectors.
                        f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'."
                    )
                helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
                from_node.remove_in_connector(dmr_symbol)

                # There is no other edge that we have to consider, so we just end here
                continue

            # We have a Passthrough connection, i.e. there exists a `OUT_` connector
            #  thus we now have to migrate the two edges.
            old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
            new_conn = to_node.next_connector(old_conn)

            to_node.add_in_connector("IN_" + new_conn)
            from_node.remove_in_connector("IN_" + old_conn)
            for e in list(state.in_edges_by_connector(from_node, "IN_" + old_conn)):
                helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn="IN_" + new_conn)
            from_node.remove_out_connector("OUT_" + old_conn)
            to_node.add_out_connector("OUT_" + new_conn)
            for e in list(state.out_edges_by_connector(from_node, "OUT_" + old_conn)):
                helpers.redirect_edge(state, e, new_src=to_node, new_src_conn="OUT_" + new_conn)

        assert (
            state.in_degree(from_node) == 0
        ), f"After moving source node '{from_node}' still has an input degree of {state.in_degree(from_node)}"
        assert (
            state.out_degree(from_node) == 0
        ), f"After moving source node '{from_node}' still has an output degree of {state.out_degree(from_node)}"

    @staticmethod
    def map_parameter_compatible(
        map_1: nodes.Map,
        map_2: nodes.Map,
        state: Union[SDFGState, SDFG],
        sdfg: SDFG,
    ) -> bool:
        """Checks if `map_1` is compatible with `map_1`.

        The check follows the following rules:
        - The names of the map variables must be the same, i.e. no renaming
            is performed.
        - The ranges must be the same.
        """
        range_1: subsets.Range = map_1.range
        params_1: Sequence[str] = map_1.params
        range_2: subsets.Range = map_2.range
        params_2: Sequence[str] = map_2.params

        # The maps are only fuseable if we have an exact match in the parameter names
        #  this is because we do not do any renaming.
        if set(params_1) != set(params_2):
            return False

        # Maps the name of a parameter to the dimension index
        param_dim_map_1: dict[str, int] = {pname: i for i, pname in enumerate(params_1)}
        param_dim_map_2: dict[str, int] = {pname: i for i, pname in enumerate(params_2)}

        # To fuse the two maps the ranges must have the same ranges
        for pname in params_1:
            idx_1 = param_dim_map_1[pname]
            idx_2 = param_dim_map_2[pname]
            # TODO(phimuell): do we need to call simplify
            if range_1[idx_1] != range_2[idx_2]:
                return False

        return True

    def is_interstate_transient(
        self,
        transient: Union[str, nodes.AccessNode],
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `transient` is an interstate transient, an can not be removed.

        Essentially this function checks if a transient is needed in a
        different state in the SDFG, because it transmit information from
        one state to the other. However, this function only checks if the
        transient is needed for transmitting information between states.
        It does _not_ check if the transient is needed multiple times within
        the state. This case can be detected by checking the number of outgoing
        edges.

        Args:
            transient: The transient that should be checked.
            sdfg: The SDFG containing the array.
        """

        # According to [rule 6](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG)
        #  the set of such transients is partially given by all source access nodes.
        #  Because of rule 3 we also include all scalars in this set, as an over
        #  approximation. Furthermore, because simplify might violate rule 3,
        #  we also include the sink nodes.

        # See if we have already computed the set
        if sdfg in self.shared_transients:
            shared_sdfg_transients: set[str] = self.shared_transients[sdfg]

        else:
            # SDFG is not known so we have to compute it.
            #  If a scalar is not a source node then it is not included in this set.
            #  Thus we do not have to look for it, instead we will check for them
            #  explicitly.
            shared_sdfg_transients = set()
            for state in sdfg.states():
                shared_sdfg_transients.update(
                    filter(
                        lambda node: isinstance(node, nodes.AccessNode)
                        and sdfg.arrays[node.data].transient,
                        itertools.chain(state.source_nodes(), state.sink_nodes()),
                    )
                )
            self.shared_transients[sdfg] = shared_sdfg_transients

        if isinstance(transient, nodes.AccessNode):
            transient = transient.data
        desc: data.Data = sdfg.arrays[transient]

        if not desc.transient:
            return False
        if isinstance(desc, data.Scalar):
            return False
        return transient in shared_sdfg_transients

    def partition_first_outputs(
        self,
        state: SDFGState,
        sdfg: SDFG,
        map_exit_1: nodes.MapExit,
        map_entry_2: nodes.MapEntry,
    ) -> Union[
        tuple[
            set[dace_graph.MultiConnectorEdge[dace.Memlet]],
            set[dace_graph.MultiConnectorEdge[dace.Memlet]],
            set[dace_graph.MultiConnectorEdge[dace.Memlet]],
        ],
        None,
    ]:
        """Partition the output edges of `map_exit_1` for serial map fusion.

        The output edges of the first map are partitioned into three distinct sets,
        defined as follows:

        - Pure Output Set `\mathbb{P}`:
            These edges exits the first map and does not enter the second map. These
            outputs will be simply be moved to the output of the second map.
        - Exclusive Intermediate Set `\mathbb{E}`:
            Edges in this set leaves the first map exit and enters an access node, from
            where a Memlet then leads immediately to the second map. The memory
            referenced by this access node is not needed anywhere else, thus it will
            be removed.
        - Shared Intermediate Set `\mathbb{S}`:
            These edges are very similar to the one in `\mathbb{E}` except that they
            are used somewhere else, thus they can not be removed and are recreated
            as output of the second map.

        Returns:
            If such a decomposition exists the function will return the three sets
            mentioned above in the same order.
            In case the decomposition does not exist, i.e. the maps can not be fused
            serially, the function returns `None`.

        Args:
            state: The in which the two maps are located.
            sdfg: The full SDFG in whcih we operate.
            map_exit_1: The exit node of the first map.
            map_entry_2: The entry node of the second map.
        """
        # The three outputs set.
        pure_outputs: set[dace_graph.MultiConnectorEdge[dace.Memlet]] = set()
        exclusive_outputs: set[dace_graph.MultiConnectorEdge[dace.Memlet]] = set()
        shared_outputs: set[dace_graph.MultiConnectorEdge[dace.Memlet]] = set()

        # Set of intermediate nodes that we have already processed.
        processed_inter_nodes: set[nodes.Node] = set()

        # Now scan all output edges of the first exit and classify them
        for out_edge in state.out_edges(map_exit_1):
            intermediate_node: nodes.Node = out_edge.dst

            # We already processed the node, this should indicate that we should
            #  run simplify again, or we should start implementing this case.
            if intermediate_node in processed_inter_nodes:
                return None
            processed_inter_nodes.add(intermediate_node)

            # Empty Memlets are currently not supported.
            #  However, they are much more important in entry nodes.
            if out_edge.data.is_empty():
                return None

            # Now let's look at all nodes that are downstream of the intermediate node.
            #  This, among other things, will tell us, how we have to handle this node.
            downstream_nodes = util.all_nodes_between(
                graph=state,
                begin=intermediate_node,
                end=map_entry_2,
            )

            # If `downstream_nodes` is `None` this means that `map_entry_2` was never
            #  reached, thus `intermediate_node` does not enter the second map and
            #  the node is a pure output node.
            if downstream_nodes is None:
                pure_outputs.add(out_edge)
                continue
            #

            # The following tests are _after_ we have determined if we have a pure
            #  output node for a reason, as this allows us to handle more exotic
            #  pure node cases, as handling them is essentially rerouting an edge.

            # In case the intermediate has more than one entry, all must come from the
            #  first map, otherwise we can not fuse them. Currently we restrict this
            #  even further by saying that it has only one incoming Memlet.
            if state.in_degree(intermediate_node) != 1:
                # TODO(phimuell): handle this case.
                return None

            # It can happen that multiple edges converges at the `IN_` connector
            #  of the first map exit, but there is only one edge leaving the exit.
            #  It is complicate to handle this, so for now we ignore it.
            # TODO(phimuell): Handle this case properly.
            inner_collector_edges = list(
                state.in_edges_by_connector(intermediate_node, "IN_" + out_edge.src_conn[3:])
            )
            if len(inner_collector_edges) > 1:
                return None

            # For us an intermediate node must always be an access node, pointing to a
            #  transient value, since it is the only thing that we know how to handle.
            if not isinstance(intermediate_node, nodes.AccessNode):
                return None
            intermediate_desc: data.Data = intermediate_node.desc(sdfg)
            if not intermediate_desc.transient:
                return None
            if isinstance(intermediate_desc, data.View):
                return None

            # There are some restrictions we have on intermediate nodes. The first one
            #  is that we do not allow WCR, this is because they need special handling
            #  which is currently not implement (the DaCe transformation has this
            #  restriction as well). The second one is that we can reduce the
            #  intermediate node and only feed a part into the second map, consider
            #  the case `b = a + 1; return b + 2`, where we have arrays. In this
            #  example only a single element must be available to the second map.
            #  However, this is hard to check so we will make a simplification.
            #  First we will not check it at the producer, but at the consumer point.
            #  There we assume if the consumer does _not consume the whole_
            #  intermediate array, then we can decompose the intermediate, by setting
            #  the map iteration index to zero and recover the shape, see
            #  implementation in the actual fusion routine.
            #  This is an assumption that is in most cases correct, but not always.
            #  However, doing it correctly is extremely complex.
            for _, produce_edge in util.find_upstream_producers(state, out_edge):
                if produce_edge.data.wcr is not None:
                    return None

            if len(downstream_nodes) == 0:
                # There is nothing between intermediate node and the entry of the
                #  second map, thus the edge belongs either in `\mathbb{S}` or
                #  `\mathbb{E}`, to which one depends on how it is used.

                # This is a very special situation, i.e. the access node has many
                #  different connections to the second map entry, this is a special
                #  case that we do not handle, instead simplify should be called.
                if state.out_degree(intermediate_node) != 1:
                    return None

                # Certain nodes need more than one element as input. As explained
                #  above, in this situation we assume that we can naturally decompose
                #  them iff the node does not consume that whole intermediate.
                #  Furthermore, it can not be a dynamic map range.
                intermediate_size = functools.reduce(lambda a, b: a * b, intermediate_desc.shape)
                consumers = util.find_downstream_consumers(state=state, begin=intermediate_node)
                for consumer_node, feed_edge in consumers:
                    # TODO(phimuell): Improve this approximation.
                    if feed_edge.data.num_elements() == intermediate_size:
                        return None
                    if consumer_node is map_entry_2:  # Dynamic map range.
                        return None

                # Note that "remove" has a special meaning here, regardless of the
                #  output of the check function, from within the second map we remove
                #  the intermediate, it has more the meaning of "do we need to
                #  reconstruct it after the second map again?".
                #  NOTE: The case "used in this state" is handled above!!
                if self.is_interstate_transient(intermediate_node, sdfg):
                    shared_outputs.add(out_edge)
                else:
                    exclusive_outputs.add(out_edge)
                continue

            else:
                # These is not only a single connection from the intermediate node to
                #  the second map, but the intermediate has more connection, thus
                #  the node might belong to the shared outputs. Of the many different
                #  possibilities, we only consider a single case:
                #  - The intermediate has a single connection to the second map, that
                #       fulfills the restriction outlined above.
                #  - All other connections have no connection to the second map.
                found_second_entry = False
                intermediate_size = functools.reduce(lambda a, b: a * b, intermediate_desc.shape)
                for edge in state.out_edges(intermediate_node):
                    if edge.dst is map_entry_2:
                        if found_second_entry:  # The second map was found again.
                            return None
                        found_second_entry = True
                        consumers = util.find_downstream_consumers(state=state, begin=edge)
                        for consumer_node, feed_edge in consumers:
                            if feed_edge.data.num_elements() == intermediate_size:
                                return None
                            if consumer_node is map_entry_2:  # Dynamic map range
                                return None
                    else:
                        # Ensure that there is no path that leads to the second map.
                        after_intermdiate_node = util.all_nodes_between(
                            graph=state, begin=edge.dst, end=map_entry_2
                        )
                        if after_intermdiate_node is not None:
                            return None
                # If we are here, then we know that the node is a shared output
                shared_outputs.add(out_edge)
                continue

        assert exclusive_outputs or shared_outputs or pure_outputs
        return (pure_outputs, exclusive_outputs, shared_outputs)

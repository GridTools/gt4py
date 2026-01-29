# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import uuid
from typing import Any, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation import helpers as dace_helpers

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


def unique_name(name: str) -> str:
    """Adds a unique string to `name`."""
    maximal_length = 200
    unique_sufix = str(uuid.uuid1()).replace("-", "_")
    if len(name) > (maximal_length - len(unique_sufix)):
        name = name[: (maximal_length - len(unique_sufix) - 1)]
    return f"{name}_{unique_sufix}"


@dace_properties.make_properties
class FuseHorizontalConditionBlocks(dace_transformation.SingleStateTransformation):
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    first_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)
    second_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = dace_graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nedge(
            cls.access_node, cls.first_conditional_block, dace.Memlet()
        )
        map_fusion_parallel_match.add_nedge(
            cls.access_node, cls.second_conditional_block, dace.Memlet()
        )
        return [map_fusion_parallel_match]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        access_node: dace_nodes.AccessNode = self.access_node
        access_node_desc = access_node.desc(sdfg)
        first_cb: dace_nodes.NestedSDFG = self.first_conditional_block
        second_cb: dace_nodes.NestedSDFG = self.second_conditional_block
        scope_dict = graph.scope_dict()

        # Check that both conditional blocks are in the same parent SDFG
        if first_cb.sdfg.parent != second_cb.sdfg.parent:
            return False

        # Check that the nested SDFGs' names starts with "if_stmt"
        if not (
            first_cb.sdfg.name.startswith("if_stmt") and second_cb.sdfg.name.startswith("if_stmt")
        ):
            return False

        # Check that the common access node is a boolean scalar
        if (
            not isinstance(access_node_desc, dace.data.Scalar)
            or access_node_desc.dtype != dace.bool_
        ):
            return False

        # Make sure that the conditional blocks contain only one conditional block each
        if len(first_cb.sdfg.nodes()) > 1 or len(second_cb.sdfg.nodes()) > 1:
            return False

        # Get the actual conditional blocks
        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))
        if not (
            isinstance(first_conditional_block, dace.sdfg.state.ConditionalBlock)
            and isinstance(second_conditional_block, dace.sdfg.state.ConditionalBlock)
        ):
            return False

        # Make sure that both conditional blocks are in the same scope
        if scope_dict[first_cb] != scope_dict[second_cb]:
            return False

        # Make sure that both conditional blocks are in a map scope
        if not isinstance(scope_dict[first_cb], dace.nodes.MapEntry):
            return False

        # Check that there is an edge to the conditional blocks with dst_conn == "__cond"
        cond_edges_first = [e for e in graph.in_edges(first_cb) if e.dst_conn == "__cond"]
        if len(cond_edges_first) != 1:
            return False
        cond_edges_second = [e for e in graph.in_edges(second_cb) if e.dst_conn == "__cond"]
        if len(cond_edges_second) != 1:
            return False
        cond_edge_first = cond_edges_first[0]
        cond_edge_second = cond_edges_second[0]
        if cond_edge_first.src != cond_edge_second.src and (
            cond_edge_first.src != access_node or cond_edge_second.src != access_node
        ):
            return False

        # Need to check also that first and second nested SDFGs are not reachable from each other
        if gtx_transformations.utils.is_reachable(
            start=first_cb,
            target=second_cb,
            state=graph,
        ) or gtx_transformations.utils.is_reachable(
            start=second_cb,
            target=first_cb,
            state=graph,
        ):
            return False

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node
        first_cb: dace_nodes.NestedSDFG = self.first_conditional_block
        second_cb: dace_nodes.NestedSDFG = self.second_conditional_block

        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))

        # Store original arrays to check later that all the necessary arrays have been moved
        original_arrays_first_conditional_block = {}
        for data_name, data_desc in first_conditional_block.sdfg.arrays.items():
            original_arrays_first_conditional_block[data_name] = data_desc
        original_arrays_second_conditional_block = {}
        for data_name, data_desc in second_conditional_block.sdfg.arrays.items():
            original_arrays_second_conditional_block[data_name] = data_desc
        total_original_arrays = len(original_arrays_first_conditional_block) + len(
            original_arrays_second_conditional_block
        )

        # Store the new names for the arrays in the second conditional block to avoid name clashes and add their data descriptors
        # to the first conditional block SDFG
        second_arrays_rename_map = {}
        for data_name, data_desc in original_arrays_second_conditional_block.items():
            if data_name == "__cond":
                continue
            if data_name in original_arrays_first_conditional_block:
                new_data_name = unique_name(data_name)
                second_arrays_rename_map[data_name] = new_data_name
                data_desc_renamed = copy.deepcopy(data_desc)
                data_desc_renamed.name = new_data_name
                if new_data_name not in first_cb.sdfg.arrays:
                    first_cb.sdfg.add_datadesc(new_data_name, data_desc_renamed)
            else:
                second_arrays_rename_map[data_name] = data_name
                if data_name not in first_cb.sdfg.arrays:
                    first_cb.sdfg.add_datadesc(data_name, copy.deepcopy(data_desc))

        second_conditional_states = list(second_conditional_block.all_states())

        # Move the connectors from the second conditional block to the first
        in_connectors_to_move = {k: v for k, v in second_cb.in_connectors.items() if k != "__cond"}
        out_connectors_to_move = second_cb.out_connectors
        in_connectors_to_move_rename_map = {}
        out_connectors_to_move_rename_map = {}
        for k, _v in in_connectors_to_move.items():
            new_connector_name = k
            if new_connector_name in first_cb.in_connectors:
                new_connector_name = second_arrays_rename_map[k]
            in_connectors_to_move_rename_map[k] = new_connector_name
            first_cb.add_in_connector(new_connector_name)
            for edge in graph.in_edges(second_cb):
                if edge.dst_conn == k:
                    dace_helpers.redirect_edge(
                        state=graph, edge=edge, new_dst_conn=new_connector_name, new_dst=first_cb
                    )
        for k, _v in out_connectors_to_move.items():
            new_connector_name = k
            if new_connector_name in first_cb.out_connectors:
                new_connector_name = second_arrays_rename_map[k]
            out_connectors_to_move_rename_map[k] = new_connector_name
            first_cb.add_out_connector(new_connector_name)
            for edge in graph.out_edges(second_cb):
                if edge.src_conn == k:
                    dace_helpers.redirect_edge(
                        state=graph, edge=edge, new_src_conn=new_connector_name, new_src=first_cb
                    )

        def _find_corresponding_state_in_second(
            inner_state: dace.SDFGState,
        ) -> dace.SDFGState:
            inner_state_name = inner_state.name
            true_branch = "true_branch" in inner_state_name
            corresponding_state_in_second = None
            for state in second_conditional_states:
                if true_branch and "true_branch" in state.name:
                    corresponding_state_in_second = state
                    break
                elif not true_branch and "false_branch" in state.name:
                    corresponding_state_in_second = state
                    break
            if corresponding_state_in_second is None:
                raise RuntimeError(
                    f"Could not find corresponding state in second conditional block for state {inner_state_name}"
                )
            return corresponding_state_in_second

        # Copy first the nodes from the second conditional block to the first
        nodes_renamed_map = {}
        for first_inner_state in first_conditional_block.all_states():
            corresponding_state_in_second = _find_corresponding_state_in_second(first_inner_state)
            nodes_to_move = list(corresponding_state_in_second.nodes())
            for node in nodes_to_move:
                new_node = node
                if isinstance(node, dace_nodes.AccessNode):
                    if node.data in first_cb.in_connectors or node.data in first_cb.out_connectors:
                        new_data_name = second_arrays_rename_map[node.data]
                        new_node = dace_nodes.AccessNode(new_data_name)
                        new_desc = copy.deepcopy(node.desc(second_cb.sdfg))
                        new_desc.name = new_data_name
                nodes_renamed_map[node] = new_node
                first_inner_state.add_node(new_node)

        # Then copy the edges
        second_to_first_connections = {}
        for node in nodes_renamed_map:
            if isinstance(node, dace_nodes.AccessNode):
                second_to_first_connections[node.data] = nodes_renamed_map[node].data
        for first_inner_state in first_conditional_block.all_states():
            corresponding_state_in_second = _find_corresponding_state_in_second(first_inner_state)
            nodes_to_move = list(corresponding_state_in_second.nodes())
            for node in nodes_to_move:
                for edge in list(corresponding_state_in_second.out_edges(node)):
                    dst = edge.dst
                    if dst in nodes_to_move:
                        new_memlet = copy.deepcopy(edge.data)
                        if edge.data.data in second_to_first_connections:
                            new_memlet.data = second_to_first_connections[edge.data.data]
                        first_inner_state.add_edge(
                            nodes_renamed_map[node],
                            nodes_renamed_map[node].data
                            if isinstance(node, dace_nodes.AccessNode) and edge.src_conn
                            else edge.src_conn,
                            nodes_renamed_map[dst],
                            second_to_first_connections[dst.data]
                            if isinstance(edge.dst, dace_nodes.AccessNode) and edge.dst_conn
                            else edge.dst_conn,
                            new_memlet,
                        )
        for edge in list(graph.out_edges(access_node)):
            if edge.dst == second_cb:
                graph.remove_edge(edge)

        # Need to remove both references to remove NestedSDFG from graph
        graph.remove_node(second_conditional_block)
        graph.remove_node(second_cb)

        new_arrays = len(first_cb.sdfg.arrays)
        assert new_arrays == total_original_arrays - 1, (
            f"After fusion, expected {total_original_arrays - 1} arrays but found {new_arrays}"
        )

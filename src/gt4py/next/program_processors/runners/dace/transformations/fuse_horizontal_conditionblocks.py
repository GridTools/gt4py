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
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation import helpers as dace_helpers

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class FuseHorizontalConditionBlocks(dace_transformation.SingleStateTransformation):
    """Fuses two conditional blocks that share the same condition variable and are
    not dependent to each other (i.e. the output of one of them is used as input to the other)
    into a single conditional block.
    The motivation for this transformation is to reduce the number of conditional blocks
    which generate if statements in the CPU or GPU code, which can lead to performance improvements.
    Example:
    Before fusion:
    ```
    if __cond:
        __output1 = __arg1 * 2.0
    else:
        __output1 = __arg2 + 3.0
    if __cond:
        __output2 = __arg3 - 1.0
    else:
        __output2 = __arg4 / 4.0
    ```
    After fusion:
    ```
    if __cond:
        __output1 = __arg1 * 2.0
        __output2 = __arg3 - 1.0
    else:
        __output1 = __arg2 + 3.0
        __output2 = __arg4 / 4.0
    ```
    """

    conditional_access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    nsdfg_a = dace_transformation.PatternNode(dace_nodes.NestedSDFG)
    nsdfg_b = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    # The fusion of the two conditional blocks can happen in any order. To avoid any indeterminism distinguish which one is the fused and which one is the extended conditional block which will include the fused one.
    @staticmethod
    def _order_conditional_blocks_based_on_label(
        conditional_block_0: dace.sdfg.state.ConditionalBlock,
        conditional_block_1: dace.sdfg.state.ConditionalBlock,
    ) -> tuple[dace.sdfg.state.ConditionalBlock, dace.sdfg.state.ConditionalBlock]:
        if conditional_block_0.label < conditional_block_1.label:
            return conditional_block_0, conditional_block_1
        else:
            return conditional_block_1, conditional_block_0

    @staticmethod
    def _order_conditional_blocks_based_on_number_of_common_branches_and_label(
        conditional_block_0: dace.sdfg.state.ConditionalBlock,
        conditional_block_1: dace.sdfg.state.ConditionalBlock,
    ) -> tuple[dace.sdfg.state.ConditionalBlock, dace.sdfg.state.ConditionalBlock] | None:
        branches_0 = conditional_block_0.branches
        branches_1 = conditional_block_1.branches

        # We compute a set of the conditions because we don't care about their order since all the
        # conditions should either be based in a scalar boolean (`__cond`) or be None which means
        # the `else` condition.
        branch_0_conditions = {branch[0].as_string if branch[0] else None for branch in branches_0}
        branch_1_conditions = {branch[0].as_string if branch[0] else None for branch in branches_1}

        intersection_of_conditions = branch_0_conditions.intersection(branch_1_conditions)

        # No matching conditions found between the two `ConditionalBlock`s
        if not intersection_of_conditions:
            return None
        # Matching conditions between the two `ConditionalBlock`s is the same as the number of branches in both `ConditionalBlock`s.
        # This means that all branches have matching conditions and thus we can use the labels to distinguish which one is the fused and which one is the extended conditional block.
        elif (
            len(intersection_of_conditions) == len(branch_0_conditions) == len(branch_1_conditions)
        ):
            return FuseHorizontalConditionBlocks._order_conditional_blocks_based_on_label(
                conditional_block_0, conditional_block_1
            )
        # This means that all the branches of `conditional_block_1` match some or all of the conditions of `conditional_block_0` so we can merge `conditional_block_1` into `conditional_block_0`.
        elif len(intersection_of_conditions) == len(branch_1_conditions):
            return conditional_block_0, conditional_block_1
        # This means that all the branches of `conditional_block_0` match some or all of the conditions of `conditional_block_1` so we can merge `conditional_block_0`` into `conditional_block_1`.
        elif len(intersection_of_conditions) == len(branch_0_conditions):
            return conditional_block_1, conditional_block_0
        return None

    @classmethod
    def expressions(cls) -> Any:
        conditionalblock_fusion_parallel_match = dace_graph.OrderedMultiDiConnectorGraph()
        conditionalblock_fusion_parallel_match.add_nedge(
            cls.conditional_access_node, cls.nsdfg_a, dace.Memlet()
        )
        conditionalblock_fusion_parallel_match.add_nedge(
            cls.conditional_access_node, cls.nsdfg_b, dace.Memlet()
        )
        return [conditionalblock_fusion_parallel_match]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        conditional_access_node: dace_nodes.AccessNode = self.conditional_access_node
        conditional_access_node_desc = conditional_access_node.desc(sdfg)
        nsdfg_a = self.nsdfg_a
        nsdfg_b = self.nsdfg_b

        scope_dict = graph.scope_dict()

        # Check that the common access node is a boolean scalar
        if (
            not isinstance(conditional_access_node_desc, dace.data.Scalar)
            or conditional_access_node_desc.dtype != dace.bool_
        ):
            return False

        # Check that both conditional blocks are in the same parent SDFG
        if nsdfg_a.sdfg.parent is not nsdfg_b.sdfg.parent:
            return False

        # Check that the nested SDFGs' names starts with "if_stmt"
        if not (
            nsdfg_a.sdfg.name.startswith("if_stmt") and nsdfg_b.sdfg.name.startswith("if_stmt")
        ):
            return False

        # Make sure that the nested SDFGs contain only one conditional block each
        if nsdfg_a.sdfg.number_of_nodes() != 1 or nsdfg_b.sdfg.number_of_nodes() != 1:
            return False

        if not (
            isinstance(nsdfg_a.sdfg.nodes()[0], dace.sdfg.state.ConditionalBlock)
            and isinstance(nsdfg_b.sdfg.nodes()[0], dace.sdfg.state.ConditionalBlock)
        ):
            return False

        conditional_block_a = nsdfg_a.sdfg.nodes()[0]
        conditional_block_b = nsdfg_b.sdfg.nodes()[0]

        # This transformation accepts only a boolean condition. This means that the conditions of
        # the branches can only be 2 or 1. Either `__cond` and `(not __cond)` or `__cond` and `None`
        # or just `__cond` or `(not __cond)`.
        if len(conditional_block_a.branches) > 2 or len(conditional_block_b.branches) > 2:
            return False

        # Make sure that the branches of both `ConditionalBlock`s have only one state
        # TODO(iomaganaris): In case there are more states we can find a way to add one after the other
        for conditional_block in [conditional_block_a, conditional_block_b]:
            for control_flow_region in conditional_block.sub_regions():
                if len(control_flow_region.nodes()) != 1 or not isinstance(
                    control_flow_region.nodes()[0], dace.sdfg.state.SDFGState
                ):
                    return False

        conditional_block_tuple = (
            self._order_conditional_blocks_based_on_number_of_common_branches_and_label(
                conditional_block_a, conditional_block_b
            )
        )
        if not conditional_block_tuple:
            return False
        extended_conditional_block, fused_conditional_block = conditional_block_tuple

        nested_sdfg_of_extended_conditional_block = (
            extended_conditional_block.sdfg.parent_nsdfg_node
        )
        nested_sdfg_of_fused_conditional_block = fused_conditional_block.sdfg.parent_nsdfg_node

        # Check that the symbol mappings are compatible. If there's a symbol that is in both mappings but mapped to different definitions then we skip fusing the conditional blocks.
        # TODO(iomaganaris): One could also rename the symbols instead of skipping the fusion but for now we keep it simple
        sym_map1 = nested_sdfg_of_extended_conditional_block.symbol_mapping
        sym_map2 = nested_sdfg_of_fused_conditional_block.symbol_mapping
        if any(str(sym_map1[sym]) != str(sym_map2[sym]) for sym in sym_map2 if sym in sym_map1):
            return False

        # Make sure that both conditional blocks are in the same scope
        if (
            scope_dict[nested_sdfg_of_extended_conditional_block]
            != scope_dict[nested_sdfg_of_fused_conditional_block]
        ):
            return False

        # Make sure that both conditional blocks are in a map scope
        if not isinstance(
            scope_dict[nested_sdfg_of_extended_conditional_block], dace.nodes.MapEntry
        ):
            return False

        # Check that there is an edge to the conditional blocks with dst_conn == "__cond"
        cond_edges_first = [
            e
            for e in graph.in_edges(nested_sdfg_of_extended_conditional_block)
            if e.dst_conn and e.dst_conn == "__cond"
        ]
        if len(cond_edges_first) != 1:
            return False
        cond_edges_second = [
            e
            for e in graph.in_edges(nested_sdfg_of_fused_conditional_block)
            if e.dst_conn and e.dst_conn == "__cond"
        ]
        if len(cond_edges_second) != 1:
            return False
        cond_edge_first = cond_edges_first[0]
        cond_edge_second = cond_edges_second[0]
        if cond_edge_first.data.is_empty() or cond_edge_second.data.is_empty():
            return False
        if not all(
            cond_edge.src is conditional_access_node
            for cond_edge in [cond_edge_first, cond_edge_second]
        ):
            return False

        # Need to check also that first and second nested SDFGs are not reachable from each other
        if gtx_transformations.utils.is_reachable(
            start=nested_sdfg_of_extended_conditional_block,
            target=nested_sdfg_of_fused_conditional_block,
            state=graph,
        ) or gtx_transformations.utils.is_reachable(
            start=nested_sdfg_of_fused_conditional_block,
            target=nested_sdfg_of_extended_conditional_block,
            state=graph,
        ):
            return False

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        conditional_access_node: dace_nodes.AccessNode = self.conditional_access_node
        nsdfg_a = self.nsdfg_a
        nsdfg_b = self.nsdfg_b
        assert nsdfg_a.sdfg.number_of_nodes() == 1 and nsdfg_b.sdfg.number_of_nodes() == 1
        conditional_block_a = nsdfg_a.sdfg.nodes()[0]
        conditional_block_b = nsdfg_b.sdfg.nodes()[0]
        conditional_block_tuple = (
            self._order_conditional_blocks_based_on_number_of_common_branches_and_label(
                conditional_block_a, conditional_block_b
            )
        )
        assert conditional_block_tuple is not None
        extended_conditional_block, fused_conditional_block = conditional_block_tuple

        nested_sdfg_of_extended_conditional_block = (
            extended_conditional_block.sdfg.parent_nsdfg_node
        )
        nested_sdfg_of_fused_conditional_block = fused_conditional_block.sdfg.parent_nsdfg_node

        # Copy missing symbols from second conditional block to the first one.
        #  For the symbols that are already in `nested_sdfg_of_extended_conditional_block.symbol_mapping` we know
        #  that the definition matches. Thus there is no need to perform symbol
        #  renaming.
        missing_symbols = {
            sym2: val2
            for sym2, val2 in nested_sdfg_of_fused_conditional_block.symbol_mapping.items()
            if sym2 not in nested_sdfg_of_extended_conditional_block.symbol_mapping
        }
        for missing_symb, symb_def in missing_symbols.items():
            nested_sdfg_of_extended_conditional_block.symbol_mapping[missing_symb] = symb_def
            nested_sdfg_of_extended_conditional_block.sdfg.add_symbol(
                missing_symb,
                nested_sdfg_of_fused_conditional_block.sdfg.symbols[missing_symb],
                find_new_name=False,
            )

        # Store number of original arrays to check later that all the necessary arrays have been moved
        total_original_arrays = len(extended_conditional_block.sdfg.arrays) + len(
            fused_conditional_block.sdfg.arrays
        )

        # Store the new names for the arrays of the second conditional block (transients and globals) to avoid name clashes and add their data descriptors
        # to the first conditional block SDFG. We don't have to add `__cond` because we know it's the same for both conditional blocks.
        # TODO(iomaganaris): Remove inputs to the conditional block that come from the same AccessNodes (same data)
        second_arrays_rename_map: dict[str, str] = {}
        for data_name, data_desc in fused_conditional_block.sdfg.arrays.items():
            if data_name == "__cond":
                continue
            new_data_name = gtx_transformations.utils.unique_name(data_name) + "_from_cb_fusion"
            data_desc_renamed = copy.deepcopy(data_desc)
            second_arrays_rename_map[data_name] = (
                nested_sdfg_of_extended_conditional_block.sdfg.add_datadesc(
                    new_data_name, data_desc_renamed, find_new_name=True
                )
            )

        # Move the connectors from the second conditional block to the first
        # TODO(iomaganaris): Here we copy empty memlets used for scheduling as well. This means that the first conditional blocks inherits the scheduling of the second one as well. Maybe that's not good in some cases to hide latency but for now we keep it as it is
        for edge in graph.in_edges(nested_sdfg_of_fused_conditional_block):
            if edge.dst_conn == "__cond":
                continue
            nested_sdfg_of_extended_conditional_block.add_in_connector(
                second_arrays_rename_map[edge.dst_conn]
            )
            dace_helpers.redirect_edge(
                state=graph,
                edge=edge,
                new_dst_conn=second_arrays_rename_map[edge.dst_conn],
                new_dst=nested_sdfg_of_extended_conditional_block,
            )
        for edge in graph.out_edges(nested_sdfg_of_fused_conditional_block):
            nested_sdfg_of_extended_conditional_block.add_out_connector(
                second_arrays_rename_map[edge.src_conn]
            )
            dace_helpers.redirect_edge(
                state=graph,
                edge=edge,
                new_src_conn=second_arrays_rename_map[edge.src_conn],
                new_src=nested_sdfg_of_extended_conditional_block,
            )

        def _find_corresponding_branch_in_fused(
            extended_branch: tuple[
                Optional[dace.sdfg.state.CodeBlock], dace.sdfg.state.ControlFlowRegion
            ],
            fused_conditional_block: dace.sdfg.state.ConditionalBlock,
        ) -> dace.sdfg.state.ControlFlowRegion | None:
            extended_branch_condition = extended_branch[0].as_string if extended_branch[0] else None
            for branch in fused_conditional_block.branches:
                branch_condition = branch[0].as_string if branch[0] else None
                if branch_condition == extended_branch_condition:
                    return branch
            return None

        # Copy first the nodes from the second conditional block to the first
        # Create a dictionary that maps the original nodes in the second conditional
        # block to the new nodes in the first conditional block to be able to properly connect the edges later
        nodes_renamed_map: dict[dace_nodes.Node, dace_nodes.Node] = {}
        for branch in extended_conditional_block.branches:
            corresponding_branch_in_second = _find_corresponding_branch_in_fused(
                branch, fused_conditional_block
            )
            if not corresponding_branch_in_second:
                continue
            corresponding_state_in_second = corresponding_branch_in_second[1].nodes()[0]
            first_inner_state = branch[1].nodes()[0]
            # Save edges of second conditional block to a state to be able to delete the nodes from the second conditional block
            edges_to_copy = list(corresponding_state_in_second.edges())
            nodes_to_move = list(corresponding_state_in_second.nodes())
            for node in nodes_to_move:
                new_node = node
                if isinstance(node, dace_nodes.AccessNode):
                    new_data_name = second_arrays_rename_map[node.data]
                    new_node = dace_nodes.AccessNode(new_data_name)
                nodes_renamed_map[node] = new_node
                # Remove the original node from the second conditional block to avoid any potential issues
                # with the nodes coexisting in two states
                corresponding_state_in_second.remove_node(node)
                first_inner_state.add_node(new_node)

            for edge_to_copy in edges_to_copy:
                new_edge = first_inner_state.add_edge(
                    nodes_renamed_map[edge_to_copy.src],
                    edge_to_copy.src_conn,
                    nodes_renamed_map[edge_to_copy.dst],
                    edge_to_copy.dst_conn,
                    edge_to_copy.data,
                )
                if not new_edge.data.is_empty():
                    new_edge.data.data = second_arrays_rename_map[new_edge.data.data]

        for edge in list(graph.out_edges(conditional_access_node)):
            if edge.dst == nested_sdfg_of_fused_conditional_block:
                graph.remove_edge(edge)

        # TODO(iomaganaris): Atm need to remove both references to remove NestedSDFG from graph
        #  fused_conditional_block is inside the SDFG of NestedSDFG nested_sdfg_of_fused_conditional_block and removing only
        #  one of them keeps a reference to the other one so none is properly deleted from the SDFG.
        #  For now remove both but maybe this can be improved in the future.
        graph.remove_node(fused_conditional_block)
        graph.remove_node(nested_sdfg_of_fused_conditional_block)

        new_arrays = len(nested_sdfg_of_extended_conditional_block.sdfg.arrays)
        assert new_arrays == total_original_arrays - 1, (
            f"After fusion, expected {total_original_arrays - 1} arrays but found {new_arrays}"
        )

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Union

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
    first_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)
    second_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    @classmethod
    def expressions(cls) -> Any:
        conditionalblock_fusion_parallel_match = dace_graph.OrderedMultiDiConnectorGraph()
        conditionalblock_fusion_parallel_match.add_nedge(
            cls.conditional_access_node, cls.first_conditional_block, dace.Memlet()
        )
        conditionalblock_fusion_parallel_match.add_nedge(
            cls.conditional_access_node, cls.second_conditional_block, dace.Memlet()
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
        first_cb: dace_nodes.NestedSDFG = self.first_conditional_block
        second_cb: dace_nodes.NestedSDFG = self.second_conditional_block
        scope_dict = graph.scope_dict()

        # Check that the common access node is a boolean scalar
        if (
            not isinstance(conditional_access_node_desc, dace.data.Scalar)
            or conditional_access_node_desc.dtype != dace.bool_
        ):
            return False

        # Check that both conditional blocks are in the same parent SDFG
        if first_cb.sdfg.parent != second_cb.sdfg.parent:
            return False

        # Check that the nested SDFGs' names starts with "if_stmt"
        if not (
            first_cb.sdfg.name.startswith("if_stmt") and second_cb.sdfg.name.startswith("if_stmt")
        ):
            return False

        # Make sure that the conditional blocks contain only one conditional block each
        if first_cb.sdfg.number_of_nodes() != 1 or second_cb.sdfg.number_of_nodes() != 1:
            return False

        # Check that the symbol mappings are compatible. If there's a symbol that is in both mappings but mapped to different definitions then we skip fusing the conditional blocks.
        # TODO(iomaganaris): One could also rename the symbols instead of skipping the fusion but for now we keep it simple
        sym_map1 = first_cb.symbol_mapping
        sym_map2 = second_cb.symbol_mapping
        if any(str(sym_map1[sym]) != str(sym_map2[sym]) for sym in sym_map2 if sym in sym_map1):
            return False

        # Get the actual conditional blocks
        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))
        # TODO(iomaganaris): For now the branches of the conditional blocks should have only one state. If there's a change in the future and they have more than one state the below checks need to be modified
        if not (
            isinstance(first_conditional_block, dace.sdfg.state.ConditionalBlock)
            and len(first_conditional_block.sub_regions()) == 2
            and isinstance(second_conditional_block, dace.sdfg.state.ConditionalBlock)
            and len(second_conditional_block.sub_regions()) == 2
        ):
            return False
        first_conditional_block_state_names = [
            state.name for state in first_conditional_block.all_states()
        ]
        second_conditional_block_state_names = [
            state.name for state in second_conditional_block.all_states()
        ]
        if not (
            any("true_branch" in name for name in first_conditional_block_state_names)
            and any("false_branch" in name for name in first_conditional_block_state_names)
            and any("true_branch" in name for name in second_conditional_block_state_names)
            and any("false_branch" in name for name in second_conditional_block_state_names)
        ):
            return False

        # Make sure that both conditional blocks are in the same scope
        if scope_dict[first_cb] != scope_dict[second_cb]:
            return False

        # Make sure that both conditional blocks are in a map scope
        if not isinstance(scope_dict[first_cb], dace.nodes.MapEntry):
            return False

        # Check that there is an edge to the conditional blocks with dst_conn == "__cond"
        cond_edges_first = [
            e for e in graph.in_edges(first_cb) if e.dst_conn and e.dst_conn == "__cond"
        ]
        if len(cond_edges_first) != 1:
            return False
        cond_edges_second = [
            e for e in graph.in_edges(second_cb) if e.dst_conn and e.dst_conn == "__cond"
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
        conditional_access_node: dace_nodes.AccessNode = self.conditional_access_node
        first_cb: dace_nodes.NestedSDFG = self.first_conditional_block
        second_cb: dace_nodes.NestedSDFG = self.second_conditional_block

        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))

        # Copy missing symbols from second conditional block to the first one.
        #  For the symbols that are already in `first_cb.symbol_mapping` we know
        #  that the definition matches. Thus there is no need to perform symbol
        #  renaming.
        missing_symbols = {
            sym2: val2
            for sym2, val2 in second_cb.symbol_mapping.items()
            if sym2 not in first_cb.symbol_mapping
        }
        for missing_symb, symb_def in missing_symbols.items():
            first_cb.symbol_mapping[missing_symb] = symb_def
            first_cb.sdfg.add_symbol(
                missing_symb, second_cb.sdfg.symbols[missing_symb], find_new_name=False
            )

        # Store number of original arrays to check later that all the necessary arrays have been moved
        total_original_arrays = len(first_conditional_block.sdfg.arrays) + len(
            second_conditional_block.sdfg.arrays
        )

        # Store the new names for the arrays of the second conditional block (transients and globals) to avoid name clashes and add their data descriptors
        # to the first conditional block SDFG. We don't have to add `__cond` because we know it's the same for both conditional blocks.
        # TODO(iomaganaris): Remove inputs to the conditional block that come from the same AccessNodes (same data)
        second_arrays_rename_map: dict[str, str] = {}
        for data_name, data_desc in second_conditional_block.sdfg.arrays.items():
            if data_name == "__cond":
                continue
            new_data_name = gtx_transformations.utils.unique_name(data_name) + "_from_cb_fusion"
            data_desc_renamed = copy.deepcopy(data_desc)
            second_arrays_rename_map[data_name] = first_cb.sdfg.add_datadesc(
                new_data_name, data_desc_renamed, find_new_name=True
            )

        second_conditional_states = list(second_conditional_block.all_states())

        # Move the connectors from the second conditional block to the first
        # TODO(iomaganaris): Here we copy empty memlets used for scheduling as well. This means that the first conditional blocks inherits the scheduling of the second one as well. Maybe that's not good in some cases to hide latency but for now we keep it as it is
        for edge in graph.in_edges(second_cb):
            if edge.dst_conn == "__cond":
                continue
            first_cb.add_in_connector(second_arrays_rename_map[edge.dst_conn])
            dace_helpers.redirect_edge(
                state=graph,
                edge=edge,
                new_dst_conn=second_arrays_rename_map[edge.dst_conn],
                new_dst=first_cb,
            )
        for edge in graph.out_edges(second_cb):
            first_cb.add_out_connector(second_arrays_rename_map[edge.src_conn])
            dace_helpers.redirect_edge(
                state=graph,
                edge=edge,
                new_src_conn=second_arrays_rename_map[edge.src_conn],
                new_src=first_cb,
            )

        def _find_corresponding_state_in_second(
            inner_state: dace.SDFGState,
        ) -> dace.SDFGState:
            is_true_branch = "true_branch" in inner_state.name
            branch_type = "true_branch" if is_true_branch else "false_branch"
            return next(state for state in second_conditional_states if branch_type in state.name)

        # Copy first the nodes from the second conditional block to the first
        # Create a dictionary that maps the original nodes in the second conditional
        # block to the new nodes in the first conditional block to be able to properly connect the edges later
        nodes_renamed_map: dict[dace_nodes.Node, dace_nodes.Node] = {}
        for first_inner_state in first_conditional_block.all_states():
            corresponding_state_in_second = _find_corresponding_state_in_second(first_inner_state)
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
            if edge.dst == second_cb:
                graph.remove_edge(edge)

        # TODO(iomaganaris): Atm need to remove both references to remove NestedSDFG from graph
        #  second_conditional_block is inside the SDFG of NestedSDFG second_cb and removing only
        #  one of them keeps a reference to the other one so none is properly deleted from the SDFG.
        #  For now remove both but maybe this can be improved in the future.
        graph.remove_node(second_conditional_block)
        graph.remove_node(second_cb)

        new_arrays = len(first_cb.sdfg.arrays)
        assert new_arrays == total_original_arrays - 1, (
            f"After fusion, expected {total_original_arrays - 1} arrays but found {new_arrays}"
        )

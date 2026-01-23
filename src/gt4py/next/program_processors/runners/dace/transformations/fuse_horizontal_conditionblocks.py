import copy
import warnings
from typing import Any, Callable, Mapping, Optional, TypeAlias, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)

from dace.sdfg import nodes as dace_nodes, graph as dace_graph
from dace.transformation import helpers as dace_helpers
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

@dace_properties.make_properties
class FuseHorizontalConditionBlocks(dace_transformation.SingleStateTransformation):
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    first_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)
    second_conditional_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = dace_graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nedge(cls.access_node, cls.first_conditional_block, dace.Memlet())
        map_fusion_parallel_match.add_nedge(cls.access_node, cls.second_conditional_block, dace.Memlet())
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

        if first_cb.sdfg.parent != second_cb.sdfg.parent:
            return False
        # Check that the nested SDFGs' names starts with "if_stmt"
        if not (first_cb.sdfg.name.startswith("if_stmt") and second_cb.sdfg.name.startswith("if_stmt")):
            return False

        # Check that the common access node is a boolean scalar
        if not isinstance(access_node_desc, dace.data.Scalar) or access_node_desc.dtype != dace.bool_:
            return False

        if len(first_cb.sdfg.nodes()) > 1 or len(second_cb.sdfg.nodes()) > 1:
            return False

        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))
        if not (isinstance(first_conditional_block, dace.sdfg.state.ConditionalBlock) and isinstance(second_conditional_block, dace.sdfg.state.ConditionalBlock)):
            return False

        if scope_dict[first_cb] != scope_dict[second_cb]:
            return False

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
        if cond_edge_first.src != cond_edge_second.src and (cond_edge_first.src != access_node or cond_edge_second.src != access_node):
            return False

        print(f"Found valid conditional blocks: {first_cb} and {second_cb}", flush=True)
        # breakpoint()

        # TODO(iomaganaris): Need to check also that first and second nested SDFGs are not reachable from each other

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        access_node: dace_nodes.AccessNode = self.access_node
        first_cb: dace.sdfg.state.ConditionalBlock = self.first_conditional_block
        second_cb: dace.sdfg.state.ConditionalBlock = self.second_conditional_block

        first_conditional_block = next(iter(first_cb.sdfg.nodes()))
        second_conditional_block = next(iter(second_cb.sdfg.nodes()))

        second_conditional_states = list(second_conditional_block.all_states())

        for first_inner_state in first_conditional_block.all_states():
            first_inner_state_name = first_inner_state.name
            corresponding_state_in_second = None
            for state in second_conditional_states:
                if state.name == first_inner_state_name:
                    corresponding_state_in_second = state
                    break
            if corresponding_state_in_second is None:
                raise RuntimeError(f"Could not find corresponding state in second conditional block for state {first_inner_state_name}")
            nodes_to_move = list(corresponding_state_in_second.nodes())
            in_connectors_to_move = {k: v for k, v in second_cb.in_connectors.items() if k != "__cond"}
            out_connectors_to_move = second_cb.out_connectors
            breakpoint()

        
        # print(f"Fused conditional blocks into: {new_nested_sdfg}", flush=True)
        # breakpoint()
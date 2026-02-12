# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes
from dace.transformation import helpers as dace_helpers
from dace.transformation.passes import analysis as dace_analysis


@dace_properties.make_properties
class RemoveScalarCopies(dace_transformation.SingleStateTransformation):
    """Removes copies between two scalar transient variables.
    Exaxmple:
     ___
    /   \
    | A |
    \___/
      |
     \/
     ___
    /   \
    | B |
    \___/
    is transformed to
     ___
    /   \
    | A |
    \___/
    and all uses of B are replaced with A.
    """

    first_access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    second_access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    assume_single_use_data = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="Always assume that `self.access_node` is single use data. Only useful if used through `SplitAccessNode.apply_to()`.",
    )

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        assume_single_use_data: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = single_use_data
        if assume_single_use_data is not None:
            self.assume_single_use_data = assume_single_use_data

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.first_access_node, cls.second_access_node)]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_node: dace_nodes.AccessNode = self.first_access_node
        first_node_desc = first_node.desc(sdfg)
        second_node: dace_nodes.AccessNode = self.second_access_node
        second_node_desc = second_node.desc(sdfg)

        scope_dict = graph.scope_dict()

        # Make sure that both access nodes are in the same scope
        if scope_dict[first_node] != scope_dict[second_node]:
            return False

        # Make sure that both access nodes are transients
        if not (first_node_desc.transient and second_node_desc.transient):
            return False

        edges = list(graph.edges_between(first_node, second_node))
        if len(edges) != 1:
            return False
        edge = edges[0]

        # Check that the second access node has only one incoming edge, which is the one from the first access node.
        if graph.in_degree(second_node) != 1:
            return False

        # Check if the edge transfers only one element
        if edge.data.num_elements() != 1:
            return False
        if edge.data.dynamic:
            return False

        # Check that all the outgoing edges of the second access node transfer only one element
        for out_edges in graph.out_edges(second_node):
            if out_edges.data.num_elements() != 1:
                return False

        # Make sure that the data descriptors of both access nodes are scalars
        # TODO(iomaganaris): We could extend this transfromation to handle AccessNodes that are arrays with 1 element as well
        if not isinstance(first_node_desc, dace.data.Scalar) or not isinstance(
            second_node_desc, dace.data.Scalar
        ):
            return False

        # Make sure that both access nodes are not views
        if isinstance(first_node_desc, dace.data.View) or isinstance(
            second_node_desc, dace.data.View
        ):
            return False

        # Make sure that the second AccessNode data is single use data since we're only going to remove that one
        if self.assume_single_use_data:
            single_use_data = {sdfg: {second_node.data}}
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if second_node.data not in single_use_data[sdfg]:
            return False

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        first_node: dace_nodes.AccessNode = self.first_access_node
        second_node: dace_nodes.AccessNode = self.second_access_node

        # Redirect all outcoming edges of the second access node to the first
        for edge in list(graph.out_edges(second_node)):
            dace_helpers.redirect_edge(
                state=graph,
                edge=edge,
                new_src=first_node,
                new_data=first_node.data if edge.data.data == second_node.data else edge.data.data,
            )

        # Remove the second access node
        graph.remove_node(second_node)

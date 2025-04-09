# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Optional

import dace
from dace import (
    dtypes as dace_dtypes,
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes


@dace_properties.make_properties
class MoveDataflowIntoIfBody(dace_transformation.SingleStateTransformation):
    """The transformation moves dataflow into the if branches.

    Essentially transforms code from this
    ```python
    __arg1 = foo(...)
    __arg2 = bar(...)
    if __cond:
        __output = __arg1
    else:
        __output = __arg2
    ```
    into this
    ```python
    if __cond:
        __output = foo(...)
    else:
        __output = bar(...)
    ```

    Note:
        The current implementation only handles the case that the if is inside a Map.
        In the context of GTIR this is not so much a limitation, but it makes the
        implementation a bit simpler. However, it also makes matching a little bit
        more complex.
    """

    if_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.if_block)]

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        scope_dict = graph.scope_dict()
        if_block: dace_nodes.NestedSDFG = self.if_block

        # We must be inside a top level Map, thus our parent must be a Map that
        #  is located in global scope.
        enclosing_map = scope_dict[if_block]
        if not isinstance(enclosing_map, dace_nodes.MapEntry):
            return False
        if scope_dict[enclosing_map] is not None:
            return False

        # Now test if the if block is really what we expect it to be, a if statement.
        if_block_spec = self._is_valid_if_block(if_block)
        if if_block_spec is None:
            return False
        cond_name, b1_name, b2_name = if_block_spec

        # There is an interesting point here. To simplify `apply()` we assumed that
        #  `if_block` is inside a Map scope. The problem is if there is another
        #  `if`-block upstream of the current one. If we would now apply the
        #  transformation to `if_block` we could not handle the upstream one.
        #  Thus we will now check if this is the case and then reject the application
        #  to `if_block` we really here that the transformation is run inside a
        #  loop.
        branch_dependencies: list[set[dace_nodes.Node]] = []
        for conn_name in [b1_name, b2_name]:
            branch_dependency = self._find_upstream_nodes(
                start=if_block,
                state=graph,
                start_connector=conn_name,
                limit_node=enclosing_map,
            )
            # Check if there are no upstream blocks that have to be handled first.
            if any(
                self._check_if_there_is_something_to_relocate(
                    if_block=upstream_if_block,
                    state=graph,
                    enclosing_map=enclosing_map,
                )
                for upstream_if_block in branch_dependency
                if isinstance(upstream_if_block, dace_nodes.NestedSDFG)
            ):
                return False
            branch_dependencies.append(branch_dependency)

        branch_dependencies.insert(
            0,
            self._find_upstream_nodes(
                start=if_block,
                state=graph,
                start_connector=cond_name,
                limit_node=enclosing_map,
            ),
        )

        if not self._check_if_there_is_something_to_relocate(
            if_block=if_block,
            state=graph,
            enclosing_map=enclosing_map,
            branch_dependencies=branch_dependencies,
        ):
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        if_block: dace_nodes.NestedSDFG = self.if_block
        if_block_spec = self._is_valid_if_block(if_block)
        assert if_block_spec is not None
        enclosing_map = graph.scope_dict()[if_block]

        branch_dependencies: list[set[dace_nodes.Node]] = [
            self._find_upstream_nodes(
                start=if_block,
                state=graph,
                start_connector=conn_name,
                limit_node=enclosing_map,
            )
            for conn_name in if_block_spec
        ]

        # Now compute which nodes could actually be moved into the branches.
        move_into_b1, move_into_b2 = self._partition_dependencies(
            state=graph,
            if_block=if_block,
            cond_dependency=branch_dependencies[0],
            b1_dependency=branch_dependencies[1],
            b2_dependency=branch_dependencies[2],
        )

        # Now replicate the data flow inside the branches.
        for nodes_to_move, connector in [
            (move_into_b1, if_block_spec[1]),
            (move_into_b2, if_block_spec[2]),
        ]:
            self._replicate_dataflow_into_branche(
                state=graph,
                sdfg=sdfg,
                if_block=if_block,
                enclosing_map=enclosing_map,
                nodes_to_move=nodes_to_move,
                connector=connector,
            )

        self._update_symbol_mapping(if_block)

        # Now remove the nodes that have been relocated from the SDFG and also
        #  clean up the registry.
        for node_to_remove in [*move_into_b1, *move_into_b2]:
            if isinstance(node_to_remove, dace_nodes.AccessNode):
                assert node_to_remove.desc(sdfg).transient
                sdfg.remove_data(node_to_remove.data, validate=False)
            graph.remove_node(node_to_remove)

    def _replicate_dataflow_into_branche(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
        nodes_to_move: set[dace_nodes.Node],
        connector: str,
    ) -> None:
        """ """

        # No work nothing to do.
        if len(nodes_to_move) == 0:
            return

        inner_sdfg: dace.SDFG = if_block.sdfg
        branch_state, connector_node = self._find_branch_for(
            if_block=if_block,
            connector=connector,
        )

        # There might be AccessNodes inside `nodes_to_move`, we now have to make sure
        #  that they are present inside the nested ones. By our base assumption they
        #  are transients, because they are only used in one place
        for node in nodes_to_move:
            if not isinstance(node, dace_nodes.AccessNode):
                continue
            assert node.data not in inner_sdfg.arrays
            assert sdfg.arrays[node.data].transient
            inner_sdfg.add_datadesc(node.data, sdfg.arrays[node.data].clone())

        # Replicate the nodes. Also make a mapping that allows to map the old ones
        #  to the new ones.
        new_nodes: dict[dace_nodes.Node, dace_nodes.Node] = {
            old_node: copy.deepcopy(old_node) for old_node in nodes_to_move
        }
        branch_state.add_nodes_from(new_nodes.values())

        # Now add the edges between the edges that have been replicated inside the
        #  branch state, these are the outgoing edges. The data dependencies of the
        #  nodes that were not relocated are still missing.
        for node in nodes_to_move:
            for oedge in state.out_edges(node):
                if oedge.dst is if_block:
                    assert (
                        oedge.dst_conn == connector
                    ), f"Expected connector '{connector}' but found '{oedge.dst_conn}'."
                    # TODO(phimuell): Make subsets complete.
                    branch_state.add_edge(
                        new_nodes[oedge.src],
                        oedge.src_conn,
                        connector_node,
                        None,
                        dace.Memlet.from_memlet(oedge.data),
                    )
                else:
                    assert (
                        oedge.dst in nodes_to_move
                    ), f"Expected that node '{oedge.dst}' was also moved but it is not."
                    branch_state.add_edge(
                        new_nodes[oedge.src],
                        oedge.src_conn,
                        new_nodes[oedge.dst],
                        oedge.dst_conn,
                        dace.Memlet.from_memlet(oedge.data),
                    )

        # Now we have to satisfy the data dependencies, i.e. forward all nodes that
        #  could not have been moved inside `if_block` but are still needed to compute
        #  the final result. We find them by scanning the input edges of the nodes
        #  that have been relocated.
        for node in nodes_to_move:
            for iedge in state.in_edges(node):
                if iedge.src in nodes_to_move:
                    # Inner data dependency, there is nothing to do and the edge was
                    #  created above.
                    continue
                if iedge.data.is_empty():
                    # These kind of Memlet was only needed to keep `iedge.dst` inside
                    #  the scope, this scope has now gone, so the Memlet is no longer
                    #  needed: We are done!
                    assert iedge.src is enclosing_map
                    continue

                # Now we have to figuring out where the data is coming from. This is
                #  `iedge.src` except it is the `enclosing_map`.
                if iedge.src is enclosing_map:
                    memlet_path = state.memlet_path(iedge)
                    assert (
                        memlet_path[0].dst is enclosing_map
                    ), "Expected that the AccessNode is directly adjacent to the Map."
                    outer_data = memlet_path[0].src
                else:
                    outer_data = iedge.src
                assert isinstance(outer_data, dace_nodes.AccessNode)

                # If the data is not yet available in the inner SDFG made
                #  patch it through.
                if outer_data.data not in inner_sdfg.arrays:
                    inner_desc = sdfg.arrays[outer_data.data].clone()
                    inner_desc.transient = False
                    inner_sdfg.add_datadesc(outer_data.data, inner_desc)
                    state.add_edge(
                        iedge.src,
                        iedge.src_conn,
                        if_block,
                        outer_data.data,
                        dace.Memlet(
                            data=outer_data.data, subset=dace_sbs.Range.from_array(inner_desc)
                        ),
                    )
                    if_block.add_in_connector(outer_data.data)
                    # TODO(phimuell): Do we have modify the subset of what is read outside now
                    #   or is it fine if we wait until Memlet propagation runs the next time?

                if outer_data not in new_nodes:
                    assert all(
                        outer_data.data != mapped_nodes.data
                        for mapped_nodes in new_nodes.values()
                        if isinstance(mapped_nodes, dace_nodes.AccessNode)
                    )
                    assert outer_data.data in inner_sdfg.arrays
                    assert not inner_sdfg.arrays[outer_data.data].transient
                    new_nodes[outer_data] = branch_state.add_access(outer_data.data)

                # Now create the edge in the inner state.
                branch_state.add_edge(
                    new_nodes[outer_data],
                    None,
                    new_nodes[iedge.dst],
                    iedge.dst_conn,
                    copy.deepcopy(iedge.data),
                )

        # The old connector name is no longer valid.
        inner_sdfg.arrays[connector].transient = True
        if_block.remove_in_connector(connector)

    def _update_symbol_mapping(
        self,
        if_block: dace_nodes.NestedSDFG,
    ) -> None:
        """Updates the symbol mapping of the nested SDFG.

        The function assumes that the symbols that are missing in the nested SDFG
        are available in the parent SDFG.
        """
        symbol_mapping = if_block.symbol_mapping
        missing_symbols = [ms for ms in if_block.sdfg.free_symbols if ms not in symbol_mapping]
        symbol_mapping.update({s: s for s in missing_symbols})
        if_block.symbol_mapping = symbol_mapping  # Performs conversion.

        # Add new global symbols to nested SDFG.
        #  The code is based on `SDFGState.add_nested_sdfg()`.
        for sym, symval in if_block.symbol_mapping.items():
            if sym not in if_block.sdfg.symbols:
                if_block.sdfg.add_symbol(
                    sym,
                    dace.codegen.tools.type_inference.infer_expr_type(symval, if_block.sdfg.symbols)
                    or dace_dtypes.typeclass(int),
                )

    def _find_branch_for(
        self,
        if_block: dace_nodes.NestedSDFG,
        connector: str,
    ) -> tuple[dace.SDFGState, dace_nodes.AccessNode]:
        inner_sdfg: dace.SDFG = if_block.sdfg
        conditional_block: dace.sdfg.state.ConditionalBlock = next(iter(inner_sdfg.nodes()))

        # This actually finds the first state (in a unspecific order that has
        #  an AccessNode that refers to `connector`, this is good enough because
        #  of the matching, but is not super robust.
        for inner_state in conditional_block.all_states():
            connector_nodes: list[dace_nodes.AccessNode] = [
                dnode for dnode in inner_state.data_nodes() if dnode.data == connector
            ]
            if len(connector_nodes) == 0:
                continue
            break
        else:
            raise ValueError(f"Did not found a branch associated to '{connector}'.")

        assert isinstance(inner_state, dace.SDFGState)
        assert inner_state.in_degree(connector_nodes[0]) == 0
        assert inner_state.out_degree(connector_nodes[0]) > 0
        return inner_state, connector_nodes[0]

    def _check_if_there_is_something_to_relocate(
        self,
        if_block: dace_nodes.NestedSDFG,
        state: dace.SDFGState,
        enclosing_map: dace_nodes.MapEntry,
        branch_dependencies: Optional[list[set[dace_nodes.Node]]] = None,
    ) -> bool:
        """Check if the transformation would apply to `if_block`.

        The function does not perform checks on scope. It simply looks if there is
        something that could be moved inside. It also does not check the upstream
        of `if_block`.
        """

        if_block_spec = self._is_valid_if_block(if_block)
        if if_block_spec is None:
            return False

        # Now compute the dependencies for each branch, if not done externally.
        if branch_dependencies is None:
            branch_dependencies = [
                self._find_upstream_nodes(
                    start=if_block,
                    state=state,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in if_block_spec
            ]

        # Now compute which nodes could actually be moved into the branches.
        move_into_b1, move_into_b2 = self._partition_dependencies(
            state=state,
            if_block=if_block,
            cond_dependency=branch_dependencies[0],
            b1_dependency=branch_dependencies[1],
            b2_dependency=branch_dependencies[2],
        )

        # The transformation is applicable if we can move something can be moved.
        return (len(move_into_b1) != 0) or (len(move_into_b2) != 0)

    def _partition_dependencies(
        self,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        cond_dependency: set[dace_nodes.Node],
        b1_dependency: set[dace_nodes.Node],
        b2_dependency: set[dace_nodes.Node],
    ) -> tuple[set[dace_nodes.Node], set[dace_nodes.Node]]:
        """Partition the dependencies.

        Returns the nodes that can be moved into the two branches.
        """

        # Now filter what can not be inside the two sets, i.e. we know that for
        #  sure the content of the condition computation can not be part of them.
        move_into_b1: set[dace_nodes.Node] = b1_dependency.difference(cond_dependency)
        move_into_b2: set[dace_nodes.Node] = b2_dependency.difference(cond_dependency)

        # Then nothing that is in both sets can exclusively be relocated into one
        #  branch, so filter that out.
        shared_nodes = move_into_b1.intersection(move_into_b2)
        move_into_b1.difference_update(shared_nodes)
        move_into_b2.difference_update(shared_nodes)

        # Furthermore, a node that generates output that it is used somewhere else,
        #  i.e. outside the branch, can not be moved inside it. This is an iterative
        #  process.
        def filter_nodes(branch_nodes: set[dace_nodes.Node]) -> set[dace_nodes.Node]:
            # Notes that have an outgoing connection to `if_block` should not be
            #  should not be removed, although it is not inside `branch_nodes`.
            #  So we add it now and remove it later. Furthermore, we also ignore
            #  it in the scanning.
            branch_nodes.add(if_block)
            has_been_updated = True
            while has_been_updated:
                has_been_updated = False
                for node in list(branch_nodes):
                    if node is if_block:
                        continue
                    if any(oedge.dst not in branch_nodes for oedge in state.out_edges(node)):
                        branch_nodes.remove(node)
                        has_been_updated = True
            assert if_block in branch_nodes
            branch_nodes.remove(if_block)
            return branch_nodes

        return filter_nodes(move_into_b1), filter_nodes(move_into_b2)

    def _is_valid_if_block(
        self,
        if_block: dace_nodes.NestedSDFG,
    ) -> Optional[tuple[str, str, str]]:
        """Checks if `if_block` is a valid block that the transformation operates on.

        If the block is not valid then return `None`. If it is valid return a
        tuple with three strings. The first string is name of the connector that
        goes to the condition. The last two are the connector names of the two
        branches.
        """
        # TODO(phimuell): Generalize the function.

        # TODO(phimuell): These names should be inferred.
        b1_name = "__arg1"
        b2_name = "__arg2"
        cond_name = "__cond"
        out_name = "__output"

        # There shall only be one output and three inputs with given names.
        if if_block.out_connectors.keys() != {out_name}:
            return None
        if if_block.in_connectors.keys() != {b1_name, b2_name, cond_name}:
            return None

        # We require that the nested SDFG contains a single node, which is a
        #  `ConditionalBlock` containing two branches.
        #  TODO(phimuell): Check if the connetors are really access nodes and
        #   if the condition is really used.
        inner_sdfg: dace.SDFG = if_block.sdfg
        if inner_sdfg.number_of_nodes() != 1:
            return None
        inner_if_block = next(iter(inner_sdfg.nodes()))

        if not isinstance(inner_if_block, dace.sdfg.state.ConditionalBlock):
            return None
        if len(inner_if_block.branches) != 2:
            return None

        # See note in `_find_branch_for()`
        reference_count: dict[str, int] = {b1_name: 0, b2_name: 0}
        for inner_state in inner_if_block.all_states():
            assert isinstance(inner_state, dace.SDFGState)
            found_output_node = False
            for dnode in inner_state.data_nodes():
                if dnode.data in reference_count:
                    reference_count[dnode.data] += 1
                    exp_in_deg, exp_out_deg = 0, 1
                elif dnode.data == out_name:
                    exp_in_deg, exp_out_deg = 1, 0
                    found_output_node = True
                else:
                    return None
                if inner_state.in_degree(dnode) != exp_in_deg:
                    return None
                if inner_state.out_degree(dnode) != exp_out_deg:
                    return None
            if not found_output_node:
                return None
        if any(count != 1 for count in reference_count.values()):
            return None

        return (cond_name, b1_name, b2_name)

    def _find_upstream_nodes(
        self,
        start: dace_nodes.Node,
        state: dace.SDFGState,
        start_connector: Optional[str] = None,
        limit_node: Optional[dace_nodes.Node] = None,
    ) -> set[dace_nodes.Node]:
        """Finds all upstream nodes, i.e. all producers, of `start`.

        Args:
            start: Start the exploring from this node.
            state: The state in which it should be explored.
            start_connector: If given then only consider edges that uses this in
                connector. If `None` consider all incoming edges.
            limit_node: Consider this node as "limiting wall", i.e. do not explore
                beyond it.
        """
        seen: set[dace_nodes.Node] = set()

        if start_connector is None:
            to_visit: list[dace_nodes.Node] = [
                iedge.src for iedge in state.in_edges(start) if iedge.src is not limit_node
            ]
        else:
            to_visit = [
                iedge.src
                for iedge in state.in_edges_by_connector(start, start_connector)
                if iedge.src is not limit_node
            ]

        while len(to_visit) != 0:
            node = to_visit.pop()
            if node in seen:
                continue
            seen.add(node)
            to_visit.extend(
                iedge.src for iedge in state.in_edges(node) if iedge.src is not limit_node
            )

        return seen

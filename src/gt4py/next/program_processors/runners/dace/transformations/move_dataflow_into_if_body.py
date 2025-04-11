# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import functools
from typing import Any, Optional

import dace
from dace import (
    dtypes as dace_dtypes,
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation


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
        - The current implementation is restricted to only handle `if_block`s that
            are inside a Map.
        - By default if there is another `if_block` in the dataflow that would be
            inlinied into the matched `if_block` then the transformation will not
            apply. This behaviour can be disabled.

    Args:
        ignore_upstream_blocks: By default if there is an `if_block` that is upstream
            of the currently matched one the transformation does not apply.
            If this is `True` then ignore this case and apply.

    Note:
        - Extend the implementation that it is also able to handle `if_blocks` that
            are not inside a Map.
    """

    if_block = dace_transformation.PatternNode(dace_nodes.NestedSDFG)

    ignore_upstream_blocks = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="If 'True' ignore 'if_block's that are upstream.",
    )

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.if_block)]

    def __init__(
        self,
        ignore_upstream_blocks: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if ignore_upstream_blocks is not None:
            self.ignore_upstream_blocks = ignore_upstream_blocks
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
        assert isinstance(if_block, dace_nodes.NestedSDFG)

        # NOTE: The main benefit of requiring that the `if_block` must be inside a
        #   Map is that we know that every transient we encounter is single use data.
        enclosing_map = scope_dict[if_block]
        if not isinstance(enclosing_map, dace_nodes.MapEntry):
            return False

        # Test if the `if_block` is valid. This will also give us the names.
        if_block_spec = self._is_valid_if_block(if_block)
        if if_block_spec is None:
            return False

        # Compute the dataflow that is relocated.
        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: self._find_upstream_nodes(
                    start=if_block,
                    state=graph,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in if_block_spec
        )
        relocatable_dataflow = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=graph,
            if_block=if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
        )

        # If no branch has something to inline then we are done.
        if all(len(rel_df) == 0 for rel_df in relocatable_dataflow.values()):
            return False

        # Because we restrict ourself to only inline `if_blocks` only if they are
        #  inside a Map, we must them inline from top to bottom, so we must make
        #  there is no `if_block` upstream.
        if not self.ignore_upstream_blocks:
            for reloc_dataflow in relocatable_dataflow.values():
                if any(
                    self._check_if_block_upstream(
                        sdfg=sdfg,
                        state=graph,
                        if_block=upstream_if_block,
                        enclosing_map=enclosing_map,
                    )
                    for upstream_if_block in reloc_dataflow
                    if isinstance(upstream_if_block, dace_nodes.NestedSDFG)
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

        # Find the dataflow that should be relocated.
        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: self._find_upstream_nodes(
                    start=if_block,
                    state=graph,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in if_block_spec
        )
        relocatable_dataflow = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=graph,
            if_block=if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
        )

        # Finally relocate the dataflow
        for conn_name, nodes_to_move in relocatable_dataflow.items():
            self._replicate_dataflow_into_branche(
                state=graph,
                sdfg=sdfg,
                if_block=if_block,
                enclosing_map=enclosing_map,
                nodes_to_move=nodes_to_move,
                connector=conn_name,
            )

        # Now fix the symbol mapping.
        self._update_symbol_mapping(if_block)

        # Now remove the nodes that have been relocated from the SDFG and also
        #  clean up the registry.
        for nodes_to_move in relocatable_dataflow.values():
            for node_to_remove in nodes_to_move:
                if isinstance(node_to_remove, dace_nodes.AccessNode):
                    assert node_to_remove.desc(sdfg).transient
                    sdfg.remove_data(node_to_remove.data, validate=False)
                graph.remove_node(node_to_remove)

        # Because we relocate some node its seems that DaCe gets a bit confused.
        #  So we have to reset the list. Without it the `test_if_mover_chain`
        #  test would fail.
        sdfg.reset_cfg_list()

        # Readjust the Subsets.
        # TODO(phimuell): Technically only needed if we patched some data from
        #   beyond the Map inside the SDFG.
        dace_propagation.propagate_memlets_nested_sdfg(
                parent_sdfg=sdfg,
                parent_state=graph,
                nsdfg_node=if_block,
        )

    def _replicate_dataflow_into_branche(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
        nodes_to_move: set[dace_nodes.Node],
        connector: str,
    ) -> None:
        """Replicate the dataflow in `nodes_to_move` from `state` into `if_block`.

        First the function will determine into which branch, inside `if_block`,
        the dataflow has to be replicated. It will then copy the dataflow, nodes
        listed in `nodes_to_move` and insert them into that state.
        The function will then create the edges to connect them in the same way
        as they where outside. If there is an outer data dependency, for example
        a read to a global memory, then the function will patch that inside the
        `if_block`.
        At the end the function will remove the `connector`, but it will not remove
        the original dataflow.

        Args:
            sdfg: The sdfg that we process, the one that contains `state`.
            state: The state we operate on, the one that contains `if_block`.
            if_block: The `if_block` into which we inline.
            enclosing_map: The enclosing map.
            nodes_to_move: The list of nodes that should be removed.
            connector: The connector that should be inlined.
        """
        # Nothing to relocate nothing to do.
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
                    assert oedge.dst_conn == connector
                    # TODO(phimuell): Make subsets complete.
                    branch_state.add_edge(
                        new_nodes[oedge.src],
                        oedge.src_conn,
                        connector_node,
                        None,
                        dace.Memlet.from_memlet(oedge.data),
                    )
                else:
                    assert oedge.dst in nodes_to_move
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
                    # Empty Memlets are there to maintain some order relation, "happens
                    #  before". Depending on the situation we can remove or have to
                    #  recreate them. The case where the connection comes from a
                    #  node within the relocated dataflow is handled above.
                    assert iedge.src is enclosing_map
                    continue

                # Now we have to figuring out where the data is coming from, since
                #  the data was not relocated, we must patch it into `if_block`.
                if iedge.src is enclosing_map:
                    # The data is coming from outside the Map scope, i.e. not defined
                    #  inside the Map scope, so we have to trace it back.
                    memlet_path = state.memlet_path(iedge)
                    outer_data = memlet_path[0].src
                else:
                    # The data is defined somewhere in the Map scope itself.
                    outer_data = iedge.src
                assert isinstance(outer_data, dace_nodes.AccessNode)

                # If the data is not yet available in the inner SDFG made
                #  patch it through.
                if outer_data.data not in inner_sdfg.arrays:
                    inner_desc = sdfg.arrays[outer_data.data].clone()
                    inner_desc.transient = False
                    inner_sdfg.add_datadesc(outer_data.data, inner_desc)
                    # TODO(phimeull): Since we pass in everything the subsets along
                    #  the way might be wrong. This must be investigated and handled.
                    #  Probably running Memlet propagation is enough.
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
        symbols = if_block.sdfg.symbols
        for sym, symval in if_block.symbol_mapping.items():
            if sym not in symbols:
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
        """
        Locates the branch and the AccessNode to where the dataflow should be relocated.
        """
        inner_sdfg: dace.SDFG = if_block.sdfg
        conditional_block: dace.sdfg.state.ConditionalBlock = next(iter(inner_sdfg.nodes()))

        # This will locate the state where the first AccessNode that refers to
        #  `connector` is found. Since `_is_valid_if_block()` makes sure that
        #  there is only one match this is okay. But it must be changed, if we
        #  lift this restriction.
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

    def _check_if_block_upstream(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
    ) -> bool:
        """Check if `if_block` would be processed.

        Essentially, the function checks if there is dataflow that can be inlined
        into `if_block`. This is used to check if an `if_block` that was found
        upstream of another `if_block` that must be inlined first.
        The function skips some checks.

        Args:
            sdfg: The SDFG on which we operate.
            state: The state on which we operate on.
            if_block: The `if_block` into which we want to inline.
            enclosing_map: The limiting node, i.e. the MapEntry of the Map `if_block`
                is located in.
        """
        if_block_spec = self._is_valid_if_block(if_block)
        if if_block_spec is None:
            return False

        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: self._find_upstream_nodes(
                    start=if_block,
                    state=state,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in if_block_spec
        )
        filtered_relocatable_dataflow = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=state,
            if_block=if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
        )
        if all(len(rel_df) == 0 for rel_df in filtered_relocatable_dataflow.values()):
            return False

        return True

    def _filter_relocatable_dataflow(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        raw_relocatable_dataflow: dict[str, set[dace_nodes.Node]],
        non_relocatable_dataflow: dict[str, set[dace_nodes.Node]],
    ) -> dict[str, set[dace_nodes.Node]]:
        """Partition the dependencies.

        The function expects the dataflow that is upstream of every connector
        of the `if_block`. The function will then scan the dataflow and compute
        the parts that actually can be relocated and returns a `dict` mapping
        every relocatable input connector to the set of nodes that can be relocated.

        Note that the sets that are returned by this function are distinct.

        Args:
            state: The sate on which we operate.
            if_block: The `if_block` that is processed.
            raw_relocatable_dataflow: The connectors and their associated dataflow
                that can be relocated, not yet filtered.
            non_relocatable_dataflow: The connectors and their associated dataflow
                that can not be relocated.
        """

        # Remove the parts of the dataflow that is unrelocatable.
        all_non_relocatable_dataflow: set[dace_nodes.Node] = functools.reduce(
            lambda s1, s2: s1.union(s2), non_relocatable_dataflow.values(), set()
        )
        relocatable_dataflow = {
            conn_name: rel_df.difference(all_non_relocatable_dataflow)
            for conn_name, rel_df in raw_relocatable_dataflow.items()
        }

        # Now we determine the nodes that are in more than one sets.
        #  These sets must be removed, from the individual sets.
        known_nodes: set[dace_nodes.Node] = set()
        multiple_nodes: set[dace_nodes.Node] = set()
        for rel_df in relocatable_dataflow.values():
            seen_before: set[dace_nodes.Node] = known_nodes.intersection(rel_df)
            if len(seen_before) != 0:
                multiple_nodes.update(seen_before)
            known_nodes.update(rel_df)
        relocatable_dataflow = {
            conn_name: rel_df.difference(multiple_nodes)
            for conn_name, rel_df in relocatable_dataflow.items()
        }

        # However, not all dataflow can be moved inside the branch. For example if
        #  something is used outside the dataflow, that is moved inside the `if`,
        #  then we can not relocate it. As a simplicity we also remove AccessNodes
        #  referring to global memory.
        # TODO(phimuell): If we operate outside a Map then we would also have to
        #   check if the data is single use or not.
        def filter_nodes(
            branch_nodes: set[dace_nodes.Node],
            sdfg: dace.SDFG,
            state: dace.SDFGState,
        ) -> set[dace_nodes.Node]:
            # For this to work the `if_block` must be considered part, we remove it later.
            branch_nodes.add(if_block)
            has_been_updated = True
            while has_been_updated:
                has_been_updated = False
                for node in list(branch_nodes):
                    if node is if_block:
                        continue
                    remove_node = False
                    if isinstance(node, dace_nodes.AccessNode) and (
                        (not node.desc(sdfg).transient) or (state.in_degree(node) == 0)
                    ):
                        remove_node = True
                    if any(oedge.dst not in branch_nodes for oedge in state.out_edges(node)):
                        remove_node = True
                    if remove_node:
                        branch_nodes.remove(node)
                        has_been_updated = True
            assert if_block in branch_nodes
            branch_nodes.remove(if_block)
            return branch_nodes

        return {
            conn_name: filter_nodes(rel_df, sdfg, state)
            for conn_name, rel_df in relocatable_dataflow.items()
        }

    def _is_valid_if_block(
        self,
        if_block: dace_nodes.NestedSDFG,
    ) -> Optional[tuple[set[str], set[str]]]:
        """Checks if `if_block` is a valid block that the transformation operates on.

        If `if_block` is not applicable, then the function returns `None` if it is
        applicable the function returns a pair with two sets.
        The first lists contains the connectors whose upstream dataflow can be moved
        inside the if branches. The second list contains the connectors whose
        upstream dataflow can not be relocated inside an if body.
        """
        out_name = "__output"

        # There shall only be one output and three inputs with given names.
        if if_block.out_connectors.keys() != {out_name}:
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

        # See note in `_find_branch_for()`
        reference_count: dict[str, int] = {conn_name: 0 for conn_name in if_block.in_connectors}
        for _, branche in inner_if_block.branches:
            for inner_state in branche.all_states():
                assert isinstance(inner_state, dace.SDFGState)
                found_output_node = False
                for node in inner_state.nodes():
                    if not isinstance(node, dace_nodes.AccessNode):
                        return None
                    if node.data in reference_count:
                        reference_count[node.data] += 1
                        exp_in_deg, exp_out_deg = 0, 1
                    elif node.data == out_name:
                        exp_in_deg, exp_out_deg = 1, 0
                        found_output_node = True
                    else:
                        return None
                    if inner_state.in_degree(node) != exp_in_deg:
                        return None
                    if inner_state.out_degree(node) != exp_out_deg:
                        return None
                if not found_output_node:
                    return None

        # The connectors that can be pulled inside must appear exactly once.
        #  In theory they could appear more, but then we would have to replicate
        #  the dataflow to different locations which is not supported.
        #  So the ones that can be relocated were found exactly once. Zero would
        #  mean they can not be relocated and more than one means that we do not
        #  support it yet.
        relocatable_connectors = {
            conn_name for conn_name, conn_count in reference_count.items() if conn_count == 1
        }
        non_relocatable_connectors = {
            conn_name
            for conn_name in reference_count.keys()
            if conn_name not in relocatable_connectors
        }
        if len(non_relocatable_connectors) == 0:
            return None
        if len(relocatable_connectors) == 0:
            return None
        return relocatable_connectors, non_relocatable_connectors

    def _find_upstream_nodes(
        self,
        start: dace_nodes.Node,
        state: dace.SDFGState,
        start_connector: Optional[str] = None,
        limit_node: Optional[dace_nodes.Node] = None,
    ) -> set[dace_nodes.Node]:
        """Finds all upstream nodes, i.e. all producers, of `start`.

        It is important to note that this function does not perform any filtering,
        it just finds the nodes. The filtering is done in `_filter_relocatable_dataflow()`.

        Args:
            start: Start the exploring from this node.
            state: The state in which it should be explored.
            start_connector: If given then only consider edges that terminate
                in this connector.
            limit_node: Consider this node as "limiting wall", i.e. do not explore
                beyond it.
        """
        seen: set[dace_nodes.Node] = set()

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

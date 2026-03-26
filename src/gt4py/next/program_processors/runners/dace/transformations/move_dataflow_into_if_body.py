# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
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
from dace.sdfg import (
    nodes as dace_nodes,
    propagation as dace_propagation,
    type_inference as dace_type_inference,
    utils as dace_sutils,
)

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


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
    I.e. computation that is needed in only one branch is relocated or inlined into
    that specific branch, which might reduce computation.

    Despite its name the transformation is not only able to handle `if` expressions,
    but more general `switch` like expressions. The requirements are:
    - The expression must be inside a nested SDFG containing only a `ConditionalBlock`.
    - The matched nested SDFG must be inside a Map (might be dropped).
    - For every incoming connector there must be exactly one AccessNode in the
        entire nested SDFG (might be dropped).
    - Every branch must write to every output of the nested SDFG.
    - The only dataflow allowed inside the branches is `(i) -> (o)` i.e. AccessNode
        to AccessNode connections.

    Furthermore, the transformation should be applied as long as possible, i.e.
    it should be applied by passing it to `SDFG.apply_transformations_repeated()`.

    Args:
        ignore_upstream_blocks: If `True` do not require that upstream `if_block`s
            have to be processed first.

    Note:
        - If there is a chain of suitable `if` expression, i.e. an `if` expression is
            in the upstream dataflow of the one that is currently matched, then the
            transformation will not apply. This is done to ensure that everything is
            properly inlined into the branches. This behaviour can be disabled by
            setting `ignore_upstream_blocks` to `True`.
            The reason for this behaviour is that the current implementation can only
            handle the case where the `if` expression is _directly_ inside a Map.

    Todo:
        - Allow that an inconnector can be used multiple times, as long as it is in
            different branches.
        - Extend the implementation that it is also able to handle `if_blocks` that
            are not inside a Map. This would allow to drop the need to process
            upstream `if` expressions first.
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
        if_block_spec = self._partition_if_block(sdfg, if_block)
        if if_block_spec is None:
            return False
        relocatable_connectors, non_relocatable_connectors, connector_usage_location = if_block_spec

        # Compute the dataflow that is relocated.
        # NOTE: That the nodes sets are not sorted in any way, however, the
        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: gtx_transformations.utils.find_upstream_nodes(
                    start=if_block,
                    state=graph,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in [relocatable_connectors, non_relocatable_connectors]
        )
        relocatable_dataflow = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=graph,
            if_block=if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
            connector_usage_location=connector_usage_location,
            enclosing_map=enclosing_map,
        )
        if len(relocatable_dataflow) == 0:
            return False

        # Check if relatability is possible.
        if not self._check_for_data_and_symbol_conflicts(
            sdfg=sdfg,
            state=graph,
            relocatable_dataflow=relocatable_dataflow,
            enclosing_map=enclosing_map,
            if_block=if_block,
        ):
            return False

        # Because the transformation can only handle `if` expressions that
        #  are _directly_ inside a Map, we must check if the upstream contains
        #  suitable `if` expressions that must be processed first. The simplest way
        #  is to return `False` and not apply. However, this implies that the
        #  transformation is applied in a loop until it applies nowhere anymore.
        # NOTE: This is a restriction due to the current implementation.
        if not self.ignore_upstream_blocks:
            if any(
                self._has_if_block_relocatable_dataflow(
                    sdfg=sdfg,
                    state=graph,
                    upstream_if_block=upstream_if_block,
                    enclosing_map=enclosing_map,
                )
                for upstream_if_block in relocatable_dataflow
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
        enclosing_map = graph.scope_dict()[if_block]
        relocatable_connectors, non_relocatable_connectors, connector_usage_location = (
            self._partition_if_block(sdfg, if_block)  # type: ignore[misc]  # Guaranteed to be not None.
        )

        # Find the dataflow that should be relocated.
        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: gtx_transformations.utils.find_upstream_nodes(
                    start=if_block,
                    state=graph,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in [relocatable_connectors, non_relocatable_connectors]
        )
        relocatable_dataflow: set[dace_nodes.Node] = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=graph,
            if_block=if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
            connector_usage_location=connector_usage_location,
            enclosing_map=enclosing_map,
        )

        # Bring the nodes in a deterministic order, which is induced by the underlying state.
        # NOTE: The following key function is equivalent to use `lambda n: graph.node_id(n)`
        #   but instead of O[N^2] it is O[N].
        node_keys = {node: i for i, node in enumerate(graph.nodes())}
        nodes_to_move = sorted(relocatable_dataflow, key=lambda n: node_keys[n])

        # For each node we have to find out in which state inside the `if_block` it will
        #  end up. `relocation_destination` has a fixed order.
        relocation_destination: dict[dace_nodes.Node, dace.SDFGState] = {}
        for node_to_move in nodes_to_move:
            # Although `node_top_move` could be reached through different connectors
            #  they are all associated to the same branch.
            target_state: Optional[dace.SDFGState] = None
            for conn, raw_reloc_dataflow_of_conn in raw_relocatable_dataflow.items():
                if node_to_move in raw_reloc_dataflow_of_conn:
                    target_state = connector_usage_location[conn][0]
                    break
            else:
                raise ValueError("Could not find node '{node_to_move}'")
            assert target_state is not None
            relocation_destination[node_to_move] = target_state

        # Relocate the dataflow.
        self._replicate_dataflow_into_branch(
            state=graph,
            sdfg=sdfg,
            if_block=if_block,
            enclosing_map=enclosing_map,
            relocation_destination=relocation_destination,
            connector_usage_location=connector_usage_location,
        )

        # Must be performed after relocation.
        self._update_symbol_mapping(
            sdfg=sdfg,
            if_block=if_block,
        )

        self._remove_outside_dataflow(
            sdfg=sdfg,
            state=graph,
            relocation_destination=relocation_destination,
        )

        # Because we relocate some node it seems that DaCe gets a bit confused.
        #  So we have to reset the list. Without it the `test_if_mover_chain`
        #  test would fail.
        sdfg.reset_cfg_list()

        # Readjust the Subsets.
        #  As a side effect this call will also properly propagate the nested SDFG `if_block`.
        dace_propagation.propagate_memlets_map_scope(sdfg, graph, enclosing_map)

    def _replicate_dataflow_into_branch(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
        relocation_destination: dict[dace_nodes.Node, dace.SDFGState],
        connector_usage_location: dict[str, tuple[dace.SDFGState, dace_nodes.AccessNode]],
    ) -> None:
        """Replicate the dataflow in `relocation_destination` into `if_block`.

        The function will replicate the dataflow listed in `relocatable_connectors.keys()`,
        that needs
        to be connected, in some way, to the `if_block`. It will remove the connectors
        that are no longer needed, but it will not remove the original dataflow nor
        update the symbol mapping.

        Args:
            sdfg: The sdfg that we process, the one that contains `state`.
            state: The state we operate on, the one that contains `if_block`.
            if_block: The `if_block` into which we inline.
            enclosing_map: The enclosing map.
            nodes_to_move: The list of nodes that should be moved.
            connector_usage_location: Maps connector names to the state and AccessNode
                where they appear inside the nested SDFG.
        """
        inner_sdfg = if_block.sdfg

        # Maps old nodes to the new relocated nodes inside the `if_block`. Note that
        #  the state _inside_ the `if_block` is part of the key. This is needed to
        #  handle the "outside Map data" which must be mapped into multiple states.
        node_map: dict[tuple[dace_nodes.Node, dace.SDFGState], dace_nodes.Node] = dict()
        rename_map: dict[tuple[str, dace.SDFGState], str] = dict()

        # Check what data is already fully mapped into the `if_block`. There might be
        #  aliasing, i.e. multiple inner names refer to the same outer name.
        # TODO(phimuell): Investigate if it would be better if we handle partially
        #   mapped in data such by fully map it in and perform the slicing outside.
        fully_mapped_in_data: dict[str, set[str]] = collections.defaultdict(set)
        for if_iedge in state.in_edges(if_block):
            if if_iedge.data.is_empty():
                continue
            outer_data = if_iedge.data.data
            mapped_in_range = if_iedge.data.subset  # Is always `.subset`.
            outer_desc = sdfg.arrays[outer_data]
            if mapped_in_range.covers(dace_sbs.Range.from_array(outer_desc)) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                fully_mapped_in_data[outer_data].add(if_iedge.dst_conn)

        # Replicate the nodes into the `if_block` and create the needed data The
        #  "outside Map data" will be handled when we handle the incoming edges.
        for origin_node, branch_state in relocation_destination.items():
            reloc_node = copy.deepcopy(origin_node)
            node_map[(origin_node, branch_state)] = reloc_node
            branch_state.add_node(reloc_node)

            # If we relocate an AccessNode, we have to make sure that the data descriptor
            #  is also added to the nested SDFG. We allow renaming of data containers
            #  but we do not allow renaming of symbols, this is checked by
            #  `_check_for_data_and_symbol_conflicts()`.
            if isinstance(origin_node, dace_nodes.AccessNode):
                assert sdfg.arrays[origin_node.data].transient
                # TODO(phimuell): Handle the case we need to rename something.
                new_data_name = inner_sdfg.add_datadesc(
                    origin_node.data,
                    sdfg.arrays[origin_node.data].clone(),
                    find_new_name=True,
                )
                reloc_node.data = new_data_name
                rename_map[(origin_node.data, branch_state)] = new_data_name

        # We now create the mapped nodes, i.e. the nodes that are not relocated but
        #  have to be put inside the `if_block`. We find them by looking at the input
        #  edges, that do not lead to a node that is relocated. Connections between
        #  relocated nodes are handled later.
        for origin_node, branch_state in relocation_destination.items():
            for iedge in state.in_edges(origin_node):
                if iedge.src in relocation_destination:
                    # Dependency between two relocated nodes: Handled below.
                    continue
                elif iedge.data.is_empty():
                    # This is an empty Memlet that is between a node that is relocated
                    #  and a node that is not relocated. Because we move the destination
                    #  of the edge into the `if_block` the "happens before" relation
                    #  is automatically handled and this edge is no longer needed.
                    continue

                # Now we have to figuring out where the data is coming from, since
                #  the data was not relocated, we must patch it into `if_block`.
                if iedge.src is enclosing_map:
                    # The data is coming from outside the Map scope, i.e. not defined
                    #  inside the Map scope, so we have to trace it back.
                    memlet_path = state.memlet_path(iedge)
                    outer_node = memlet_path[0].src
                else:
                    # The data is defined somewhere in the Map scope itself.
                    outer_node = iedge.src

                # TODO(phimuell): It is possible that this does not lead to an
                #   AccessNode on the outside, but to something inside the Map scope
                #   such as the MapExit of an inner map. To handle such a case we need
                #   to construct the set of nodes to move differently, i.e.
                #   considering this case already there.
                if not isinstance(outer_node, dace_nodes.AccessNode):
                    raise NotImplementedError()
                assert not gtx_transformations.utils.is_view(outer_node, sdfg)

                outer_data = outer_node.data
                outer_desc = sdfg.arrays[outer_data]

                if (outer_node, branch_state) in node_map:
                    # The node is already mapped into this state.
                    assert (outer_data, branch_state) in rename_map
                    assert not node_map[(outer_node, branch_state)].desc(inner_sdfg).transient
                    pass

                elif outer_data in fully_mapped_in_data:
                    # The data has already been mapped into the `if_block`, but not in
                    #  `branch_state`. We first look if the state contains an AccessNode
                    #  referring to that data.
                    outer_aliases = fully_mapped_in_data[outer_data]
                    candidate_nodes: list[dace_nodes.AccessNode] = sorted(
                        (
                            dnode
                            for dnode in branch_state.data_nodes()
                            if dnode.data in outer_aliases
                        ),
                        key=lambda dnode: dnode.data,
                    )

                    if len(candidate_nodes) == 0:
                        # There is no AccessNode in the state so we have to create one.
                        inner_data = sorted(outer_aliases)[0]
                        inner_node = branch_state.add_access(inner_data)

                    else:
                        # There is an AccessNode in the state. To handle some legal
                        #  but unlikely case we check that nodes we found are all
                        #  source nodes. We have to do this to prevent read-write
                        #  conflicts.
                        candidate_source_nodes = [
                            dnode for dnode in candidate_nodes if branch_state.in_degree(dnode) == 0
                        ]
                        if len(candidate_source_nodes) != len(candidate_nodes):
                            raise NotImplementedError()

                        # We take the first node, since they are sorted it is deterministic.
                        inner_node = candidate_source_nodes[0]

                    assert (outer_data, branch_state) not in rename_map
                    assert not inner_sdfg.arrays[inner_node.data].transient
                    rename_map[(outer_data, branch_state)] = inner_node.data
                    node_map[(outer_node, branch_state)] = inner_node

                else:
                    # The data is not already mapped in and is also unknown.
                    #  Here we rely on that we do not have to perform symbol renaming.
                    inner_data = inner_sdfg.add_datadesc(
                        outer_data,
                        outer_desc.clone(),
                        find_new_name=True,
                    )
                    inner_sdfg.arrays[inner_data].transient = False

                    state.add_edge(
                        iedge.src,
                        iedge.src_conn,
                        if_block,
                        inner_data,
                        dace.Memlet.from_array(outer_data, outer_desc),
                    )
                    if_block.add_in_connector(inner_data)

                    inner_node = branch_state.add_access(inner_data)
                    rename_map[(outer_data, branch_state)] = inner_node.data
                    node_map[(outer_node, branch_state)] = inner_node
                    fully_mapped_in_data[outer_data].add(inner_data)

                # Now create the edge in the inner state.
                new_edge = branch_state.add_edge(
                    node_map[(outer_node, branch_state)],
                    None,
                    node_map[(iedge.dst, branch_state)],
                    iedge.dst_conn,
                    copy.deepcopy(iedge.data),
                )
                new_edge.data.data = rename_map[(outer_data, branch_state)]

        # Now create the edges between the relocated nodes, which are all the outgoing
        #  edges, the `if_block` is handled as a special relocated node and its
        #  connectors (but not the edges) are removed to.
        # NOTE: This loop can not be fused with the one above and must run after it.
        for origin_node, branch_state in relocation_destination.items():
            for oedge in state.out_edges(origin_node):
                if oedge.dst is if_block:
                    # This defines the "argument" to the nested SDFG. This means that
                    #  the new destination now is the single node inside `if_block`
                    #  that represents the argument.
                    assert not inner_sdfg.arrays[oedge.dst_conn].transient
                    assert branch_state is connector_usage_location[oedge.dst_conn][0]
                    assert isinstance(oedge.src, dace_nodes.AccessNode)
                    assert oedge.data.wcr is None and oedge.data.other_subset is None

                    branch_state.add_edge(
                        node_map[(oedge.src, branch_state)],
                        oedge.src_conn,
                        connector_usage_location[oedge.dst_conn][1],
                        None,
                        dace.Memlet(
                            data=rename_map[(oedge.data.data, branch_state)],
                            subset=oedge.data.subset,  # Is always subset.
                            other_subset=dace_sbs.Range.from_array(
                                inner_sdfg.arrays[oedge.dst_conn]
                            ),
                            volume=oedge.data.volume,
                            dynamic=oedge.data.dynamic,
                        ),
                    )

                    # The inner data is no longer a global but has become a transient.
                    assert oedge.dst_conn in if_block.in_connectors
                    inner_sdfg.arrays[oedge.dst_conn].transient = True
                    if_block.remove_in_connector(oedge.dst_conn)

                else:
                    # Edges that do not go to the `if_block` must lead to a node
                    #  that is also relocated.
                    assert origin_node in relocation_destination
                    new_oedge = branch_state.add_edge(
                        node_map[(oedge.src, branch_state)],
                        oedge.src_conn,
                        node_map[(oedge.dst, branch_state)],
                        oedge.dst_conn,
                        dace.Memlet.from_memlet(oedge.data),
                    )
                    if not oedge.data.is_empty():
                        new_oedge.data.data = rename_map[(oedge.data.data, branch_state)]

    def _remove_outside_dataflow(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        relocation_destination: dict[dace_nodes.Node, dace.SDFGState],
    ) -> None:
        """Removes the original dataflow, that has been relocated.

        The function will also remove data containers that are no longer in use.
        """

        # Clean up the dataflow first, before removing the nodes.
        for node_to_remove in relocation_destination:
            # Create all "interface edges", i.e. connecting a relocated node with one
            #  that is not. This is needed to properly remove dangling Memlet paths.
            for iedge in list(state.in_edges(node_to_remove)):
                if iedge.src in relocation_destination:
                    continue
                dace_sutils.remove_edge_and_dangling_path(state, iedge)

            if isinstance(node_to_remove, dace_nodes.AccessNode):
                # NOTE: We can remove the data here, because by assumption data that is
                #   referred to by an AccessNode inside a Map is single use data and
                #   used nowhere else.
                # NOTE: This will temporarily create an invalid SDFG.
                assert node_to_remove.desc(sdfg).transient
                sdfg.remove_data(node_to_remove.data, validate=False)

        # Remove the original nodes (data descriptors were deleted in the loop above).
        state.remove_nodes_from(relocation_destination.keys())

    def _update_symbol_mapping(
        self,
        sdfg: dace.SDFG,
        if_block: dace_nodes.NestedSDFG,
    ) -> None:
        """Updates the symbol mapping of the nested SDFG.

        The function assumes that the symbols that are missing in the nested SDFG
        are available in the parent SDFG.
        """
        symbol_mapping = if_block.symbol_mapping
        missing_symbols = sorted(
            (ms for ms in if_block.sdfg.free_symbols if ms not in symbol_mapping),
            key=lambda sym: str(sym),
        )
        symbol_mapping.update({s: s for s in missing_symbols})
        if_block.symbol_mapping = symbol_mapping  # Performs conversion.

        # Add new global symbols to nested SDFG.
        #  The code is based on `SDFGState.add_nested_sdfg()`.
        if_block_symbols = if_block.sdfg.symbols
        parent_symbols = sdfg.symbols
        for new_sym in missing_symbols:
            if new_sym in if_block_symbols:
                # The symbol is already known, so we check that it is the same type as in the
                #  parent SDFG.
                assert if_block_symbols[new_sym] == parent_symbols[new_sym]

            elif new_sym in parent_symbols:
                # The symbol is known to the parent SDFG, so take the type from there.
                if_block.sdfg.add_symbol(new_sym, parent_symbols[new_sym])

            else:
                # Figuring out the type of the symbol based on the computation we do.
                # TODO(phimuell): Maybe switch to `symbols_defined_at()` as it is indicated
                #   in the `SDFGState.add_nested_sdfg()` function.
                if_block.sdfg.add_symbol(
                    new_sym,
                    dace_type_inference.infer_expr_type(new_sym, parent_symbols)
                    or dace_dtypes.typeclass(int),
                )

    def _check_for_data_and_symbol_conflicts(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        relocatable_dataflow: set[dace_nodes.Node],
        if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
    ) -> bool:
        """Check if the relocation would cause any conflict, such as a symbol clash."""

        # TODO(phimuell): There is an obscure case where the nested SDFG, on its own,
        #   defines a symbol that is also mapped, for example a dynamic Map range.
        #   It is probably not a problem, because of the scopes DaCe adds when
        #   generating the C++ code.

        # This will give us the "internal symbols" that need to be mapped into `if_block`.
        #  It does not include all symbols, see bellow.
        requiered_symbols: set[str] = dace.sdfg.state.StateSubgraphView(
            state, relocatable_dataflow
        ).free_symbols
        assert all(isinstance(sym, str) for sym in requiered_symbols)

        # The internal symbols missing the symbols that are needed by the nodes that
        #  are just mapped into the `if_block` as well as the connections that connects
        #  relocated and mapped nodes.
        for node_to_check in relocatable_dataflow:
            for iedge in state.in_edges(node_to_check):
                if iedge.src in relocatable_dataflow:
                    continue  # Ignore internal connections, handled in subgraph.
                elif iedge.data.is_empty():
                    continue  # Empty Memlets do not have symbols.

                if iedge.src is enclosing_map:
                    # Outside-Map data must be mapped. Here we only have to consider
                    #  the symbols of the node and can ignore the symbols of the edge.
                    memlet_path = state.memlet_path(iedge)
                    node_to_map = memlet_path[0].src
                else:
                    # The mapped node is inside the Map this means we replicate this
                    #  edge thus in addition to the symbols of the data, we need the
                    #  symbols needed by the edge.
                    node_to_map = iedge.src
                    requiered_symbols |= {
                        str(sym) for sym in iedge.data.used_symbols(True, edge=iedge)
                    }

                # Only AccessNodes can be mapped into `if_block`.
                if not isinstance(node_to_map, dace_nodes.AccessNode):
                    return False

                # Add the symbols of the data.
                requiered_symbols |= {
                    str(sym) for sym in sdfg.arrays[node_to_map.data].used_symbols(True)
                }

        # A conflicting symbol is a free symbol of the relocatable dataflow, that is not a
        #  direct mapping. For example if there is a symbol `n` on the inside and outside
        #  then everything is okay if the symbol mapping is `{n: n}` i.e. the symbol has the
        #  same meaning inside and outside. Everything else is not okay.
        symbol_mapping = if_block.symbol_mapping
        conflicting_symbols = requiered_symbols.intersection((str(k) for k in symbol_mapping))
        for conflicting_symbol in conflicting_symbols:
            if conflicting_symbol != str(symbol_mapping[conflicting_symbol]):
                return False

        return True

    def _has_if_block_relocatable_dataflow(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        upstream_if_block: dace_nodes.NestedSDFG,
        enclosing_map: dace_nodes.MapEntry,
    ) -> bool:
        """Check if `upstream_if_block` has relocatable dataflow.

        The function is used to enforce the rule that no `if` expression should be
        processed before all suitable `if` expressions in its upstream dataflow of
        the matched `if_block` has been processed first.

        Args:
            sdfg: The SDFG on which we operate.
            state: The state on which we operate on.
            upstream_if_block: The `if` expression that was found in the relocatable
                dataflow of the matched `if_block`.
            enclosing_map: The limiting node, i.e. the MapEntry of the Map `if_block`
                is located in.
        """
        if_block_spec = self._partition_if_block(sdfg, upstream_if_block)
        if if_block_spec is None:
            return False
        *classified_connectors, connector_usage_location = if_block_spec

        raw_relocatable_dataflow, non_relocatable_dataflow = (
            {
                conn_name: gtx_transformations.utils.find_upstream_nodes(
                    start=upstream_if_block,
                    state=state,
                    start_connector=conn_name,
                    limit_node=enclosing_map,
                )
                for conn_name in conn_names
            }
            for conn_names in classified_connectors
        )
        filtered_relocatable_dataflow = self._filter_relocatable_dataflow(
            sdfg=sdfg,
            state=state,
            if_block=upstream_if_block,
            raw_relocatable_dataflow=raw_relocatable_dataflow,
            non_relocatable_dataflow=non_relocatable_dataflow,
            connector_usage_location=connector_usage_location,
            enclosing_map=enclosing_map,
        )
        if len(filtered_relocatable_dataflow) == 0:
            return False

        return True

    def _filter_relocatable_dataflow(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        if_block: dace_nodes.NestedSDFG,
        raw_relocatable_dataflow: dict[str, set[dace_nodes.Node]],
        non_relocatable_dataflow: dict[str, set[dace_nodes.Node]],
        connector_usage_location: dict[str, tuple[dace.SDFGState, dace_nodes.AccessNode]],
        enclosing_map: dace_nodes.MapEntry,
    ) -> set[dace_nodes.Node]:
        """Compute the final set of the relocatable nodes.

        The function expects the dataflow that is upstream of every connector
        of the `if_block`. The function will then scan the dataflow and compute
        the parts that actually can be relocated. It will then return a `set`
        containing all nodes that can actually be relocated. If this set is empty
        then nothing can be relocated.
        Note that the returned `set` is in an unspecific order and before processing
        should be ordered.

        Args:
            state: The state on which we operate.
            if_block: The `if_block` that is processed.
            raw_relocatable_dataflow: The connectors and their associated dataflow
                that can be relocated, not yet filtered.
            non_relocatable_dataflow: The connectors and their associated dataflow
                that can not be relocated.
            connector_usage_location: Maps a connector to the state and AccessNode
                inside the if block.
            enclosing_map: The limiting node, i.e. the MapEntry of the Map where
                `if_block` is located in.
        """

        # These are the nodes that can not be relocated anyway.
        all_non_relocatable_dataflow: set[dace_nodes.Node] = functools.reduce(
            lambda s1, s2: s1.union(s2), non_relocatable_dataflow.values(), set()
        )

        # While we can relocate nodes that are needed by multiple connectors, we can
        #  not handle the case if they end up in multiple branches.
        nodes_in_states: dict[dace.SDFGState, set[dace_nodes.Node]] = collections.defaultdict(set)
        for conn_name, rel_df in raw_relocatable_dataflow.items():
            nodes_in_states[connector_usage_location[conn_name][0]].update(rel_df)
        state_nodes_sets = list(nodes_in_states.values())  # Order is unimportant here.
        for i, state_nodes in enumerate(state_nodes_sets):
            for j in range(i + 1, len(state_nodes_sets)):
                all_non_relocatable_dataflow.update(state_nodes.intersection(state_nodes_sets[j]))

        # The dataflow that must happen before the `if_block`, i.e that is connected
        #  with it by an empty Memlet can not be reconnected.
        for if_block_iedge in state.in_edges(if_block):
            if if_block_iedge.src is enclosing_map:
                continue
            elif not if_block_iedge.data.is_empty():
                continue
            all_non_relocatable_dataflow.update(
                gtx_transformations.utils.find_upstream_nodes(
                    start=if_block_iedge.src,
                    state=state,
                )
            )
            all_non_relocatable_dataflow.add(if_block_iedge.src)

        # Instead of scanning the nodes associated to each connector separately we will
        #  process all of them together. We do this because a node can be associated to
        #  multiple connectors and as such data dependencies can show up. We will,
        #  after the filtering distribute them back.
        nodes_proposed_for_reloc: set[dace_nodes.Node] = functools.reduce(
            lambda s1, s2: s1.union(s2), raw_relocatable_dataflow.values(), set()
        )

        # Filtering out all nodes that can not be relocated anyway.
        if all_non_relocatable_dataflow:
            nodes_proposed_for_reloc.difference_update(all_non_relocatable_dataflow)

        # TODO(phimuell): Better screening of empty Memlets.
        has_been_updated = True
        while has_been_updated:
            has_been_updated = False

            for reloc_node in list(nodes_proposed_for_reloc):
                # The node was already removed in a previous iteration.
                if reloc_node not in nodes_proposed_for_reloc:
                    continue

                # Because we are currently always inside a Map
                assert state.in_degree(reloc_node) > 0

                # If the node is needed by anything that is not also moved
                #  into the `if` body, then it has to remain outside. For that we
                #  have to pretend that `if_block` is also relocated.
                if any(
                    oedge.dst not in nodes_proposed_for_reloc
                    for oedge in state.out_edges(reloc_node)
                    if oedge.dst is not if_block
                ):
                    nodes_proposed_for_reloc.remove(reloc_node)
                    has_been_updated = True
                    continue

                # We do not look at incoming edges that comes from nodes that are not
                #  mappable, i.e. AccessNodes. In addition to AccessNodes we also
                #  ignore `enclosing_map` because it acts as a boundary anyway and
                #  on its other side is an AccessNode anyway.
                non_mappable_incoming_nodes: set[dace_nodes.Node] = {
                    iedge.src
                    for iedge in state.in_edges(reloc_node)
                    if not (
                        (iedge.src is enclosing_map) or isinstance(iedge.src, dace_nodes.AccessNode)
                    )
                }
                if non_mappable_incoming_nodes.issubset(nodes_proposed_for_reloc):
                    # All nodes that can not be mapped into the `if` body are
                    #  currently scheduled to be relocated, thus there is no
                    #  problem.
                    pass

                else:
                    # Only some of the non mappable nodes are selected to be moved
                    #  inside the `if` body. This means that `reloc_node` can also
                    #  not be moved because of its input dependencies. Since we can
                    #  not relocate `reloc_node` this also implies that none of its
                    #  inputs either.
                    nodes_proposed_for_reloc.difference_update(non_mappable_incoming_nodes)
                    nodes_proposed_for_reloc.remove(reloc_node)
                    has_been_updated = True

        return nodes_proposed_for_reloc

    def _partition_if_block(
        self,
        sdfg: dace.SDFG,
        if_block: dace_nodes.NestedSDFG,
    ) -> Optional[
        tuple[list[str], list[str], dict[str, tuple[dace.SDFGState, dace_nodes.AccessNode]]]
    ]:
        """Check if `if_block` can be processed and partition the input connectors.

        The function will check if `if_block` has the right structure, i.e. if it is
        roughly equivalent to the Python expression `a if cond else b`.
        It will also identify which input connectors refer to dataflow that can
        be inlined into the `if_block` and which can not.

        Returns:
            If `if_block` is unsuitable the function will return `None`. In case the
            `if_block` is suitable a `tuple` of length three is returned.
            The first element is a `list`, which is never empty, containing all
            input connectors that can be relocated. The list is sorted in a stable
            order. The second element is a list containing all input connectors that
            can not be relocated, it can be empty and is not in a particular order.
            The third element is a `dict` that maps connectors to a pair containing
            the state (inside the nested SDFG) and the only `AccessNode` that refers
            to that connector.
            It is important that only the first element of the `tuple` has a guaranteed
            order.
        """
        if len(if_block.out_connectors.keys()) == 0:
            return None

        input_names: set[str] = set(if_block.in_connectors.keys())
        output_names: set[str] = set(if_block.out_connectors.keys())

        # If data is used as input and output we ignore it.
        # TODO(phimuell): Think if this case can be handled.
        input_names.difference_update(output_names)
        if len(input_names) == 0:
            return None

        # We require that the nested SDFG contains a single node, which is a `ConditionalBlock`.
        inner_sdfg: dace.SDFG = if_block.sdfg
        if inner_sdfg.number_of_nodes() != 1:
            return None
        inner_if_block = next(iter(inner_sdfg.nodes()))
        if not isinstance(inner_if_block, dace.sdfg.state.ConditionalBlock):
            return None

        # Mapping between the connector and the inner access node.
        connector_usage_location: dict[str, tuple[dace.SDFGState, dace_nodes.AccessNode]] = {}

        # This is the dataflow that can not be relocated.
        non_relocatable_connectors: set[str] = set()

        # Now inspect all states.
        for _, if_branch in inner_if_block.branches:
            for inner_state in if_branch.all_states():
                for dnode in inner_state.data_nodes():
                    node_data = dnode.data

                    # Check if we can skip the data.
                    if node_data in non_relocatable_connectors:
                        continue
                    elif node_data in output_names:
                        continue
                    elif dnode.desc(inner_sdfg).transient:
                        continue
                    assert node_data in input_names

                    if node_data in connector_usage_location:
                        # There are multiple AccessNodes referring to the same connector
                        #  which is currently not supported. In theory they could appear
                        #  more, but then we would have to replicate the dataflow to
                        #  different locations which is not supported. We allow such
                        #  situations but consider the connector non relocatable.
                        connector_usage_location.pop(node_data)
                        non_relocatable_connectors.add(node_data)

                    elif inner_state.in_degree(dnode) != 0:
                        # The node is also written to, allowed by SDFG grammar, but we
                        #  do not allow it.
                        non_relocatable_connectors.add(node_data)

                    else:
                        # This is a proper input connector node.
                        connector_usage_location[node_data] = (inner_state, dnode)

                    # If all input connectors were classified as non relocatable
                    #  then the partition does not exist.
                    if len(non_relocatable_connectors) == len(input_names):
                        assert non_relocatable_connectors == input_names
                        return None

        # There is nothing to relocate.
        if len(connector_usage_location) == 0:
            return None

        # In addition to the non relocatable connectors that were found above, we also
        #  mark all connectors that were not found as non relocatable.
        non_relocatable_connectors.update(
            conn for conn in input_names if conn not in connector_usage_location
        )

        # We require that at least one non relocatable dataflow is there, this is for
        #  the condition. This is not strictly needed, as it could also be passed as
        #  a symbol, but currently the lowering does not do this and we keep it as
        #  a sanity check.
        if len(non_relocatable_connectors) == 0:
            return None

        # We only guarantee that `relocatable_connectors` has an stable order,
        #  everything else has no guaranteed order, even `connector_usage_location`.
        relocatable_connectors = sorted(connector_usage_location.keys())

        return relocatable_connectors, list(non_relocatable_connectors), connector_usage_location

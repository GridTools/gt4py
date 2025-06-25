# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Final, Literal, Optional, Sequence, Union, overload

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    splitting_tools as gtx_dace_split,
)


@dace_properties.make_properties
class SingleStateGlobalSelfCopyElimination(dace_transformation.SingleStateTransformation):
    """Remove global self copy.

    This transformation matches the following case `(G) -> (T) -> (G)`, i.e. parts
    of `G` are copied into `T` and then back into `G`.
    According to ADR-18 we have the guarantee that the data is "pointwise", i.e.
    that the data is not copied to another location, as the NumPy instruction
    `A[2:10] = A[0:8]` would do, as this is invalid to do with a Memlet.

    In the simplest case the three nodes are merged together into one.
    However, depending on the situation, for example if this would create a cycle,
    the function will only partially fuse them together. Currently only the case
    where `T` and the second `G` node are merged is implemented. In this mode the
    transformation will remove all _direct_ links between the first `G` node and
    `T` and redirect all read writes from `T` to the second node.

    Todo:
        - Investigate if read write conflict checking is really needed.
        - Handle the case of where cycles would be created better, especially
            in conjunction with indirect reads.
    """

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and must be passed at construction time. Needed
    #  until [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    node_g1 = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_tmp = dace_transformation.transformation.PatternNode(dace_nodes.AccessNode)
    node_g2 = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # These are the different merging strategies, see `_check_merging_strategy()`
    #  for more information:
    _FULL_MERGE: Final[int] = 0
    _MERGE_TMP_G2: Final[int] = 1

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.node_g1, cls.node_tmp, cls.node_g2)]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = None
        if single_use_data is not None:
            self._single_use_data = single_use_data

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        node_g1 = self.node_g1
        node_g2 = self.node_g2
        node_tmp = self.node_tmp
        g_desc = node_g1.desc(sdfg)
        tmp_desc = node_tmp.desc(sdfg)

        if node_g1.data != node_g2.data:
            return False
        assert not g_desc.transient
        if not tmp_desc.transient:
            return False
        if graph.scope_dict()[node_g1] is not None:
            return False
        if gtx_transformations.utils.is_view(tmp_desc, sdfg):
            return False
        if gtx_transformations.utils.is_view(g_desc, sdfg):
            return False

        # If there is an empty Memlet connected to any of the tree nodes we give
        #  up. We might be able to handle them, but it is too complicated.
        # TODO(phimuell): Figuring out if we have to check the reads more as this one.
        #   I am pretty sure that this is enough, because the reads to `g1` would
        #   be concurrent, thus there is no guarantee if the path `g1 -> tmp -> g2`
        #   is executed before or after another potential read to `g1`. Thus, I think
        #   that there is no need for further checks.
        if any(
            any(e.data.is_empty() for e in graph.out_edges(node) + graph.in_edges(node))
            for node in [node_g1, node_g2, node_tmp]
        ):
            return False

        # Determine which merging strategy should be used, if any.
        # TODO(phimuell): Figuring out if the strategy somehow has to influence
        #   the checks? I do not think so because `can_be_applied()` essentially
        #   tries to figure out if we can consider the three nodes as a single one.
        #   If we fully merge then every effect (write to global) will materialize
        #   at the same time. But if we merge, then it will materialize again
        #   in two steps, as it was already before. Thus I think we do not need
        #   to consider the different strategies here.
        merging_strategy = self._check_merging_strategy(
            state=graph,
            node_g1=node_g1,
            node_tmp=node_tmp,
            node_g2=node_g2,
        )
        if merging_strategy is None:
            return False

        # To avoid race conditions we require that no other node referring to
        #  `G` is inside this state.
        # NOTE: In case we start supporting the splitting of global AccessNodes we will
        #   have to handle this case, to make sure that the other AccessNodes affect
        #   other memory regions and we can safely perform the merge.
        if any(
            dnode.data == node_g1.data
            for dnode in graph.data_nodes()
            if not (dnode is node_g2 or dnode is node_g1)
        ):
            return False

        # Get the mapping, i.e. subsets on `tmp` to subsets on `g`. The function will
        #  check that every `tmp` subset is translatable to `g` subset, `None` if this
        #  mapping does not exist.
        tmp_to_g_mapping = self._compute_tmp_to_g_mapping(
            state=graph,
            node_g1=node_g1,
            node_tmp=node_tmp,
            node_g2=node_g2,
            check_only=True,
        )
        if tmp_to_g_mapping is None:
            return False

        # Now we check if all writes into G1 and G2 are disjunct, i.e. they kind of
        #  act like one big memory, thus this makes it possible to merge it.
        if self._check_read_write_conflicts(
            state=graph,
            node_g1=node_g1,
            node_tmp=node_tmp,
            node_g2=node_g2,
            tmp_to_g_mapping=tmp_to_g_mapping,
            merging_strategy=merging_strategy,
        ):
            return False

        # Check if we can remove the transient.
        if not self._is_single_use_data(node_tmp, sdfg):
            return False

        # NOTE: We do not check for "silent writes", these are writes to `g` that
        #   are not visible in `tmp`, for example because they are outside the
        #   subset `tmp` represents. We could do that, but it is a little bit of
        #   work and I am pretty sure, if we would have it we would have an
        #   invalid, in the sense of GT4Py, SDFG to begin with.

        return True

    def _check_read_write_conflicts(
        self,
        state: dace.SDFGState,
        node_g1: dace_nodes.AccessNode,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
        merging_strategy: int,
    ) -> bool:
        """Checks if the merging would cause a data access conflict.

        The function returns `True` if there is a conflict and `False` if there is
        none. It is important that these cases are very bizarre and are here mostly
        for completeness.
        """
        # What is written into `g1`; described in terms of `g`.
        all_writes_into_g1 = gtx_dace_split.describe_incoming_edges(state, node_g1)

        # What is written into `g2`, that does not come from `tmp` (checked
        #  separately) or from `g1` (according to ADR-18 can only be the same
        #  position). See it as "pure writes"; described in terms of `g`.
        pure_writes_into_g2 = [
            gtx_dace_split.describe_edge(edge, incoming_edge=True)
            for edge in state.in_edges(node_g2)
            if edge.src not in [node_tmp, node_g1]
        ]

        # Check for conflicts that are generated by writes into `g1` that conflicts
        #  with writes into `g2`.
        # NOTE: We only consider it a conflict if we can show that it intersects.
        for write_into_g1 in all_writes_into_g1:
            write_write_conflicts = [
                write_into_g2
                for write_into_g2 in pure_writes_into_g2
                if gtx_dace_split.are_intersecting(write_into_g2.subset, write_into_g1.subset)
            ]
            if len(write_write_conflicts) != 0:
                return True

        # What is read from `g1`, ignoring the transfers to `tmp`, see them as
        #  "pure reads"; described in terms of `g`.
        pure_g1_reads = [
            gtx_dace_split.describe_edge(edge, incoming_edge=False)
            for edge in state.out_edges(node_g1)
            if edge.dst is not node_tmp
        ]

        # Check if we read something from `g1` that is later overwritten by an
        #  updating write to `g2`.
        # NOTE: We only consider it a conflict if we can show that it intersects.
        for write_into_g2 in pure_writes_into_g2:
            write_after_read_conflicts = [
                conflicting_read
                for conflicting_read in pure_g1_reads
                if gtx_dace_split.are_intersecting(conflicting_read.subset, write_into_g2.subset)
            ]
            if len(write_after_read_conflicts) != 0:
                return True

        if len(all_writes_into_g1) == 0 and len(pure_writes_into_g2) == 0:
            return False

        # What is written into `t`, that does not come from `g1`. However, the
        #  subset is expressed in g terms.
        if merging_strategy == self._MERGE_TMP_G2:
            # In the partial merging strategy we ignore all reads from `g1`, the
            #  direct ones and indirects, i.e. the one that read from `g1` compute
            #  something and then write to `t`. This is a simplification.
            pure_writes_into_t = [
                self._compute_transfromed_subset(tdesc, tmp_to_g_mapping)
                for tdesc in gtx_dace_split.describe_incoming_edges(state, node_tmp)
                if not gtx_transformations.utils.is_source_node_of(
                    sink=tdesc.other_node,
                    possible_sources=node_g1,
                    state=state,
                )
            ]

        elif merging_strategy == self._FULL_MERGE:
            # In the full merging strategy we ignore all direct transfers between `g1` and `t`.
            pure_writes_into_t = [
                self._compute_transfromed_subset(tdesc, tmp_to_g_mapping)
                for tdesc in gtx_dace_split.describe_incoming_edges(state, node_tmp)
                if tdesc.other_node is not node_g1
            ]

        else:
            raise NotImplementedError(f"Merging strategy {merging_strategy} is not implemented.")

        # Test for write conflicts that arise because we write something into `t`
        #  that conflicts with a write to either `g1` or `g2`.
        # NOTE: We only consider it a conflict if we can show that it intersects.
        for writes_into_t_wrt_g in pure_writes_into_t:
            write_write_conflicts_g1 = [
                conflicting_g1_write
                for conflicting_g1_write in all_writes_into_g1
                if gtx_dace_split.are_intersecting(writes_into_t_wrt_g, conflicting_g1_write.subset)
            ]
            if len(write_write_conflicts_g1) != 0:
                return True

            write_write_conflicts_g2 = [
                conflicting_g2_write
                for conflicting_g2_write in pure_writes_into_g2
                if gtx_dace_split.are_intersecting(writes_into_t_wrt_g, conflicting_g2_write.subset)
            ]
            if len(write_write_conflicts_g2) != 0:
                return True

        return False

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        node_g1 = self.node_g1  # Have to be kept alive!
        node_tmp = self.node_tmp
        node_g2 = self.node_g2

        merging_strategy = self._check_merging_strategy(
            state=graph,
            node_g1=node_g1,
            node_tmp=node_tmp,
            node_g2=node_g2,
        )
        assert merging_strategy is not None

        tmp_to_g_mapping = self._compute_tmp_to_g_mapping(
            state=graph,
            node_g1=node_g1,
            node_tmp=node_tmp,
            node_g2=node_g2,
            check_only=False,
        )

        if merging_strategy == self._FULL_MERGE:
            self._full_merge_implementation(
                state=graph,
                sdfg=sdfg,
                tmp_to_g_mapping=tmp_to_g_mapping,
            )

        elif merging_strategy == self._MERGE_TMP_G2:
            self._merge_tmp_into_g2_implementation(
                state=graph,
                sdfg=sdfg,
                tmp_to_g_mapping=tmp_to_g_mapping,
            )

        else:
            raise NotImplementedError(f"The strategy '{merging_strategy}' is unknown.")

    def _merge_tmp_into_g2_implementation(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
    ) -> None:
        """Merges `tmp` and `g2` together and remove all links with `g1`.

        This is the implementation of the `_MERGE_TMP_G2` strategy. The
        function will keep `g1` and `g2` alive, but without a direct
        connection. `tmp` will be merged with `g2` and it will be removed
        from the SDFG.
        """
        node_g1 = self.node_g1
        node_g2 = self.node_g2
        node_tmp = self.node_tmp

        # As a first step we will remove all direct edges between `g1` and
        #  the other two nodes, and then we are done with it. The node should
        #  never become isolated.
        for oedge in state.out_edges(node_g1):
            if oedge.dst is node_tmp or oedge.dst is node_g2:
                state.remove_edge(oedge)
        assert state.degree(node_g1) != 0

        # These are all (reads and writes) that we have to relocate from `tmp` to `g2`.
        edges_to_relocate_desc = [
            desc
            for desc in gtx_dace_split.describe_all_edges(state, node_tmp)
            if desc.other_node is not node_g2
        ]

        # Now we relocate the edges from `tmp` to `g2`.
        already_reconfigured_dataflow: set[tuple[dace_nodes.Node, str]] = set()
        self._merge_tmp_node_with_a_g_node(
            state=state,
            sdfg=sdfg,
            node_tmp=node_tmp,
            node_g=node_g2,
            tmp_to_g_mapping=tmp_to_g_mapping,
            edges_to_relocate_desc=edges_to_relocate_desc,
            already_reconfigured_dataflow=already_reconfigured_dataflow,
        )

        # After the temporary node has been isolated, remove the node and
        #  the data descriptor.
        assert state.in_degree(node_tmp) == 0
        assert all(oedge.dst is node_g2 for oedge in state.out_edges(node_tmp))
        state.remove_node(node_tmp)
        sdfg.remove_data(node_tmp.data, validate=False)

        # In this mode the `node_g2` should never become isolated, we now also
        #  perform stride propagation from it, because of the edges that no
        #  longer refer to `tmp`.
        # TODO(phimuell): Use the edges here to restrict propagation only to
        #   the minimum.
        assert state.degree(node_g2) != 0
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=state,
            outer_node=node_g2,
        )

    def _full_merge_implementation(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
    ) -> bool:
        """Merges the three nodes together, such that only `g2` remains.

        The function implements the case for the `_FULL_MERGE` case.
        It will redirect all writes from/to `tmp` and `g1` to `g2`
        (with the exception of the copies between them). It will then
        remove `node_g1` and `node_tmp` and remove `tmp` from the
        registry. Furthermore it will perform stride propagation.

        In certain cases `node_g2` will be isolated, in that case the
        function will remove that node from the SDFG as well and it
        will return `False`. In case `node_g2` has not become isolated
        the function will return `True`.
        """
        node_g1 = self.node_g1
        node_g2 = self.node_g2
        node_tmp = self.node_tmp

        # First we relocate the reads and writes from/to `g1`, that are not
        #  needed to copy redundant information, to `g2`.
        #  This will isolate the `g1` node and we can remove it.
        for iedge in list(state.in_edges(node_g1)):
            state.add_edge(
                iedge.src,
                iedge.src_conn,
                node_g2,
                iedge.dst_conn,
                iedge.data,
            )
            state.remove_edge(iedge)
        for oedge in list(state.out_edges(node_g1)):
            if oedge.dst is node_g2 or oedge.dst is node_tmp:
                pass
            elif oedge.dst is not node_tmp:
                state.add_edge(
                    node_g2,
                    oedge.src_conn,
                    oedge.dst,
                    oedge.dst_conn,
                    oedge.data,
                )
            state.remove_edge(oedge)
        assert state.degree(node_g1) == 0
        state.remove_node(node_g1)

        # These are all (reads and writes) that we have to relocate from `tmp` to `g2`.
        edges_to_relocate_desc = [
            desc
            for desc in gtx_dace_split.describe_all_edges(state, node_tmp)
            if desc.other_node is not node_g2
        ]

        # Now we relocate the edges from `tmp` to `g2`.
        already_reconfigured_dataflow: set[tuple[dace_nodes.Node, str]] = set()
        self._merge_tmp_node_with_a_g_node(
            state=state,
            sdfg=sdfg,
            node_tmp=node_tmp,
            node_g=node_g2,
            tmp_to_g_mapping=tmp_to_g_mapping,
            edges_to_relocate_desc=edges_to_relocate_desc,
            already_reconfigured_dataflow=already_reconfigured_dataflow,
        )

        # After the temporary node has been isolated, remove the node and
        #  the data descriptor.
        assert state.in_degree(node_tmp) == 0
        assert all(oedge.dst is node_g2 for oedge in state.out_edges(node_tmp))
        state.remove_node(node_tmp)
        sdfg.remove_data(node_tmp.data, validate=False)

        # In certain cases it will happen that `g2` has become isolated. In
        #  that case we have to delete it, otherwise we have to propagate
        #  the strides.
        if state.degree(node_g2) != 0:
            gtx_transformations.gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=state,
                outer_node=node_g2,
            )
            return True
        else:
            state.remove_node(node_g2)
            return False

    def _merge_tmp_node_with_a_g_node(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
        node_tmp: dace_nodes.AccessNode,
        node_g: dace_nodes.AccessNode,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
        edges_to_relocate_desc: Sequence[gtx_dace_split.EdgeConnectionSpec],
        already_reconfigured_dataflow: set[tuple[dace_nodes.Node, str]],
    ) -> list[dace_graph.MultiConnectorEdge]:
        """This function merges `tmp_node` with one of two global nodes.

        The function will relocate all edges, that are listed in
        `edges_to_relocate_desc` to `node_g` and rewrite the subset as described
        by `tmp_to_g_mapping`. The function will update the dataflow.

        Note:
            `edges_to_relocate_desc` can not contain edges that connects `tmp_node`
            with a global node.
        """

        # Now we have to relocate the reads/writes involving `tmp` to `g`. This is
        #  a bit more complicated as the two might have different sizes and we have
        #  to reconfigure the dataflow at the other side.
        new_edges: list[dace_graph.MultiConnectorEdge] = []
        for edge_to_relocate_desc in edges_to_relocate_desc:
            # Get the offset for the edge correction
            ss_offset = self._compute_offset(
                tdesc=edge_to_relocate_desc,
                tmp_to_g_mapping=tmp_to_g_mapping,
                check_only=False,
            )

            if gtx_dace_split.describes_incoming_edge(edge_to_relocate_desc):
                is_producer_edge = True
                relocate_key = (edge_to_relocate_desc.edge.src, edge_to_relocate_desc.edge.src_conn)
            else:
                is_producer_edge = False
                relocate_key = (edge_to_relocate_desc.edge.dst, edge_to_relocate_desc.edge.dst_conn)

            new_edge = gtx_transformations.utils.reroute_edge(
                is_producer_edge=is_producer_edge,
                current_edge=edge_to_relocate_desc.edge,
                ss_offset=ss_offset,
                state=state,
                sdfg=sdfg,
                old_node=node_tmp,
                new_node=node_g,
            )
            new_edges.append(new_edge)

            if relocate_key not in already_reconfigured_dataflow:
                already_reconfigured_dataflow.add(relocate_key)
                # Stride propagation is done later.
                gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                    is_producer_edge=is_producer_edge,
                    new_edge=new_edge,
                    ss_offset=ss_offset,
                    state=state,
                    sdfg=sdfg,
                    old_node=node_tmp,
                    new_node=node_g,
                )

            state.remove_edge(edge_to_relocate_desc.edge)

        return new_edges

    @overload
    def _compute_tmp_to_g_mapping(
        self,
        state: dace.SDFGState,
        node_g1: dace_nodes.AccessNode,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        check_only: Literal[True],
    ) -> Union[None, dict[dace_sbs.Subset, dace_sbs.Subset]]: ...

    @overload
    def _compute_tmp_to_g_mapping(
        self,
        state: dace.SDFGState,
        node_g1: dace_nodes.AccessNode,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        check_only: Literal[False],
    ) -> dict[dace_sbs.Subset, dace_sbs.Subset]: ...

    def _compute_tmp_to_g_mapping(
        self,
        state: dace.SDFGState,
        node_g1: dace_nodes.AccessNode,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
        check_only: bool,
    ) -> Union[None, dict[dace_sbs.Subset, dace_sbs.Subset]]:
        """Computes a mapping that describes how `tmp` maps into `g`.

        The function returns a `dict`, that maps subsets of the `tmp_node`
        to the corresponding subset at the `g`. If this mapping does
        not exists then the function returns `None`. Each subsets
        can be seen as a part of `g`.

        The function has two modes, that are selected by specifying
        `check_only`. If it is `True` then the function assumes that
        it is called in the context of `can_be_applied()`. In that
        mode the function might return `None` to indicate that the
        mapping does not exits. In this mode the function will also
        make sure that everything that is written into `tmp`, can
        be translated to a subset inside `g`.

        If `check_only` is `False` then the function assumes that it
        is called by `apply()` and will only compute the mapping
        and skip some tests.

        Note:
            - The function must be called before any modification, which
                is only important for `check_only=False`.
            - This function is currently not able to detect the case
                `tmp[0:5] -> [5:10]` and `tmp[6:10] -> [0:5]` however, this
                should not be important for GT4Py.
        """

        # TODO(phimuell): Rewrite this function such that the mapping is computed
        #   from G.

        # This is the set of edges that link `tmp` with `g`. We need it to understand
        #  how we have to rewrite the writes to `tmp` such that they go through `g`
        #  instead. Note that the description here is always in terms of `tmp`.
        t_descriptions: list[gtx_dace_split.EdgeConnectionSpec] = []

        # The first source of `t_descriptions` are the edges that copies data from
        #  `tmp` to `g2`, i.e. write back. But they are not the only ones.
        # NOTE: It is important that the subset mapping is only build by the edges
        #   between `tmp` and `g2`. Because it is used by `_check_read_write_conflicts()`
        #   to identify conflicts!
        t_descriptions.extend(
            gtx_dace_split.describe_edge(e, False) for e in state.edges_between(node_tmp, node_g2)
        )

        # The second source of information are the initial writes from `g1` to `tmp`.
        #  However, we have to be carefully here, as it might be already included
        #  in the description trough the write back edges.
        for transfer_edge in state.edges_between(node_g1, node_tmp):
            transfer_g1_desc_t = gtx_dace_split.describe_edge(transfer_edge, incoming_edge=True)

            # Test if the edge writes into `tmp` at a location that is not yet known,
            #  i.e. is not written back, if not known add it.
            # TODO(phimuell): We use intersection here, because cover would be too
            #   liberal. However, what we should actually do is, decompose the
            #   subset and only add the parts that are not yet known.
            if not any(
                gtx_dace_split.are_intersecting(known_t_patch.subset, transfer_g1_desc_t.subset)
                for known_t_patch in t_descriptions
            ):
                t_descriptions.append(transfer_g1_desc_t)

        assert not any(tp.other_subset is None for tp in t_descriptions)

        # Now merge all subsets we found together. This describes at `tmp`.
        # TODO(phimuell): Figuring out what it means if the merge does not
        #   result in a single subset. I think that the tests are strong
        #   enough to handle that case.
        merged_subsets_at_tmp = gtx_dace_split.subset_merger(t_descriptions)

        # Now also merges the subsets at the `g` side, but we have to do it
        #  in the same way. Thus we first find out which edges, were merged
        #  together and then we merge their `other_subset`.
        subset_map: dict[dace_sbs.Subset, dace_sbs.Subset] = {}
        for merged_subset_at_tmp in merged_subsets_at_tmp:
            merged_subset_at_g2 = gtx_dace_split.subset_merger(
                [
                    desc.other_subset
                    for desc in t_descriptions
                    if merged_subset_at_tmp.covers(desc.subset)
                ]
            )

            # If we do not find exactly one subset then it is not consecutive.
            #  An example would be `tmp[0:5] -> g[0:5]` and `tmp[5:10] -> g[6:11]`.
            #  However, the case `tmp[0:5] -> g[5:10]` and `tmp[5:10] -> g[0:5]`
            #  is not detected.
            if len(merged_subset_at_g2) != 1:
                if not check_only:
                    raise RuntimeError("Found indeterministic behaviour.")
                return None

            # Ensure that there is no intersection between the subsets.
            #  Handles "unable to compare" as not an intersection.
            assert not any(
                gtx_dace_split.are_intersecting(merged_subset_at_tmp, known_tmp_subset) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
                for known_tmp_subset in subset_map.keys()
            )
            assert not any(
                gtx_dace_split.are_intersecting(merged_subset_at_g2[0], known_g_subset) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
                for known_g_subset in subset_map.values()
            )

            subset_map[merged_subset_at_tmp] = merged_subset_at_g2[0]

        if check_only:
            # Make sure that the `tmp` subset of any every edge that read from/write to
            #  `tmp` (ignoring the connections to and from `G`) can be translated into
            #  a `G` subset.
            for edge_desc in gtx_dace_split.describe_all_edges(state, node_tmp):
                if edge_desc.other_node in [node_g1, node_g2]:
                    continue
                if not self._compute_offset(
                    tdesc=edge_desc,
                    tmp_to_g_mapping=subset_map,
                    check_only=True,
                ):
                    return None

            # Ensure that the writes to `tmp` are also written back to `g2`. This is
            #  needed to ensure that `tmp` is not only used as buffer to discard some
            #  writes.
            # TODO(phimuell): Improve this, maybe also checking with swapped order
            #   for `covered()`?
            pure_tmp_writes = [
                gtx_dace_split.describe_edge(edge, incoming_edge=True)
                for edge in state.in_edges(node_tmp)
                if edge.src is not node_g1
            ]
            for tmp_write in pure_tmp_writes:
                if not any(
                    tmp_to_g2_transfer.covers(tmp_write.subset)
                    for tmp_to_g2_transfer in subset_map.keys()
                ):
                    return None

        return subset_map

    def _is_single_use_data(
        self,
        access_node: dace_nodes.AccessNode,
        sdfg: dace.SDFG,
    ) -> bool:
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        return access_node.data in single_use_data[sdfg]

    def _check_merging_strategy(
        self,
        state: dace.SDFGState,
        node_g1: dace_nodes.AccessNode,
        node_tmp: dace_nodes.AccessNode,
        node_g2: dace_nodes.AccessNode,
    ) -> Union[None, int]:
        """Tests which merging strategy should be used.

        By default the transformation tries to merge the three nodes together,
        but this is not always possible, if this would create cycles.

        Currently the following modes are implemented:
        - `_FULL_MERGE`: In this mode all three nodes are combined together.
            This means that there are no cycles to begin with.
        - `_MERGE_TMP_G2`: In this mode the `tmp` and the `g2` mode are merged.
            Thus the `g1` node will remain.

        If the cycle can not be resolved the function will return `None`.

        Todo:
            Handle more cases.
        """

        found_cycle_from_g1 = False
        for oedge in state.out_edges(node_g1):
            if oedge.dst is node_tmp:
                continue
            if oedge.dst is node_g2:
                continue
            if gtx_transformations.utils.is_reachable(start=oedge.dst, target=node_g2, state=state):
                found_cycle_from_g1 = True
                break

        if not found_cycle_from_g1:
            return self._FULL_MERGE

        # We found a cycle from `g1` we will now check if we can merge `tmp` and `g2`.
        #  For this we start a cycle test from `tmp`.
        found_cycle_from_tmp = False
        for oedge in state.out_edges(node_tmp):
            if oedge.dst is node_g2:
                continue
            if gtx_transformations.utils.is_reachable(start=oedge.dst, target=node_g2, state=state):
                found_cycle_from_tmp = True
                break

        if not found_cycle_from_tmp:
            return self._MERGE_TMP_G2

        return None

    @overload
    def _compute_offset(
        self,
        tdesc: gtx_dace_split.EdgeConnectionSpec,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
        check_only: Literal[True],
    ) -> bool: ...

    @overload
    def _compute_offset(
        self,
        tdesc: gtx_dace_split.EdgeConnectionSpec,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
        check_only: Literal[False],
    ) -> list[dace_sym.SymbolicType]: ...

    def _compute_offset(
        self,
        tdesc: gtx_dace_split.EdgeConnectionSpec,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
        check_only: bool,
    ) -> Union[list[dace_sym.SymbolicType], bool]:
        """Computes the offset to turn a subset described in terms of `tmp` into a `g`.

        `tdesc` describes an edge that interacts with `tmp`, the function returns
        the offset that describes the how to turn the subset of `tmp` into one
        that describes on `g`.
        """

        # First we have to identify the which patch on the `tmp` side describes
        #  the matching. One idea would be to use `cover()`, however, there are
        #  some rare cases where this is not possible. For that reason we use
        #  the lower bound. This might be a bit dangerous.
        lower_bound = dace_sbs.Indices(tdesc.subset.min_element())
        associated_tmp_subset: Optional[dace_sbs.Subset] = next(
            (
                merged_tmp_subset
                for merged_tmp_subset in tmp_to_g_mapping.keys()
                if gtx_dace_split.are_intersecting(merged_tmp_subset, lower_bound)
            ),
            None,
        )
        if check_only:
            # NOTE: We could also test if the translation is unique.
            return associated_tmp_subset is not None
        assert associated_tmp_subset is not None

        # Compute the offset between the lower bound of the t patch to the
        #  g patch. This is also the offset of the `tdesc` that we have to
        #  translate.
        associated_g_subset = tmp_to_g_mapping[associated_tmp_subset]
        tmp_subset_start = associated_tmp_subset.min_element()
        g_subset_start = associated_g_subset.min_element()
        ss_offset = [
            g_patch_start - tmp_start
            for g_patch_start, tmp_start in zip(g_subset_start, tmp_subset_start)
        ]
        return ss_offset

    def _compute_transfromed_subset(
        self,
        tdesc: gtx_dace_split.EdgeConnectionSpec,
        tmp_to_g_mapping: dict[dace_sbs.Subset, dace_sbs.Subset],
    ) -> dace_sbs.Range:
        """Transform the subset at `t` to the corresbonding subset at at `g`."""
        return dace_sbs.Range(
            [
                (ss_start + ss_offset, ss_start + ss_offset + ss_size - 1, 1)
                for ss_start, ss_offset, ss_size in zip(
                    tdesc.subset.min_element(),
                    self._compute_offset(tdesc, tmp_to_g_mapping, False),
                    tdesc.subset.size(),
                )
            ]
        )


@dace_properties.make_properties
class SingleStateGlobalDirectSelfCopyElimination(dace_transformation.SingleStateTransformation):
    """Removes an unbuffered self copy in a state.

    This transformation is extremely similar to `SingleStateGlobalSelfCopyElimination`,
    however, this transformation does not need a buffer between the two global
    AccessNodes. Therefore it matches the pattern `(G) -> (G)`, where `G` refers to
    global data. Note that the transformation only considers copies on global scope.
    The transformation has two modes in how this is achieved, which one is chosen,
    depends on if cycles are created or not. The first mode, which is selected if
    cycles would be created, is to merge the two AccessNodes together. For this
    the edges of the first AccessNode are reconnected to the second AccessNode.

    In the second mode, which is selected if cycles would be created, for example in
    case of `if` branches, then the edges are removed, which will create two
    independent `G` AccessNodes.

    Todo:
        Merge this transformation with `SingleStateGlobalSelfCopyElimination`.
    """

    node_read_g = dace_transformation.PatternNode(dace_nodes.AccessNode)
    node_write_g = dace_transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.node_read_g, cls.node_write_g)]

    def can_be_applied(
        self,
        graph: dace.SDFGState | dace.SDFG,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        read_g = self.node_read_g
        write_g = self.node_write_g
        g_desc = read_g.desc(sdfg)

        # NOTE: We do not check if `G` is read downstream.
        if read_g.data != write_g.data:
            return False
        if g_desc.transient:
            return False
        # Restrict to global scope.
        if graph.scope_dict()[read_g] is not None:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState | dace.SDFG,
        sdfg: dace.SDFG,
    ) -> None:
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        # Select in which mode the transformation should run, depending on
        #  if this would lead to cycles.
        merge_g_nodes = True
        for oedge in graph.out_edges(read_g):
            if oedge.dst is write_g:
                continue
            if gtx_transformations.utils.is_reachable(
                start=oedge.dst,
                target=write_g,
                state=graph,
            ):
                merge_g_nodes = False
                break

        if merge_g_nodes:
            self._merge_g_nodes(state=graph, sdfg=sdfg)
        else:
            self._split_g_nodes(state=graph)

    def _split_g_nodes(
        self,
        state: dace.SDFGState,
    ) -> None:
        """Second mode of the transformation, i.e. the splitting.

        All edges between the two nodes, will be removed. There is no longer a
        direct connection between the two nodes.
        """
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        for oedge in list(state.out_edges(read_g)):
            if oedge.dst is write_g:
                state.remove_edge(oedge)

        assert state.degree(read_g) != 0
        assert state.degree(write_g) != 0

    def _merge_g_nodes(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """The first mode of the transformation, i.e. the merging of the nodes.

        All incoming edges of the first node will be reconnected to the second node.
        The outgoing edges will be reconnected to the second node, if they connect
        the two nodes, they will be removed.
        """
        read_g: dace_nodes.AccessNode = self.node_read_g
        write_g: dace_nodes.AccessNode = self.node_write_g

        # Reconnect any incoming edges of the read AccessNode, aka. the first one.
        for iedge in list(state.in_edges(read_g)):
            # Since the names are the same, just creating a new edge is enough.
            state.add_edge(
                iedge.src,
                iedge.src_conn,
                write_g,
                iedge.dst_conn,
                iedge.data,
            )
            state.remove_edge(iedge)

        # Now reconnect all edges that leave the first AccessNodes, with the
        #  exception of the edges that connecting the two, they are just removed.
        for oedge in list(state.out_edges(read_g)):
            if oedge.dst is not write_g:
                state.add_edge(
                    write_g,
                    oedge.src_conn,
                    oedge.dst,
                    oedge.dst_conn,
                    oedge.data,
                )
            state.remove_edge(oedge)

        # Now remove the first node to `G` and if the second has become isolated
        #  then also remove it.
        state.remove_node(read_g)

        if state.degree(write_g) == 0:
            state.remove_node(write_g)

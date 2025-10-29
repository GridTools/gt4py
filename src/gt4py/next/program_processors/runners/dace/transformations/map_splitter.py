# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_utils as gtx_mfutils,
    splitting_tools as gtx_stools,
)


@dace_properties.make_properties
class MapSplitter(dace_transformation.SingleStateTransformation):
    """Split a Map based on a transient.

    The transformation matches the following pattern `MapExit -> (AccessNode)`,
    where `AccessNode` needs to be a transient single use data whose only producer
    is the matched Map and has only a single consumer. Furthermore, more data
    is written into `AccessNode` than is read from it. In that case the
    producing Map and `AccessNode` is split into fragments, such that a fragment
    either generates data that is fully consumed or not at all.
    Fragments that generate data that is not used are removed, unless
    `remove_dead_dataflow` is set to `False`.

    Args:
        removed_dead_dataflow: If `True`, the default, remove dataflow that is not
            read anywhere.
        single_use_data: Use this to classify single use data, see `dace.FindSingleUseData()`.

    Note:
        - The "more data is written into `AccessNode` than read from it" technically
            violates ADR18, but is the result of some broadcast expressions in
            relation to `concat_where`.
    """

    map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    remove_dead_dataflow = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="Remove dead dataflow directly.",
    )

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_exit, cls.access_node)]

    def __init__(
        self,
        remove_dead_dataflow: Optional[bool] = None,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._single_use_data = single_use_data
        if remove_dead_dataflow is not None:
            self.remove_dead_dataflow = remove_dead_dataflow
        super().__init__(*args, **kwargs)

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        # We assume that the Map only generates a single data.
        map_exit: dace_nodes.MapExit = self.map_exit
        if graph.out_degree(map_exit) != 1:
            return False

        # The AccessNode must be single use data and have only one producer. Single use
        #  data is tested at the very end, to avoid computing it too often.
        # NOTE: Requiring that `self.access_node` is a transient is mainly to simplify
        #   the implementation, but the transient should always exists anyway.
        access_node: dace_nodes.AccessNode = self.access_node
        ac_desc: dace_data.Data = access_node.desc(sdfg)
        if not ac_desc.transient:
            return False
        if graph.in_degree(access_node) != 1:
            return False

        # For simplicity we assume that the AccessNode only has a single consumer.
        #  This makes handling a bit simpler, as we do not have to compose the
        #  subset that is read.
        # TODO(phimuell): Remove this requirement.
        if graph.out_degree(access_node) != 1:
            return False

        map_ac_edge = next(iter(graph.in_edges(access_node)))
        ac_consumer_edge = next(iter(graph.out_edges(access_node)))
        produced_subset: dace_sbs.Range = map_ac_edge.data.get_dst_subset(map_ac_edge, graph)
        consumed_subset: dace_sbs.Range = ac_consumer_edge.data.get_src_subset(
            ac_consumer_edge, graph
        )

        # This simplifies the implementation of the correction.
        # TODO(phimuell): Not sure how useful lifting this restriction would be.
        if len(map_exit.map.params) != produced_subset.dims():
            return False
        if not all(
            (sbs[0] == 0) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            for sbs in produced_subset
        ):
            return False
        if not all(
            (psize == msize) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            for psize, msize in zip(produced_subset.size(), map_exit.map.range.size())
        ):
            return False

        # In order for this transformation to make sense, we require that more data is
        #  written to the AccessNode than read from it.
        if consumed_subset == produced_subset:
            return False
        if not produced_subset.covers(consumed_subset):
            return False

        # We now test if the AccessNode is a single used data. We do it here such that
        #  we can postpone the scanning of the SDFG as long as possible.
        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data
        if access_node.data not in single_use_data[sdfg]:
            return False

        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG) -> None:
        map_exit: dace_nodes.MapExit = self.map_exit
        map_entry: dace_nodes.MapEntry = graph.entry_node(map_exit)
        access_node: dace_nodes.AccessNode = self.access_node

        map_ac_edge = next(iter(graph.in_edges(access_node)))
        ac_consumer_edge = next(iter(graph.out_edges(access_node)))
        assert graph.in_degree(access_node) == 1 and graph.out_degree(access_node) == 1

        produced_subset: dace_sbs.Range = map_ac_edge.data.get_dst_subset(map_ac_edge, graph)
        consumed_subset: dace_sbs.Range = ac_consumer_edge.data.get_src_subset(
            ac_consumer_edge, graph
        )

        # Now decompose the production subset, we will use this information to "split"
        #  the Map.
        # NOTE: Name swapping is intentional.
        split_production_subsets = gtx_stools.decompose_subset(
            consumer=produced_subset, producer=consumed_subset
        )
        assert split_production_subsets is not None and len(split_production_subsets) > 1

        # Now create new Maps of the appropriate ranges.
        # NOTE: `split_production_subsets` is only based on what is accessed at
        #   `self.access_node`, however, we want to split the Map based on it.
        #   The main problem is if there are offsets, i.e. `output[__i - 3]`.
        #   To solve this we realize that while the actual decomposed ranges are
        #   meaningless, they still have the same size. Thus we simply correct them.
        map_start_index = [map_exit.map.range[i][0] for i in range(len(map_exit.map.range))]
        new_map_exits: list[dace_nodes.MapExit] = []
        for i, production_subset in enumerate(split_production_subsets):
            sub_me, sub_mx = gtx_mfutils.copy_map_graph_with_new_range(
                sdfg=sdfg,
                state=graph,
                map_entry=map_entry,
                map_exit=map_exit,
                map_range=production_subset.offset_new(map_start_index, negative=False),
                suffix=str(i),
            )
            # `copy_map_graph_with_new_range()` just copies all Memlets, thus the
            #  subsets that are read and written by the new Maps are still the same
            #  as the original one. We must fix that otherwise `SplitAccessNode` will
            #  fail.
            # TODO(phimuell): Find out how to limit propagation only to the Maps that
            #   are not dead dataflow, in case we remove them anyway.
            dace.sdfg.propagation.propagate_memlets_map_scope(
                sdfg=sdfg,
                state=graph,
                map_entry=sub_me,
            )
            new_map_exits.append(sub_mx)
        gtx_mfutils.delete_map(graph=graph, map_entry=map_entry, map_exit=map_exit)

        # Now perform the split of `self.access_node`. Parts that are not read will
        #  later be eliminated by dead dataflow elimination.
        gtx_transformations.SplitAccessNode.apply_to(
            sdfg=sdfg,
            options={
                "assume_single_use_data": True,  # Was tested in `can_be_applied()`.
            },
            verify=True,
            access_node=access_node,
        )

        if not self.remove_dead_dataflow:
            return

        # Because `self.access_node` is single use data, we know that if the split
        #  node is not directly used, i.e. has an output degree of 0, that it is dead
        #  dataflow and we can remove it.
        for mx in new_map_exits:
            output_node = next(iter(graph.out_edges(mx))).dst
            assert graph.out_degree(mx) == 1
            assert (
                isinstance(output_node, dace_nodes.AccessNode) and output_node.desc(sdfg).transient
            )

            if graph.out_degree(output_node) == 0:
                gtx_transformations.gt_remove_map(
                    sdfg=sdfg,
                    state=graph,
                    map_entry=graph.entry_node(mx),
                    # We would need to update single use data to really use it.
                    remove_unused_data=False,
                )

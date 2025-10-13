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
class MapSpliter(dace_transformation.SingleStateTransformation):
    """ """

    map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_exit, cls.access_node)]

    def __init__(
        self,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._single_use_data = single_use_data
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
        produced_subset: dace_sbs.Range = map_ac_edge.data.get_dst_subset(map_ac_edge, graph)

        ac_consumer_edge = next(iter(graph.out_edges(access_node)))
        consumed_subset: dace_sbs.Range = ac_consumer_edge.data.get_src_subset(
            ac_consumer_edge, graph
        )

        # That this transformation makes sense, we require that more data is written
        #  to the AccessNode than read from it.
        if consumed_subset == produced_subset:
            return False
        if not produced_subset.covers(consumed_subset):
            return False

        # Test for single use data here such that we can maybe avoid it.
        if self._single_use_data is None:
            # This is actually to much, there should be a simple function to answer the
            #  question for a single data.
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
        assert graph.in_degree(access_node) == 1
        ac_consumer_edge = next(iter(graph.out_edges(access_node)))
        assert graph.out_degree(access_node) == 1

        produced_subset: dace_sbs.Range = map_ac_edge.data.get_dst_subset(map_ac_edge, graph)
        consumed_subset: dace_sbs.Range = ac_consumer_edge.data.get_src_subset(
            ac_consumer_edge, graph
        )

        # TODO: Explain why?
        map_subranges = gtx_stools.decompose_subset(
            consumer=produced_subset, producer=consumed_subset
        )
        assert map_subranges is not None
        assert len(map_subranges) > 1

        new_sub_map_entries = []
        map_start_index = [map_exit.map.range[i][0] for i in range(len(map_exit.map.range))]
        for i, map_subrange in enumerate(map_subranges):
            sub_me, _ = gtx_mfutils.copy_map_graph_with_new_range(
                sdfg=sdfg,
                state=graph,
                map_entry=map_entry,
                map_exit=map_exit,
                # TODO(phimuell): Explain why.
                map_range=map_subrange.offset_new(map_start_index, negative=False),
                suffix=str(i),
            )
            new_sub_map_entries.append(sub_me)
        gtx_mfutils.delete_map(graph=graph, map_entry=map_entry, map_exit=map_exit)

        # TODO(phimuell): WHy here and not in loop above.
        for sub_me in new_sub_map_entries:
            dace.sdfg.propagation.propagate_memlets_map_scope(
                sdfg=sdfg,
                state=graph,
                map_entry=sub_me,
            )

        # Call the splitting node transformation on the access node.
        # TODO(phimuell): Create a free function for this.
        gtx_transformations.SplitAccessNode.apply_to(
            sdfg=sdfg,
            options={
                "assume_single_use_data": True,  # Was tested in `can_be_applied()`.
            },
            verify=True,
            access_node=access_node,
        )

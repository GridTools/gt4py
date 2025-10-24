# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Union

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class MapToCopy(dace_transformation.SingleStateTransformation):
    """Changes a copy Map into a Memlet.

    The transformation matches the pattern `MapEntry --> Tasklet --> MapExit`. Being
    a copy Map, the MapEntry has exactly one incoming edge that reads from an
    AccessNode (`Input`), the Tasklet is trivial and MapExit writes into an
    AccessNode (`Output`).

    The transformation has two modes. In the first mode the Map is just replaced by
    a Memlet between `Input` and `Output`, thus the copy still takes place.
    The second mode, known as bypass mode `Output` is completely bypassed, i.e.
    all reads from `Output` will now be satisfied directly from `Input`.
    Which mode is used depends on the concrete situation, but in order for the
    bypass mode to be selected the set of single use data has to be passed to the
    transformation at construction.

    Args:
        single_use_data: The result of `SingleUseData`, needs to be passed if
            bypass mode should be used.
    """

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    # Pattern Matching
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    tasklet = dace_transformation.PatternNode(dace_nodes.Tasklet)
    map_exit = dace_transformation.PatternNode(dace_nodes.MapExit)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.map_entry,
                cls.tasklet,
                cls.map_exit,
            )
        ]

    def __init__(
        self,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._single_use_data = single_use_data

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_entry = self.map_entry
        tasklet = self.tasklet
        map_exit = self.map_exit

        # The nodes should only have one input and one output.
        for node in [self.map_entry, self.tasklet, self.map_exit]:
            if graph.out_degree(node) != 1:
                return False
            if graph.in_degree(node) != 1:
                return False

        # Test if the Tasklet is a copy tasklet.
        in_conn = next(iter(tasklet.in_connectors.keys()))
        out_conn = next(iter(tasklet.out_connectors.keys()))
        tasklet_code_without_spaces = "".join(
            char for char in tasklet.code.as_string if not char.isspace()
        )
        if tasklet_code_without_spaces != f"{out_conn}={in_conn}":
            return False

        # Ensure that the Map is continuous.
        if any((step != 1) == True for _, _, step in map_entry.map.range):  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return False

        # Ensure that the Memlets are not "funny", we allow dynamic, but nothing else.
        src_edge = next(iter(graph.in_edges(map_entry)))
        dst_edge = next(iter(graph.out_edges(map_exit)))
        for memlet in [src_edge.data, dst_edge.data]:
            if memlet.wcr is not None:
                return False
            if memlet.allow_oob:
                return False

        # TODO: Should we disallow views?
        if not isinstance(src_edge.src, dace_nodes.AccessNode):
            return False
        if not isinstance(dst_edge.dst, dace_nodes.AccessNode):
            return False
        src_access_node = src_edge.src
        src_subset = src_edge.data.get_src_subset(src_edge, graph)
        dst_access_node = dst_edge.dst
        dst_subset = dst_edge.data.get_dst_subset(dst_edge, graph)

        # Memelts between AccessNodes referring to the same data is only allowed if
        #  they are point wise, thus we have to ensure this.
        if src_access_node.data == dst_access_node.data:
            assert not src_access_node.desc(sdfg).transient
            if src_subset != dst_subset:
                return False

        # This is a primitive test that ensures that we do not perform broadcasts,
        #  i.e. that we have a real copy operation and each element in source goes
        #  to a distinct location in destination.
        # TODO(phimuell): May need to adapt if we have to handle the results of
        #   `concat_where`, but I am not sure how important that case is.
        if isinstance(sdfg.arrays[src_access_node.data], dace_data.Scalar):
            return False
        if (src_subset.num_elements() == src_edge.data.volume) != True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return False
        if (dst_subset.num_elements() == dst_edge.data.volume) != True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return False
        assert all(
            all(
                (step == 1) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
                for _, _, step in sbs
            )
            for sbs in [src_subset, dst_subset]
        )

        return True

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG) -> None:
        map_entry = self.map_entry
        tasklet = self.tasklet
        map_exit = self.map_exit
        src_edge = next(iter(graph.in_edges(map_entry)))
        dst_edge = next(iter(graph.out_edges(map_exit)))

        src_access_node = src_edge.src
        src_subset = src_edge.data.get_src_subset(src_edge, graph)
        dst_access_node = dst_edge.dst
        dst_subset = dst_edge.data.get_dst_subset(dst_edge, graph)

        # Remove the nodes that are no longer needed.
        graph.remove_nodes_from([map_entry, tasklet, map_exit])

        # Decide which scheme should be used.
        #  This mostly depends on if we can handle the case and if it is possible to
        #  bypass and by that removing `dst_access_node`.
        bypass_dst_node = True
        if src_subset.dims() != dst_subset.dims():
            bypass_dst_node = False
        elif self._single_use_data is None:
            bypass_dst_node = False
        elif dst_access_node.data not in self._single_use_data[sdfg]:
            bypass_dst_node = False
        elif graph.in_degree(dst_access_node) != 0:
            bypass_dst_node = False
        elif not dst_access_node.desc(sdfg).transient:
            bypass_dst_node = False
        elif (not src_access_node.desc(sdfg).transient) and any(
            dst_oedge.dst.data == src_access_node.data
            for dst_oedge in graph.out_edges(dst_access_node)
            if isinstance(dst_oedge.dst, dace_nodes.AccessNode)
        ):
            # If by eliminating `dst_access_node` `src_access_node` is directly
            #  connected to a node referencing the same data, we do not allow it.
            #  We could allow it if we made sure that the access is pointwise, i.e.
            #  there is no shift, but currently we do not support that.
            # Important for ADR18 compatibility regarding global self copy.
            bypass_dst_node = False

        if bypass_dst_node:
            # Instead of just connecting `src_access_node` to `dst_access_node` we
            #  fully eliminate `dst_access_node` and all of its consumers directly
            #  read from `src_access_node`.
            offset_correction = [
                sm - dm for sm, dm in zip(src_subset.min_element(), dst_subset.min_element())
            ]
            already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]] = set()
            for old_dst_consumer_edge in graph.out_edges(dst_access_node):
                reconfigure_key = (old_dst_consumer_edge.dst, old_dst_consumer_edge.dst_conn)
                new_edge = gtx_transformations.utils.reroute_edge(
                    is_producer_edge=False,
                    current_edge=old_dst_consumer_edge,
                    ss_offset=offset_correction,
                    state=graph,
                    sdfg=sdfg,
                    old_node=dst_access_node,
                    new_node=src_access_node,
                )
                graph.remove_edge(old_dst_consumer_edge)

                if reconfigure_key not in already_reconfigured_nodes:
                    gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                        is_producer_edge=False,
                        new_edge=new_edge,
                        ss_offset=offset_correction,
                        state=graph,
                        sdfg=sdfg,
                        old_node=dst_access_node,
                        new_node=src_access_node,
                    )
                    already_reconfigured_nodes.add(reconfigure_key)

            # Remove the bypassed node and the associated data.
            assert graph.degree(dst_access_node) == 0
            graph.remove_node(dst_access_node)
            sdfg.remove_data(dst_access_node.data, validate=False)

            # Fixing up the strides.
            gtx_transformations.gt_propagate_strides_from_access_node(
                sdfg=sdfg,
                state=graph,
                outer_node=src_access_node,
            )

        else:
            # We only eliminate the Map but the copy into `dst_access_node` is still
            #  there, this is either because we can not handle it (different
            #  dimensionality) or we need it for consistency.
            graph.add_nedge(
                src_access_node,
                dst_access_node,
                dace.Memlet(
                    data=src_access_node.data,
                    subset=src_subset,
                    other_subset=dst_subset,
                    dynamic=src_edge.data.dynamic or dst_edge.data.dynamic,
                ),
            )

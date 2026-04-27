# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO: Figuring out where to put this file.

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

from gt4py.next.program_processors.runners.dace.library_nodes import broadcast as gtx_dace_broadcast


@dace_properties.make_properties
class InlineBroadcastAccess(dace_transformation.SingleStateTransformation):
    """

    translate:
    ```
        (value_to_broadcast) -> [BroadCast] -> (tmp) -> MapEnty
    ```
    to:
    ```
        (value_to_broadcase) -> MapEntry
    ```

    Todo:
        - Think if it makes sense to allow different consumers than `MapEntry`s.
    """

    bcast_value = dace_transformation.PatternNode(dace_nodes.AccessNode)
    bcast_lib_node = dace_transformation.PatternNode(gtx_dace_broadcast.Broadcast)
    bcast_result = dace_transformation.PatternNode(dace_nodes.AccessNode)
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    clean_dead_dataflow = dace_properties.Property(
        dtype=bool,
        allow_none=False,
        default=True,
        help="Clean dead dataflow.",
    )

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.bcast_value,
                cls.bcast_lib_node,
                cls.bcast_result,
                cls.map_entry,
            )
        ]

    def __init__(
        self,
        *args: Any,
        clean_dead_dataflow: Optional[bool] = None,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
        **kwargs: Any,
    ) -> None:
        self._single_use_data = single_use_data
        if clean_dead_dataflow is None:
            self.clean_dead_dataflow = clean_dead_dataflow

        super().__init__(*args, **kwargs)

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        bcast_lib_node = self.bcast_lib_node
        bcast_value = self.bcast_value
        bcast_result = self.bcast_result
        map_entry = self.map_entry
        bcast_result_desc = bcast_result.desc(sdfg)
        bcast_value_desc = bcast_value.desc(sdfg)
        assert graph.in_edges(bcast_lib_node) == 1
        assert graph.out_edges(bcast_lib_node) == 1

        # NOTE: There is no need to check if the broadcast result is a single use data
        #   or so. We only require that it has only one producer, the data type is the
        #   same and is not global data. We can do that because we do some magic
        #   rewriting. Knowing the single use status of the data is only important for
        #   the `apply()` method.
        if graph.in_degree(bcast_result) != 1:
            return False
        if not bcast_result_desc.transient:
            return False
        if bcast_value_desc.dtype != bcast_result_desc.dtype:
            return False

        # NOTE: As as a simplification, we currently only handle scalars that are broadcast.
        if not isinstance(bcast_value_desc, dace_data.Scalar):
            return False

        # Check the Map's body.
        for outer_map_edge in graph.out_edges(bcast_result):
            # All edges between `bcast_result` and `map_entry` needs to be "okay". So
            #  no empty Memlets or other special cases.
            if outer_map_edge.dst is not map_entry:
                continue
            if outer_map_edge.data.is_empty():
                return False
            if not outer_map_edge.dst_conn.startswith("IN_"):
                return False

            # Inspect the final consumer and handle the case where we fan out.
            inner_connector = "OUT_" + outer_map_edge.dst_conn[3:]
            for inner_map_edge in graph.out_edges_by_connector(map_entry, inner_connector):
                for final_consumer in graph.memlet_tree(inner_map_edge).leaves():
                    sbs_to_inspect = (
                        final_consumer.data.subset
                        if final_consumer.data.data == bcast_result.data
                        else final_consumer.data.other_subset
                    )
                    assert sbs_to_inspect is not None

                    # Currently the only test we do is, that only one element from
                    #  `bcast_result` is loaded. However, the location of this
                    #  element needs to be known. This rejects neighbourhood accesses.
                    if (sbs_to_inspect.num_elements() == 1) == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                        return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        bcast_lib_node = self.bcast_lib_node
        bcast_value = self.bcast_value
        bcast_result = self.bcast_result
        map_entry = self.map_entry

        # Make `bcast_value` available inside the Map body.
        bcast_value_conn: str | None = None
        for edge in graph.out_edges(bcast_value):
            if edge.dst is not map_entry:
                continue
            if edge.data.is_empty():
                continue
            if not edge.dst_conn.startswith("IN_"):
                continue
            bcast_value_conn = "OUT_" + edge.dst_conn[3:]
            break

        else:
            # There was no connection between them so we have to create one.
            bcast_value_conn_raw = map_entry.next_connector(bcast_value.data)
            graph.add_edge(
                bcast_value,
                None,
                map_entry,
                "IN_" + bcast_value_conn_raw,
                dace.Memlet(data=bcast_value.data, subset="0"),
            )
            bcast_value_conn = "OUT_" + bcast_value_conn_raw
            map_entry.add_scope_connectors(bcast_value_conn_raw, force=True)
        assert bcast_value_conn in map_entry.out_connectors

        for outer_map_edge in graph.out_edges(bcast_result):
            if outer_map_edge.dst is not map_entry:
                continue

            inner_connector = "OUT_" + outer_map_edge.dst_conn[3:]
            for inner_map_edge in graph.out_edges_by_connector(map_entry, inner_connector):
                for mtree in graph.memlet_tree(inner_map_edge).traverse_children(True):
                    tree_edge = mtree.edge
                    assert tree_edge.data.wcr is None

                    # Modify the consumer.
                    #  We only have to modify the respective subset and the data attribute,
                    #  no other fancy things are needed. The reason for this is that we
                    #  only operate on scalar reads, thus no strides that needs to be updated.
                    if tree_edge.data.data == bcast_result.data:
                        tree_edge.data.data = bcast_value.data
                        tree_edge.data.subset = dace_sbs.Range.from_string("0")
                    else:
                        tree_edge.data.other_subset = dace_sbs.Range.from_string("0")

                # Now reroute `inner_map_edge` such that it reads from `bcast_value`
                #  directly, which is available from `bcast_value_conn` at `map_entry`.
                graph.add_edge(
                    map_entry,
                    bcast_value_conn,
                    inner_map_edge.dst,
                    inner_map_edge.dst_conn,
                    inner_map_edge.data,  # Was modified above.
                )
                graph.remove_edge(inner_map_edge)

            # Now remove the connection `(bcast_result) -> map_entry`, but keep the node
            map_entry.remove_out_connector(inner_map_edge.src_conn)
            map_entry.remove_in_connector(outer_map_edge.dst_conn)
            graph.remove_edge(outer_map_edge)

        # Check if we can remove the `bcast_result` node.
        if self.clean_dead_dataflow and graph.out_degree(bcast_result) == 0:
            # We have to figuring out
            if self._single_use_data is None:
                find_single_use_data = dace_analysis.FindSingleUseData()
                single_use_data = find_single_use_data.apply_pass(sdfg, None)
            else:
                single_use_data = self._single_use_data

            if bcast_result.data in single_use_data[sdfg]:
                graph.remove_node(bcast_result)
                graph.remove_node(bcast_lib_node)
                sdfg.remove_data(bcast_result.data, validate=__debug__)

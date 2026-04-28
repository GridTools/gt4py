# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
TODO:
    - Figuring out where to put this file.
    - Finalize the broadcast library node implementation. Currently the implementation
        kinds of assume that all information is stored inside the destination subset.
        Furthermore, at least `InlineBroadcastAccess` assumes that there is a source.
"""

import copy
from typing import Any, Optional

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import (
    library_nodes as gtx_lib_nodes,
    transformations as gtx_xtrans,
)


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
    bcast_node = dace_transformation.PatternNode(gtx_lib_nodes.Broadcast)
    bcast_result = dace_transformation.PatternNode(dace_nodes.AccessNode)
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    clean_dead_dataflow = dace_properties.Property(
        dtype=bool,
        allow_none=False,
        default=True,
        desc="Clean dead dataflow.",
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
                cls.bcast_node,
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
        if clean_dead_dataflow is not None:
            self.clean_dead_dataflow = clean_dead_dataflow

        super().__init__(*args, **kwargs)

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        bcast_value = self.bcast_value
        bcast_result = self.bcast_result
        map_entry = self.map_entry
        bcast_result_desc = bcast_result.desc(sdfg)
        bcast_value_desc = bcast_value.desc(sdfg)
        assert graph.in_degree(self.bcast_node) == 1
        assert graph.out_degree(self.bcast_node) == 1

        if gtx_xtrans.utils.is_view(bcast_result_desc):
            return False
        if gtx_xtrans.utils.is_view(bcast_value_desc):
            return False

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

            if not self._check_supplier_edge(
                state=graph,
                supplier_edge=outer_map_edge,
            ):
                return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        bcast_node = self.bcast_node
        bcast_value = self.bcast_value
        bcast_result = self.bcast_result
        map_entry = self.map_entry

        self._handle_static_access_in_map(
            state=graph,
            bcast_value=bcast_value,
            bcast_result=bcast_result,
            map_entry=map_entry,
        )

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
                graph.remove_node(bcast_node)
                sdfg.remove_data(bcast_result.data, validate=__debug__)

    @staticmethod
    def _check_supplier_edge(
        state: dace.SDFGState,
        supplier_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    ) -> bool:
        """Check if the transformation can be applied to.

        The function expects that `supplier_edge` connects the result of the broadcast
        to the Map that finally consumes it. This function is designed such that it
        can also be used by other transformations.

        The function also checks conditions on the edge.
        """
        assert isinstance(supplier_edge.src, dace_nodes.AccessNode)
        assert isinstance(supplier_edge.dst, dace_nodes.MapEntry)

        if supplier_edge.data.is_empty():
            return False
        if not supplier_edge.dst_conn.startswith("IN_"):
            return False

        bcast_result: dace_nodes.AccessNode = supplier_edge.src
        map_entry: dace_nodes.MapEntry = supplier_edge.dst
        inner_connector = "OUT_" + supplier_edge.dst_conn[3:]

        for consumer_edge in state.out_edges_by_connector(map_entry, inner_connector):
            for final_consumer in state.memlet_tree(consumer_edge).leaves():
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

    @staticmethod
    def _handle_static_access_in_map(
        state: dace.SDFGState,
        bcast_value: dace_nodes.AccessNode,
        bcast_result: dace_nodes.AccessNode,
        map_entry: dace_nodes.MapEntry,
    ) -> None:
        """Replace brodcast accesses to `bcast_result` with accesses to `bcast_value`.

        Note:
            - The function will ignore all outgoing edges of `bcast_result` that are
                not going to `map_entry`.
        """

        # Make `bcast_value` available inside the Map body.
        bcast_value_conn: str | None = None
        for edge in state.out_edges(bcast_value):
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
            state.add_edge(
                bcast_value,
                None,
                map_entry,
                "IN_" + bcast_value_conn_raw,
                dace.Memlet(data=bcast_value.data, subset="0"),
            )
            bcast_value_conn = "OUT_" + bcast_value_conn_raw
            map_entry.add_scope_connectors(bcast_value_conn_raw, force=True)
        assert bcast_value_conn in map_entry.out_connectors

        for outer_map_edge in state.out_edges(bcast_result):
            if outer_map_edge.dst is not map_entry:
                continue

            inner_connector = "OUT_" + outer_map_edge.dst_conn[3:]
            for inner_map_edge in state.out_edges_by_connector(map_entry, inner_connector):
                for mtree in state.memlet_tree(inner_map_edge).traverse_children(True):
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
                state.add_edge(
                    map_entry,
                    bcast_value_conn,
                    inner_map_edge.dst,
                    inner_map_edge.dst_conn,
                    inner_map_edge.data,  # Was modified above.
                )
                state.remove_edge(inner_map_edge)

            # Now remove the connection `(bcast_result) -> map_entry`, but keep the node
            map_entry.remove_out_connector(inner_map_edge.src_conn)
            map_entry.remove_in_connector(outer_map_edge.dst_conn)
            state.remove_edge(outer_map_edge)


@dace_properties.make_properties
class BroadcastReduction(dace_transformation.SingleStateTransformation):
    """

    translate:
    ```
        (value_to_broadcast) -> (tmp1) -> (tmp2)
    ```
    to:
    ```
        (value_to_broadcast) -> (tmp2)
    ```
    """

    bcast_value = dace_transformation.PatternNode(dace_nodes.AccessNode)
    bcast_node = dace_transformation.PatternNode(gtx_lib_nodes.Broadcast)
    bcast_result = dace_transformation.PatternNode(dace_nodes.AccessNode)

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: Optional[dict[dace.SDFG, set[str]]]

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.bcast_value,
                cls.bcast_node,
                cls.bcast_result,
            )
        ]

    def __init__(
        self,
        *args: Any,
        single_use_data: Optional[dict[dace.SDFG, set[str]]] = None,
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
        bcast_value = self.bcast_value
        bcast_result = self.bcast_result
        bcast_result_desc = bcast_result.desc(sdfg)
        bcast_value_desc = bcast_value.desc(sdfg)
        assert graph.in_degree(self.bcast_node) == 1
        assert graph.out_degree(self.bcast_node) == 1

        if gtx_xtrans.utils.is_view(bcast_result_desc):
            return False
        if gtx_xtrans.utils.is_view(bcast_value_desc):
            return False

        # NOTE: As a simplification we assume that we only broadcast a scalar. There
        #   is no real reason for it, just less analysis.
        if not isinstance(bcast_value_desc, dace_data.Scalar):
            return False

        if graph.in_degree(bcast_result) != 1:
            return False
        if not bcast_result_desc.transient:
            return False
        # Check broadcast edge if it is fance, probably not.

        # NOTE: The big question is if it favourable to perform the transformation
        #   every time. If there are only AccessNodes consumers then the answer is
        #   probably yes, as we can remove the read and write of the initial data
        #   only the write to final destination is left. If the consumers are Maps
        #   the thing is a bit different. As we have to create the intermediate
        #   allocation. If the read of the memory is okay the `InlineBroadcastAccess`
        #   transformation can get rid of it. However, if this is not possible then
        #   you would need to allocate more memory than before. Thus, we require
        #   that the Maps consumers can be handled by `InlineBroadcastAccess`.

        # Now we have to inspect all consumers of the result node.
        for consumer_edge in graph.out_edges(bcast_result):
            if consumer_edge.data.is_empty():
                return False

            match consumer := consumer_edge.dst:
                case dace_nodes.AccessNode():
                    if gtx_xtrans.utils.is_view(consumer, sdfg):
                        return False

                case dace_nodes.MapEntry():
                    if not InlineBroadcastAccess._check_supplier_edge(
                        state=graph,
                        supplier_edge=consumer_edge,
                    ):
                        return False

                case dace_nodes.NestedSDFG():
                    # TODO(phimuell): Consider implementing this case.
                    return False

                case _:
                    # This kind of node can not be handled.
                    return False

        if self._single_use_data is None:
            find_single_use_data = dace_analysis.FindSingleUseData()
            single_use_data = find_single_use_data.apply_pass(sdfg, None)
        else:
            single_use_data = self._single_use_data

        if bcast_result.data not in single_use_data[sdfg]:
            return False

        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        # Since the information what should be written is fully encoded in the
        #  destination subset of the `(bcast_result) -> (access_node)` edge
        #  we can now copy the edge.
        bcast_value = self.bcast_value
        bcast_node = self.bcast_node
        bcast_result = self.bcast_result

        for consumer_edge in graph.out_edges(bcast_result):
            match consumer := consumer_edge.dst:
                case dace_nodes.AccessNode():
                    InlineBroadcastAccess._handle_access_node_consumer(
                        bcast_value=bcast_value,
                        bcast_node=bcast_node,
                        bcast_result=bcast_result,
                        consumer_edge=consumer_edge,
                    )

                case dace_nodes.MapEntry():
                    self._handle_map_consumer(
                        bcast_value=bcast_value,
                        bcast_node=bcast_node,
                        bcast_result=bcast_result,
                        consumer_edge=consumer_edge,
                    )

                case _:
                    raise NotImplementedError(
                        f"Node type `{type(consumer).__name__}` is not supported."
                    )

    def _handle_access_node_consumer(
        state: dace.SDFGState,
        bcast_value: dace_nodes.AccessNode,
        bcast_node: gtx_lib_nodes.Broadcast,
        bcast_result: dace_nodes.AccessNode,
        consumer_edge: dace_graph.MultiConnectorEdge[dace.Memlet],
    ) -> dace_graph.MultiConnectorEdge[dace.Memlet]:
        """Bypass `bcast_result` and perform the broadcast directly into `consumer_edge.dst`.

        The function will replicate the broadcast node but it will be connected to
        `bcast_value`. Furthermore, the function will remove `consumer_edge`.
        """

        new_bcast_node = copy.deepcopy(bcast_node)
        state.add_node(new_bcast_node)

        state.add_edge(
            bcast_value,
            None,
            new_bcast_node,
            "_inp",
            dace.Memlet(data=bcast_value.data, subset="0"),
        )

        new_dst_subset: dace_sbs.Subset = copy.deepcopy(
            consumer_edge.data.get_dst_subset(consumer_edge, state)
        )
        assert new_dst_subset is not None
        new_consumer_edge = state.add_edge(
            new_bcast_node,
            "_outp",
            consumer_edge.dst,
            consumer_edge.dst_conn,
            dace.Memlet(data=consumer_edge.dst.data, subset=new_dst_subset),
        )
        state.remove_edge(consumer_edge)

        return new_consumer_edge

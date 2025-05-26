# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal, Optional, Union, overload

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_sbs,
    symbolic as dace_sym,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dace_properties.make_properties
class SplitMemlet(dace_transformation.SingleStateTransformation):
    """Preparation stage for the `SplitAccessNode`.

    Essentially splits consumer edges, such that `SplitAccessNode` become applicable.
    The function matches the following situations: `(S) -> (i) -> (D)`.
    Where `i` is the node that would be split by the `SplitAccessNode` transformation.

    The transformation essentially targets the following situation:
    ```python
    tmp = concat_where(cond1, foo(...), a)
    tmp2 = concat_where(cond1 & cond2, foo2(tmp, ...), tmp)
    ```
    It essentially rewrites it into:
    tmp = concat_where(cond1 & cond2, foo(...), a)
    tmp2_1 = concat_where(cond1 & cond2, foo2(tmp, ...), tmp)
    tmp2 = concat_where(cond1 & !cond2, foo(...), tmp2_1)
    ```
    """

    source_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    intermediate_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    target_node = dace_transformation.PatternNode(dace_nodes.AccessNode)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(cls.source_node, cls.intermediate_node, cls.target_node)
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        src_node: dace_nodes.AccessNode = self.source_node
        tmp_node: dace_nodes.AccessNode = self.intermediate_node
        tmp_desc: dace_data.Data = tmp_node.desc(sdfg)

        # If there is less than one incoming connection then it is useless to
        #  split the edges. Furthermore, `SplitAccessNode` must be able to get
        #  rid of `tmp_node`.
        if graph.in_degree(tmp_node) <= 1:
            return False
        if not tmp_desc.transient:
            return False
        if gtx_transformations.utils.is_view(tmp_desc, sdfg):
            return False

        # There can only be one connection between the source and the intermediate.
        #  This is to simplify implementation and also a restriction from the
        #  actual `SplitAccessNode`.
        src_tmp_edges = [oedge for oedge in graph.out_edges(src_node) if oedge.dst is tmp_node]
        if len(src_tmp_edges) != 1:
            return False
        src_tmp_edge: dace_graph.MultiConnectorEdge = src_tmp_edges[0]

        # We require that the producer of `tmp_node` are all distinct, these is a
        #  requirement from the splitter.
        if graph.in_degree(tmp_node) != len({iedge.src for iedge in graph.in_edges(tmp_node)}):
            return False

        tmp_subset: dace_sbs.Subset = src_tmp_edge.data.dst_subset
        if tmp_subset is None:
            return False

        # The splitting is only possible if the data, that comes from `src_node` can
        #  really be separated. For that we have to make sure that no map consumes
        #  what we write. However it is fully allowed that the Map consumes everything.
        found_edge_to_split = False
        for oedge in graph.out_edges(tmp_node):
            consumer = oedge.dst
            consumer_read = oedge.data.src_subset
            if consumer_read is None:
                return False
            elif isinstance(consumer, dace_nodes.AccessNode):
                # This transformation only makes sense if we can split some reads,
                #  thus there must be an intersection.
                if any((rs == 1) == False for _, _, rs in consumer_read):  # noqa: E712 [true-false-comparison]  # SymPy comparison
                    continue
                elif self._split_consumer_subset(
                    producer=tmp_subset,
                    consumer=consumer_read,
                    for_check=True,
                ):
                    # TODO: extend this to see that all edges could be split, also see note
                    #   At the end of this function.
                    found_edge_to_split = True
                continue
            elif isinstance(consumer, dace_nodes.MapEntry):
                try:
                    invalid_subset = tmp_subset.intersects(consumer_read) and (
                        not tmp_subset.covers(consumer_read)
                    )
                except TypeError:
                    # sympy cannot determine truth value of Relational
                    invalid_subset = False
                if invalid_subset:
                    return False
                continue
            else:
                # TODO(phimuell): Implement these case.
                return False

        if not found_edge_to_split:
            return False

        # TODO(phimuell): These tests might not be enough, meaning this transformation
        #   might apply, but there are other things that prevent `SplitAccessNode`
        #   from applying. I guess the best thing would be to apply some collapsing
        #   pass that merges the edges together.
        return True

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        src_node: dace_nodes.AccessNode = self.source_node
        tmp_node: dace_nodes.AccessNode = self.intermediate_node
        src_tmp_edge: dace_graph.MultiConnectorEdge = next(
            oedge for oedge in graph.out_edges(src_node) if oedge.dst is tmp_node
        )

        edges_to_split = self._find_edges_to_split(state=graph, src_tmp_edge=src_tmp_edge)
        self._split_consumer_edges(
            sdfg=sdfg,
            state=graph,
            src_tmp_edge=src_tmp_edge,
            edges_to_split=edges_to_split,
        )

    def _find_edges_to_split(
        self,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
    ) -> list[dace_graph.MultiConnectorEdge]:
        tmp_subset: dace_sbs.Subset = src_tmp_edge.data.dst_subset
        edges_to_split: list[dace_graph.OrderedMultiDiGraph] = []
        for oedge in state.out_edges(src_tmp_edge.dst):
            consumer = oedge.dst
            consumer_read = oedge.data.src_subset
            if isinstance(consumer, dace_nodes.AccessNode) and consumer_read is not None:
                if self._split_consumer_subset(
                    producer=tmp_subset,
                    consumer=consumer_read,
                    for_check=True,
                ):
                    edges_to_split.append(oedge)
        return edges_to_split

    def _split_consumer_edges(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
        edges_to_split: list[dace_graph.MultiConnectorEdge],
    ) -> None:
        """Split all edges in `edges_to_split` into multiple edges.

        The edges will be split such that the source subset of the new edges
        are either fully convered by the destination subset of `src_tmp_edge`
        edge or have no intersection with it at all.
        The old edges will also be removed.

        Args:
            sdfg: The SDFG on which we operate.
            state: The state in which we operate.
            src_tmp_edges: The producing source edge.
            edges_to_split: The list of edges that should be split.
        """
        for edge_to_split in edges_to_split:
            self._split_consumer_edge(
                sdfg=sdfg,
                state=state,
                src_tmp_edge=src_tmp_edge,
                edge_to_split=edge_to_split,
            )
            state.remove_edge(edge_to_split)

    def _split_consumer_edge(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        src_tmp_edge: dace_graph.MultiConnectorEdge,
        edge_to_split: dace_graph.MultiConnectorEdge,
    ) -> None:
        """Split a single edge, see `_split_consumer_edges()`  for more."""
        producer_subset = src_tmp_edge.data.dst_subset
        consumer_subset = edge_to_split.data.src_subset
        assert isinstance(edge_to_split.dst, dace_nodes.AccessNode)

        # If the subset is not given assume that it starts at zero.
        consumer_destination_subset = edge_to_split.data.dst_subset
        if consumer_destination_subset is None:
            consumer_destination_subset = dace_sbs.Range.from_array(edge_to_split.dst.desc(sdfg))

        # Perform the actual splitting.
        new_consumer_subsets = self._split_consumer_subset(
            producer=producer_subset,
            consumer=consumer_subset,
            for_check=False,
        )

        old_consumer_start = consumer_subset.min_element()
        consumer_dest_start = consumer_destination_subset.min_element()

        new_edges = []
        for new_consumer_subset in new_consumer_subsets:
            new_subset_size = new_consumer_subset.size()
            new_consumer_start = new_consumer_subset.min_element()

            # The subset at the source was computed by `_split_consumer_subset()`,
            #  but we also need the subset at the destination, i.e. where do we
            #  write to. For this we assume that we always write into some hypercube.
            #  We then compute the offset of the now source subset, compared to the
            #  original origin of the source subset and apply the same shift also
            #  to the original destination subset.
            new_consumer_dest_start = [
                dace_sym.pystr_to_symbolic(f"({dstart}) + (({ncstart}) - ({ocstart}))")
                for dstart, ocstart, ncstart in zip(
                    consumer_dest_start, old_consumer_start, new_consumer_start
                )
            ]
            new_consumer_dest_end = [
                dace_sym.pystr_to_symbolic(f"({ncdstart}) + ({ss} - 1)")
                for ncdstart, ss in zip(new_consumer_dest_start, new_subset_size)
            ]
            new_consumer_dest_subset = dace_sbs.Range(
                [
                    (start, end, 1)
                    for start, end in zip(new_consumer_dest_start, new_consumer_dest_end)
                ]
            )

            # Create the new edge, and copy the Memlet, afterwards set the subsets
            #  accordingly.
            # NOTE: The volume is not updated, but we do not care about that.
            # NOTE: Because the consumer are only AccessNodes, and the data has not
            #   changed, there is no need to propagate or update anything.
            new_edges.append(
                state.add_edge(
                    edge_to_split.src,
                    edge_to_split.src_conn,
                    edge_to_split.dst,
                    edge_to_split.dst_conn,
                    dace.Memlet.from_memlet(edge_to_split.data),
                )
            )
            new_edges[-1].data.src_subset = new_consumer_subset
            new_edges[-1].data.dst_subset = new_consumer_dest_subset

    @overload
    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: Literal[True],
    ) -> bool: ...

    @overload
    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: Literal[False],
    ) -> list[dace_sbs.Range]: ...

    def _split_consumer_subset(
        self,
        producer: dace_sbs.Range,
        consumer: dace_sbs.Range,
        for_check: bool,
    ) -> Union[list[dace_sbs.Range], bool]:
        """Splits the `consumer` subset.

        The resulting subsets are either fully covered by `producer` or have no
        intersection with it. If `for_check` is `True` the function will return
        a boolean to indicate if the subset can be split or not (for any reason).
        It is an error to call the function with `for_check` set to `False`
        but the subset can not be split.

        Args:
            producer: The subset describing the producer.
            consumer: The subset describing what the consumer reads.
            for_check: Only check if the subset can be split.

        Todo:
            The current implementation is only able to handle the case where
            the consumer subset must only be split in one dimension. This
            restriction must be solved.
        """
        assert producer.dims() == consumer.dims()

        # Currently we require that we have to split only along one dimension.
        dimension_in_which_to_split: Optional[int] = None
        splitted_subsets_in_dim: list[tuple[Any, ...]] = []
        for dim in range(producer.dims()):
            prod_low = producer[dim][0]
            prod_high = producer[dim][1]
            consu_low = consumer[dim][0]
            consu_high = consumer[dim][1]

            # In this dimension the consumer consumes everything the producer
            #  generates. Therefore no splitting is needed.
            embedded_cond1 = (prod_low <= consu_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            embedded_cond2 = (consu_high <= prod_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            if embedded_cond1 and embedded_cond2:
                continue

            # Check if there is an intersection at all.
            #  I am pretty sure that there is no strange `-1` correction needed.
            intersec_cond1 = consu_low <= prod_high
            intersec_cond2 = prod_low <= consu_high
            if intersec_cond1 == False or intersec_cond2 == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                assert for_check
                return False
            if not (intersec_cond1 == True and intersec_cond2 == True):  # noqa: E712 [true-false-comparison]  # SymPy comparison
                assert for_check
                return False

            # The consumer is not fully embedded in the producer, so this dimension
            #  we must split. If we found before ignore it.
            # TODO(phimuell): By ignoring this case here, i.e. "pretend that no split
            #   was needed", we could handle that and then recursively handle the
            #   rest.
            if dimension_in_which_to_split is not None:
                assert for_check
                return False
            dimension_in_which_to_split = dim

            # Determine the splitting case that we have.
            #  I am pretty sure about the `<` here.
            read_right = (prod_high < consu_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            read_left = (consu_low < prod_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
            assert read_right or read_left

            # If we only want to check then we do not need the exact splitting.
            if for_check:
                continue

            # Now we determine the split mode. There are three cases.
            if read_right and read_left:
                # The consumer starts reading before the producer starts to write
                #  and also reads more than the producer writes to, so it is split
                #  into three parts.
                splitted_subsets_in_dim = [
                    (consu_low, prod_low - 1, 1),
                    (prod_low, prod_high, 1),
                    (prod_high + 1, consu_high, 1),
                ]
            elif read_left:
                # The consumer starts reading before the producer starts writing.
                #  Thus there are two parts.
                splitted_subsets_in_dim = [(consu_low, prod_low - 1, 1), (prod_low, consu_high, 1)]
            elif read_right:
                # The consumer starts reading inside the range the producer writes to
                #  but reads more, so again two splits.
                splitted_subsets_in_dim = [
                    (consu_low, prod_high, 1),
                    (prod_high + 1, consu_high, 1),
                ]

        # In check mode we are done.
        if for_check:
            return dimension_in_which_to_split is not None

        assert dimension_in_which_to_split is not None
        assert len(splitted_subsets_in_dim) > 0
        assert all(((e - s) >= 0) == True for s, e, _ in splitted_subsets_in_dim)  # noqa: E712 [true-false-comparison]  # SymPy comparison

        splitted_subsets: list[dace_sbs.Range] = []
        for splitted_subset_in_dim in splitted_subsets_in_dim:
            splitted_subsets.append(
                dace_sbs.Range(
                    [
                        (
                            splitted_subset_in_dim
                            if dim == dimension_in_which_to_split
                            else org_consumer_sbs
                        )
                        for dim, org_consumer_sbs in enumerate(consumer)
                    ]
                )
            )

        return splitted_subsets

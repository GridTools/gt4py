# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
from typing import Any, Iterable, Optional, Sequence, Union

import dace
from dace import data as dace_data, subsets as dace_sbs, symbolic as dace_sym
from dace.sdfg import graph as dace_graph, nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


@dataclasses.dataclass(frozen=True)
class EdgeConnectionSpec:
    """Describes an edge in an abstract way, that is kind of independent of the direction.

    It is always described in terms of a node, which is eader the `src` or the `dst`
    of the edge, in terms of the subset the edge read or writes and the edge itself.

    To construct `EdgeConnectionSpec` you can use the `describe_incoming_edges()`
    and `describe_outgoing_edges()` function.
    """

    node: dace_nodes.Node
    subset: dace_sbs.Subset
    edge: dace_graph.MultiConnectorEdge

    def __post_init__(self) -> None:
        assert self.subset is not None
        assert all((r[2] == 1) == True for r in self.subset)  # noqa: E712 [true-false-comparison]  # SymPy comparison

    def __hash__(self) -> int:
        return hash(self.edge)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EdgeConnectionSpec):
            return self.edge.other.self
        return self.edge == other

    @property
    def is_incoming(self) -> bool:
        """Checks if the edge describes an incoming edge.

        An edge is incoming if `self.node` is the same as `self.edge.dst`.
        """
        return self.node is self.edge.dst

    @property
    def other_node(self) -> dace_nodes.Node:
        """Returns the node at the other side of the edge."""
        return self.edge.src if self.is_incoming else self.edge.dst

    @property
    def other_subset(self) -> Optional[dace_sbs.Subset]:
        """Returns the subset at the other end of the edge.

        It is important that there is no real connection with the `other_subset`
        attribute of `dace.Memlet`, the differences are subtitle but important.
        `self.subset` is always the subset the edge has at the node that it
        describes, while `self.other_subset` is "the other one".

        Note:
            While `self.subset` is never `None`, this property might be `None`.
        """
        return self.edge.data.src_subset if self.is_incoming else self.edge.data.dst_subset


def describe_edge(
    edge: dace_graph.MultiConnectorEdge,
    incoming_edge: bool,
) -> EdgeConnectionSpec:
    """Create a description for a single edge."""
    get_sbs = lambda e: e.data.dst_subset if incoming_edge else e.data.src_subset  # noqa: E731 [lambda-assignment]
    get_node = lambda e: e.dst if incoming_edge else e.src  # noqa: E731 [lambda-assignment]
    return EdgeConnectionSpec(
        node=get_node(edge),
        subset=get_sbs(edge),
        edge=edge,
    )


def describe_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
    incoming_edges: bool,
) -> list[EdgeConnectionSpec]:
    """Generate the description of the edges.

    Function will either describes the incoming edges on node `node`, if
    `incoming_edges` is `True` or the out going edges of node `node` if
    `incoming_edges` is `False`.

    Args:
        state: The state in which `node` is located.
        node: The node whose edges should be described.
        incoming_edges: Describe the incoming (`True`) or out going (`False`) edges
            of `node`.
    """
    edges = state.in_edges(node) if incoming_edges else state.out_edges(node)
    return [describe_edge(e, incoming_edges) for e in edges]


def describe_incoming_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
) -> list[EdgeConnectionSpec]:
    """Describes the incoming edges of `node`."""
    return describe_edges(state, node, True)


def describe_outgoing_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
) -> list[EdgeConnectionSpec]:
    """Describes the out going edges of `node`."""
    return describe_edges(state, node, False)


def describe_all_edges(
    state: dace.SDFG,
    node: dace_nodes.Node,
) -> list[EdgeConnectionSpec]:
    """Describes the all edges of `node`."""
    return describe_edges(state, node, False) + describe_edges(state, node, True)


def describes_incoming_edge(desc: EdgeConnectionSpec) -> bool:
    """Test if `desc` describes an incoming edge."""
    return desc.node is desc.edge.dst


def get_other_node(desc: EdgeConnectionSpec) -> dace_nodes.Node:
    """Extract the "other" node of the edge.

    Deprecated, use `desc.other_node` instead.
    """
    return desc.other_node


def get_other_subset(desc: EdgeConnectionSpec) -> dace_sbs.Subset:
    """Get the subset of the other side of the edge.

    Note, this is not the same as `desc.edge.data.other_subset`.
    Deprecated, use `desc.other_subset` instead.
    """
    return desc.other_subset


def split_node(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    split_description: Sequence[dace_sbs.Subset],
    allow_to_bypass_nodes: bool = False,
    already_reconfigured_nodes: Optional[set[tuple[dace_nodes.Node, str]]] = None,
) -> dict[dace_sbs.Subset, dace_nodes.AccessNode]:
    """The function will split `node_to_split` into several smaller AccessNodes.

    How the split is performed is described by `split_description`, which is a list
    of subsets that describes how to partition `node_to_split`.
    The function will then create new data descriptors for the for each of the fragment
    and create new AccessNodes. Then it will relocate all edges from `node_to_split`
    to the respective AccessNode of the fragment. The dataflow of the producer and
    consumer region will be reconfigured appropriate and strides are propagated.

    In some cases it might be possible to avoid to create an intermediate AccessNode.
    For example, if there is a Map that writes into `node_to_split` and a consumer
    to that node happens to be an AccessNode. In that case, the function will skip
    the write into the fragment and write directly into the consumer. However,
    for this `allow_to_bypass_nodes` must be `True`. Furthermore, note that the
    function will still create the data descriptors and the AccessNodes associated
    to the fragments.

    The function returns a `dict` that maps the subset describing each split to the
    AccessNode that was created for it.

    Note:
        - It is not required that`split_description` covers `node_to_split` completely.
            It is only required that a read or write is fully covered by a single
            fragment.
        - It is the responsibility of the caller to remove the isolated AccessNodes
            and remove the unused data descriptors.

    Todo:
        Make it possible to pass the AccessNodes and/or data descriptors from the
        outside.
    """
    if already_reconfigured_nodes is None:
        already_reconfigured_nodes = set()

    desc_to_split = node_to_split.desc(sdfg)
    assert desc_to_split.transient
    assert not gtx_transformations.utils.is_view(desc_to_split)
    assert isinstance(desc_to_split, dace_data.Array)

    input_descriptions = describe_incoming_edges(state, node_to_split)
    output_descriptions = describe_outgoing_edges(state, node_to_split)
    edge_descriptions = input_descriptions + output_descriptions

    # Ensure that no edge is on multiple new fragments and that every edge
    #  lands on a fragment.
    assert all(
        any(split.covers(desc.subset) for split in split_description) for desc in edge_descriptions
    ), "There is an edge that would be split."

    # Assign the edges of the node to a split.
    assignment = _compute_assignement_for_split(edge_descriptions, split_description)

    # Now we create the data descriptors and the AccessNodes that we need.
    new_data_descriptors = _generate_data_descriptors_for_split(
        state=state,
        sdfg=sdfg,
        node_to_split=node_to_split,
        split_description=split_description,
    )
    new_access_nodes: dict[dace_sbs.Subset, dace_nodes.AccessNode] = {
        split: state.add_access(dname) for split, dname in new_data_descriptors.items()
    }

    # This will also remove the `node_to_split` node but not the data descriptor
    #  it referred to.
    _perform_node_split(
        state=state,
        sdfg=sdfg,
        node_to_split=node_to_split,
        new_access_nodes=new_access_nodes,
        assignment=assignment,
        allow_to_bypass_nodes=allow_to_bypass_nodes,
        already_reconfigured_nodes=already_reconfigured_nodes,
    )

    return new_access_nodes


def split_copy_edge(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    edge_to_split: dace_graph.MultiConnectorEdge,
    split_description: Sequence[dace_sbs.Subset],
) -> dict[Union[dace_sbs.Range, None], set[dace_graph.MultiConnectorEdge]]:
    """Tries to split `edge_to_split` into multiple edges.

    How the edge is split is described by `split_description`, which is a
    sequence of subsets. The function will decompose `edge_to_split` using
    `decompose_subset()` and then create a new edge for each resulting subset.

    The function returns a `dict` that maps a split subset to the set of
    edges that were created. Note that the `dict` also contains the
    key `None` which contains all the edges that are not associated to a
    split.

    Args:
        state: The state in which we operate.
        sdfg: The SDFG on which we operate.
        edge_to_split: The edge that should be split.
        split_description: How the split should be executed.
    """

    # NOTE: The implementation is written from the point of view that
    #  `edge_to_split` is an outgoing edge. But this is an implementation
    #  detail that does not limit the applicability of this function.
    # TODO(phimuell): Implements some check that nothing is lost.

    assert isinstance(edge_to_split.src, dace_nodes.AccessNode)
    assert not isinstance(edge_to_split.src.desc(sdfg), dace_data.View)
    assert isinstance(edge_to_split.src.desc(sdfg), dace_data.Array)
    assert isinstance(edge_to_split.dst, dace_nodes.AccessNode)
    assert not isinstance(edge_to_split.dst.desc(sdfg), dace_data.View)
    assert isinstance(edge_to_split.dst.desc(sdfg), dace_data.Array)

    memlet_to_split: dace.Memlet = edge_to_split.data
    assert memlet_to_split.wcr is None
    assert memlet_to_split.subset is not None
    assert memlet_to_split.other_subset is not None
    assert all((r[2] == 1) == True for r in memlet_to_split.subset)  # noqa: E712 [true-false-comparison]  # SymPy comparison

    src: dace_nodes.AccessNode = edge_to_split.src
    consumer_subset: dace_sbs.Subset = memlet_to_split.src_subset

    # Fully split the consuming edge.
    # TODO(phimuell): Find a better way.
    fully_splitted_subsets: list[dace_sbs.Subset] = [consumer_subset]
    for split in split_description:
        new_fully_splitted_subsets: list[dace_sbs.Subset] = []
        for consumer in fully_splitted_subsets:
            split_res = decompose_subset(producer=split, consumer=consumer)
            if split_res:
                # It was possible to further split the consumer subset.
                new_fully_splitted_subsets.extend(split_res)
            else:
                # It was not possible to further split the consumer subset. Thus
                #  `consumer` is final.
                new_fully_splitted_subsets.append(consumer)
        fully_splitted_subsets = new_fully_splitted_subsets

    new_edges: dict[Union[dace_sbs.Range, None], dace_graph.MultiConnectorEdge] = {
        split: set() for split in split_description
    }
    new_edges[None] = set()

    consumer_dest_subset: dace_sbs.Subset = memlet_to_split.dst_subset
    consumer_dest_subset_start = consumer_dest_subset.min_element()
    consumer_subset_start = consumer_subset.min_element()
    for new_consumer_subset in fully_splitted_subsets:
        new_consumer_subset_start = new_consumer_subset.min_element()
        new_consumer_subset_size = new_consumer_subset.size()
        new_consumer_offset = [
            new_start - old_start
            for new_start, old_start in zip(new_consumer_subset_start, consumer_subset_start)
        ]

        new_consumer_dest_subset = dace_sbs.Range(
            [
                (old_dst_start + offset, old_dst_start + offset + size - 1, 1)
                for old_dst_start, size, offset in zip(
                    consumer_dest_subset_start, new_consumer_subset_size, new_consumer_offset
                )
            ]
        )
        new_edge = state.add_edge(
            src,
            edge_to_split.src_conn,
            edge_to_split.dst,
            edge_to_split.dst_conn,
            dace.Memlet(
                data=src.data,
                subset=new_consumer_subset,
                other_subset=new_consumer_dest_subset,
                dynamic=memlet_to_split.dynamic,
            ),
        )
        new_edges[
            next((split for split in split_description if split.covers(new_consumer_subset)), None)
        ].add(new_edge)

    state.remove_edge(edge_to_split)
    return new_edges


def decompose_subset(
    producer: dace_sbs.Subset,
    consumer: dace_sbs.Subset,
) -> Union[list[dace_sbs.Subset], None]:
    """This function performs is able to split `consumer` in one dimension.

    The function decomposes `consumer` into fragments, each fragment is either
    fully covered by `producer` or there has no intersection
    with it.

    The function returns the list of the fragments of `consumer`. However,
    there are some special return values:
    - `None`: If the split is not applicable, for example there is no
        intersection in at least one dimensions.
    - The empty `list`: Indicates that `producer` fully covers `consumer`.

    Args:
        producer: The subset that can not be split.
        consumer: The subset that should be decomposed.
    """
    assert producer.dims() == consumer.dims()

    # Currently we require that we have to split only along one dimension.
    dimension_in_which_to_split: Optional[int] = None
    splitted_subsets_in_dim: list[tuple[Any, ...]] = []
    needs_further_spliting = False
    for dim in range(producer.dims()):
        prod_low = producer[dim][0]
        prod_high = producer[dim][1]
        consu_low = consumer[dim][0]
        consu_high = consumer[dim][1]

        # Check if the domains are the same.
        #  It seems that this must be a special case, at least it is handled that
        #  way in DaCe.
        equal_cond1 = (prod_low == consu_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        equal_cond2 = (consu_high == prod_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        if equal_cond1 and equal_cond2:
            continue

        # In this dimension the consumer consumes everything the producer
        #  generates. Therefore no splitting is needed.
        embedded_cond1 = (prod_low <= consu_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        embedded_cond2 = (consu_high <= prod_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        if embedded_cond1 and embedded_cond2:
            continue

        # Check if there is an intersection at all.
        #  I am pretty sure that there is no strange `-1` correction needed here.
        intersec_cond1 = consu_low <= prod_high
        intersec_cond2 = prod_low <= consu_high
        if intersec_cond1 == False or intersec_cond2 == False:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return None
        if not (intersec_cond1 == True and intersec_cond2 == True):  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return None

        # `consumer` must be split in multiple dimensions, we already found one,
        #  stored in `dimension_in_which_to_split` but also found a new one, `dim`.
        #  We will now pretend that we did not see this. Instead we will do the
        #  split in the first dimension we found. We rely on the driver that it
        #  subsequently calls this function.
        if dimension_in_which_to_split is not None:
            needs_further_spliting = True
            continue
        else:
            dimension_in_which_to_split = dim

        # Determine the splitting case that we have.
        #  I am pretty sure about the `<` here.
        read_right = (prod_high < consu_high) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        read_left = (consu_low < prod_low) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        assert read_right or read_left

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

    # If we are here this means that no split is applicable, because `consumer` is
    #  fully covered by `producer`. To indicate this we return the empty list.
    if dimension_in_which_to_split is None:
        assert not needs_further_spliting
        assert producer.covers(consumer)
        return []

    assert dimension_in_which_to_split is not None
    assert len(splitted_subsets_in_dim) > 0
    assert all(((e - s) >= 0) == True for s, e, _ in splitted_subsets_in_dim)  # noqa: E712 [true-false-comparison]  # SymPy comparison

    splitted_subsets: list[dace_sbs.Subset] = []
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

    # If we need further splitting then we must call the split subsets recursively.
    if needs_further_spliting:
        fully_splitted_subsets: list[dace_sbs.Subset] = []
        for consumer_fragment in splitted_subsets:
            next_decomposed_level = decompose_subset(producer=producer, consumer=consumer_fragment)
            if next_decomposed_level:
                fully_splitted_subsets.extend(next_decomposed_level)
            else:
                # There was nothing to split.
                fully_splitted_subsets.append(consumer_fragment)
        return fully_splitted_subsets
    else:
        return splitted_subsets


def subset_merger(
    subsets: Union[Sequence[EdgeConnectionSpec], Sequence[dace_sbs.Subset]],
) -> list[dace_sbs.Subset]:
    """Merges subsets together.

    The function tries the merge as many subsets together as possible.
    While, the function does not guarantees to find the largest possible
    subset, it guarantees that the result does not depend on the particular
    order of the passed subsets.

    Note:
        The function achieves its stability by first order the subsets in
        a particular order. For this it will serialize the subsets to strings
        and then sort them alphabetical. As this is the only way that also
        works in the presence of symbols.
    """
    assert len(subsets) > 0

    # Bring everything down to subsets. Note that it is important, that
    #  we do make a copy, because `_subset_merger_impl()` modifies its
    #  argument inplace.
    if isinstance(subsets[0], EdgeConnectionSpec):
        subsets = [copy.deepcopy(desc.subset) for desc in subsets]
    else:
        assert all(isinstance(sbs, dace_sbs.Subset) for sbs in subsets)
        subsets = [copy.deepcopy(sbs) for sbs in subsets]

    if len(subsets) == 1:
        return subsets

    return _subset_merger_impl(subsets)


def _subset_merger_impl(
    subsets: list[dace_sbs.Subset],
) -> list[dace_sbs.Subset]:
    """Implementation of the subset merger.

    The function will modify `subsets` inplace.
    """

    # The best we can do to ensure some kind of stability is sort them, however,
    #  in case of symbols we can not do that. Thus as the _only_ solution, we
    #  serialize them to string and order them accordingly.
    # NOTE: We can modify it inplace.
    subsets.sort(key=lambda sub: str(sub))

    performed_merge = True
    while performed_merge and (len(subsets) > 1):
        performed_merge = False

        # We could use `itertools.combinations()` here, but to guarantee deterministic
        #  processing, we will implement it yourselves.
        for idx1 in range(len(subsets)):
            for idx2 in range(idx1 + 1, len(subsets)):
                subset1, subset2 = subsets[idx1], subsets[idx2]
                merged_subset = _try_to_merge_subsets(subset1, subset2)
                if merged_subset is not None:
                    subsets.remove(subset1)
                    subsets.remove(subset2)
                    subsets.append(merged_subset)
                    performed_merge = True
                    break
            if performed_merge:
                break

    return subsets


def _try_to_merge_subsets(
    subset1: dace_sbs.Subset,
    subset2: dace_sbs.Subset,
) -> Union[None, dace_sbs.Subset]:
    """Tries to merge the subsets together, it it is impossible return `None`.

    Two subset can only be merged if they have the same bounds in all but one
    dimension. In that dimension the end index of one of the subset is the
    same as the start index of the other.
    """
    if subset1.dims() != subset2.dims():
        return None

    has_found_merge_dim = False
    merged_subset: list[dace_sym.SymbolicType] = []
    for dim in range(subset1.dims()):
        start1, end1, step1 = subset1[dim]
        start2, end2, step2 = subset2[dim]

        if (step1 != 1) == True or (step2 != 1) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            return None

        elif (start1 == start2) == True and (end1 == end2) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
            merged_subset.append((start1, end1, 1))

        else:
            # We found a possible merge dimension.
            if has_found_merge_dim:
                # It is only possible to merge, actually extend, along in one dimension.
                return None

            if ((end1 + 1) == start2) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                merged_subset.append((start1, end2, 1))
            elif ((end2 + 1) == start1) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
                merged_subset.append((start2, end1, 1))
            else:
                return None
            has_found_merge_dim = True

    return dace_sbs.Range(merged_subset)


def _perform_node_split(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    new_access_nodes: dict[dace_sbs.Subset, dace_nodes.AccessNode],
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]],
    allow_to_bypass_nodes: bool,
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> None:
    """Performs the actual split of `node_to_split` based on `assignment`.

    `assignment` describes how the edges of `node_to_split` should be distributed
    among the `new_access_nodes`. It maps a "split", i.e. a subset that describes
    one of the shards of the original AccessNode, to the edges that should be
    moved to another AccessNode, which is described through `new_access_nodes`.

    The function will also reconfigure the dataflow of the old edges. The argument
    `already_reconfigured_nodes` is used to keep track which dataflow has already
    been reconfigured.
    The function will then propagate the strides starting from the new functions.

    Furthermore, the function will remove the `node_to_split` AccessNode, but it
    will not remove the data.
    """
    assert all(sbs in assignment for sbs in new_access_nodes.keys())

    handled_edges: set[dace_graph.MultiConnectorEdge] = set()
    for split in new_access_nodes:
        edges_to_relocate = assignment[split]
        new_access_node = new_access_nodes[split]

        if len(edges_to_relocate) == 0:
            continue

        assert state.degree(new_access_node) == 0
        assert all(
            edge_to_relocate.edge not in handled_edges for edge_to_relocate in edges_to_relocate
        )

        if allow_to_bypass_nodes and _can_use_bypass_version_of_node_spliter(edges_to_relocate):
            _perform_node_split_with_bypass_impl(
                state=state,
                sdfg=sdfg,
                node_to_split=node_to_split,
                edges_to_relocate=edges_to_relocate,
                already_reconfigured_nodes=already_reconfigured_nodes,
            )
        else:
            for edge_to_relocate in edges_to_relocate:
                assert edge_to_relocate.edge not in handled_edges
                _perform_node_split_impl(
                    state=state,
                    sdfg=sdfg,
                    node_to_split=node_to_split,
                    new_access_node=new_access_node,
                    split_description=split,
                    edge_to_relocate=edge_to_relocate,
                    already_reconfigured_nodes=already_reconfigured_nodes,
                )
        handled_edges.update(edesc.edge for edesc in edges_to_relocate)

    # While we remove the AccessNode we do not remove the underlying data. We do this
    #  because it might be that there is another node that is still referring to
    #  it, which happens if we split across multiple states.
    assert state.degree(node_to_split) == 0
    state.remove_node(node_to_split)

    # Propagate the strides starting from the new access nodes.
    # NOTE: If the bypass version was used then the propagation has been done there
    #   already.
    for new_access_node in new_access_nodes.values():
        if state.degree(new_access_node) == 0:
            continue
        gtx_transformations.gt_propagate_strides_from_access_node(
            sdfg=sdfg,
            state=state,
            outer_node=new_access_node,
        )


def _perform_node_split_impl(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    new_access_node: dace_nodes.AccessNode,
    split_description: dace_sbs.Subset,
    edge_to_relocate: EdgeConnectionSpec,
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> dace_graph.MultiConnectorEdge:
    """Performs the actual split.

    In essence this function will create a new edge, based on but `edge_to_relocate`,
    but where `node_to_split` is replaced with `new_access_node`. Furthermore, the
    subset of the edge is modified to reflect this change and the dataflow on the
    "other side of the edge" is also modified by means of `reconfigure_dataflow_after_rerouting()`.
    However, it is important that the function does not propagate strides.

    The old edge is removed and the new one is returned.
    """
    # TODO(phimuell): In case the other node is an AccessNode it is potentially
    #   possible to bypass `new_access_node`. This is an optimization that we
    #   should do as it also acts as kind of redundant array removal.

    # Because the edge no longer writes into `node_to_split` but into `new_access_node`,
    #  which is smaller, we have to adapt the size. However, because we assume that
    #  the split covers the subset the edge transfers, the offset is simply given
    #  by where the split starts. The minus is needed for the correct format in
    #  `reroute_edge()`.
    assert split_description.covers(edge_to_relocate.subset)
    subset_correction: Any = [-min_elem for min_elem in split_description.min_element()]

    is_producer_edge = describes_incoming_edge(edge_to_relocate)

    new_edge = gtx_transformations.utils.reroute_edge(
        is_producer_edge=is_producer_edge,
        current_edge=edge_to_relocate.edge,
        ss_offset=subset_correction,
        state=state,
        sdfg=sdfg,
        old_node=node_to_split,
        new_node=new_access_node,
    )

    # Depending on the situation we do not need a reconfiguration.
    if is_producer_edge:
        other_node: dace_nodes.Node = edge_to_relocate.edge.src
        other_node_conn = edge_to_relocate.edge.src_conn
    else:
        other_node = edge_to_relocate.edge.dst
        other_node_conn = edge_to_relocate.edge.dst_conn

    # Reconfigure the data flow on the other side.
    if (other_node, other_node_conn) not in already_reconfigured_nodes:
        already_reconfigured_nodes.add((other_node, other_node_conn))

        if isinstance(other_node, dace_nodes.NestedSDFG):
            # There is nothing special to do. In case of a nested SDFG, we will also have to do
            #  stride propagation, but we will postpone that.
            pass

        elif is_producer_edge and isinstance(other_node, dace_nodes.MapExit):
            pass

        elif (not is_producer_edge) and isinstance(other_node, dace_nodes.MapEntry):
            pass

        elif isinstance(other_node, dace_nodes.AccessNode):
            # Required by `reconfigure_dataflow_after_rerouting()`.
            if isinstance(other_node.desc(sdfg), dace_data.Scalar):
                subset_correction = None

        else:
            raise TypeError(f"Can not handle a producer of type '{type(other_node).__name__}'")

        gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
            is_producer_edge=is_producer_edge,
            new_edge=new_edge,
            ss_offset=subset_correction,
            state=state,
            sdfg=sdfg,
            old_node=node_to_split,
            new_node=new_access_node,
        )

    state.remove_edge(edge_to_relocate.edge)
    return new_edge


def _can_use_bypass_version_of_node_spliter(
    edges_to_relocate: Iterable[EdgeConnectionSpec],
) -> bool:
    producer_edges: list[EdgeConnectionSpec] = [
        desc for desc in edges_to_relocate if describes_incoming_edge(desc)
    ]
    if len(producer_edges) != 1:
        return False
    if not isinstance(producer_edges[0].other_node, dace_nodes.AccessNode):
        return False
    return True


def _perform_node_split_with_bypass_impl(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    edges_to_relocate: set[EdgeConnectionSpec],
    already_reconfigured_nodes: set[tuple[dace_nodes.Node, str]],
) -> list[dace_graph.MultiConnectorEdge]:
    """Performs the splitting but the edge might go directly to the consumer.

    # TODO: Remove the producer edge, run reconfiguration, split operation.
    # TODO ADDING PRODUCER TO THE SET OF PROCESSED NODES

    """
    producer_edge_desc = next(
        edesc for edesc in edges_to_relocate if describes_incoming_edge(edesc)
    )
    data_producer: dace_nodes.Node = producer_edge_desc.other_node
    producer_edge = producer_edge_desc.edge
    assert producer_edge.dst is node_to_split
    assert isinstance(data_producer, dace_nodes.AccessNode)

    old_producer_read = producer_edge.data.src_subset.min_element()
    old_producer_write = producer_edge.data.dst_subset.min_element()

    consumer_edges = [edesc for edesc in edges_to_relocate if edesc is not producer_edge_desc]
    new_consumer_edges: list[dace_graph.MultiConnectorEdge] = []
    for consumer_edge_desc in consumer_edges:
        consumer_edge = consumer_edge_desc.edge
        consumer_node: dace_nodes.Node = consumer_edge.dst
        consumer_conn = consumer_edge.dst_conn

        # Index from where the consumer should start reading from the producer
        #  directly. But since it has gone through the intermediate AccessNode,
        #  the indexes are different and we have to compute them. At the end it
        #  is some kind of projection starting at what was read from the
        #  intermediate to the indexes where it has been written to, to the
        #  intermediate and finally from where it is originally coming from.
        #  We only do this projection for the start index, the end index
        #  is computed by adding the length to the start index.
        old_consumer_read = consumer_edge.data.src_subset.min_element()
        consumer_read_size = consumer_edge.data.src_subset.size()
        consumer_direct_read: list[tuple[dace_sym.SymbolicType, dace_sym.SymbolicType, int]] = []
        for i in range(len(old_producer_read)):
            old_producer_read_start = old_producer_read[i]
            old_consumer_read_start = old_consumer_read[i]
            old_producer_write_start = old_producer_write[i]
            transfer_size = consumer_read_size[i]

            consumer_direct_read_start = dace.symbolic.pystr_to_symbolic(
                f"({old_producer_read_start}) + (({old_consumer_read_start}) - ({old_producer_write_start}))",
                simplify=True,
            )
            # The `-1` is because the end is considered inclusive in DaCe.
            consumer_direct_read_end = dace.symbolic.pystr_to_symbolic(
                f"({consumer_direct_read_start}) + ({transfer_size}) - 1", simplify=True
            )
            consumer_direct_read.append((consumer_direct_read_start, consumer_direct_read_end, 1))

        new_consumer_direct_read_subset = dace_sbs.Range(consumer_direct_read)

        # Create a new edge that reads from the producer directly and remove the
        #  old edge.
        new_consumer_edge = state.add_edge(
            producer_edge.src,
            producer_edge.src_conn,
            consumer_edge.dst,
            consumer_edge.dst_conn,
            dace.Memlet(
                data=data_producer.data,
                subset=new_consumer_direct_read_subset,
                other_subset=consumer_edge.data.dst_subset,
                dynamic=consumer_edge.data.dynamic or producer_edge.data.dynamic,
            ),
        )
        state.remove_edge(consumer_edge)
        new_consumer_edges.append(new_consumer_edge)

        # If needed reconfigure the consumers since it now involves the
        #  original producer.
        #  The stride propagation is done after all edges have been updated.
        if (consumer_node, consumer_conn) not in already_reconfigured_nodes:
            already_reconfigured_nodes.add((consumer_node, consumer_conn))

            # The subset correct we have to apply to the consumer depends on the
            #  type of the consumer.
            if isinstance(data_producer.desc(sdfg), dace_data.Scalar):
                # This is required by the reconfigure function.
                consumer_subset_correction = None

            elif isinstance(consumer_node, dace_nodes.AccessNode):
                # Here we only have to consider the offset between the reading
                #  from the intermediate and where we write into it. The minus
                #  is because we have to subtract this value.
                consumer_subset_correction = [
                    dace.symbolic.pystr_to_symbolic(
                        f"-(({old_consumer_read_start}) - ({old_producer_write_start}))",
                        simplify=True,
                    )
                    for old_consumer_read_start, old_producer_write_start in zip(
                        old_consumer_read,
                        old_producer_write,
                        strict=True,
                    )
                ]
            elif isinstance(consumer_node, dace_nodes.MapEntry):
                # Things are different here, because the map ranges (and the
                #  offsets in the inner Memlets handle everything, Thus, in
                #  this case we have to correct the case that they now read
                #  from the producer instead of the intermediate.
                consumer_subset_correction = [
                    dace.symbolic.pystr_to_symbolic(
                        f"-(({old_producer_write_start}) - ({old_producer_read_start}))",
                        simplify=True,
                    )
                    for old_producer_read_start, old_producer_write_start in zip(
                        old_producer_read,
                        old_producer_write,
                        strict=True,
                    )
                ]

            elif isinstance(consumer_node, dace_nodes.NestedSDFG):
                # Since a NestedSDFG can only read from an AccessNode there is
                #  nothing to do, except for the stride propagation, which is
                #  done later.
                continue

            else:
                raise TypeError(
                    f"Can not correct a consumer of type '{type(consumer_node).__name__}'"
                )

            gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
                is_producer_edge=False,
                new_edge=new_consumer_edge,
                ss_offset=consumer_subset_correction,
                state=state,
                sdfg=sdfg,
                old_node=node_to_split,
                new_node=data_producer,
            )

    # We do not reconfigure the dataflow of the producer, because upstream of
    #  `data_producer` nothing has changed. In fact, we can not do it since
    #  there is also no edge. For that reason we do not add the producer to the
    #  `already_reconfigured_nodes` set, although one could do that.
    #  However, we have to propagate the strides, we have to do it here because on
    #  the outside it would only propagate from the new AccessNodes that we have
    #  bypassed.
    # TODO(phimuell): Find a way to avoid doing the propagation here, where the
    #   dataflow might be in some invalid state.
    state.remove_edge(producer_edge)
    processed_nsdfgs: set = set()
    for new_consumer_edge in new_consumer_edges:
        gtx_transformations.gt_map_strides_to_dst_nested_sdfg(
            outer_node=data_producer,
            edge=new_consumer_edge,
            sdfg=sdfg,
            state=state,
            processed_nsdfgs=processed_nsdfgs,
        )

    return new_consumer_edges


def _generate_data_descriptors_for_split(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    node_to_split: dace_nodes.AccessNode,
    split_description: Iterable[dace_sbs.Subset],
) -> dict[dace_sbs.Subset, str]:
    """Creates the data descriptor for every split.

    While the size is taken from `split_description` the other properties needed
    for the data descriptor are taken from `node_to_split`.
    """
    desc_to_split = node_to_split.desc(sdfg)
    new_data_descriptors: dict[dace_sbs.Subset, str] = {}

    for i, split_subset in enumerate(split_description):
        tmp_shape = split_subset.size()
        tmp_name, _ = sdfg.add_transient(
            name=f"{node_to_split.data}_split_{i}",
            shape=tmp_shape,
            dtype=desc_to_split.dtype,
            storage=desc_to_split.storage,
            find_new_name=True,
        )
        new_data_descriptors[split_subset] = tmp_name

    return new_data_descriptors


def _compute_assignement_for_split(
    edge_descriptions: Sequence[EdgeConnectionSpec],
    split_description: Sequence[dace_sbs.Subset],
) -> dict[dace_sbs.Subset, set[EdgeConnectionSpec]]:
    """For every subset, that defines a split find the set of edges that belongs into it.

    Note that it might happens that some splits have zero assigned edges.
    """
    assert all(split is not None for split in split_description)
    assignment: dict[dace_sbs.Subset, set[EdgeConnectionSpec]] = {
        split: set() for split in split_description
    }

    for edge_description in edge_descriptions:
        assert edge_description.subset is not None
        assigned_split = next(
            iter(split for split in split_description if split.covers(edge_description.subset)),
            None,
        )
        assert assigned_split is not None
        assignment[assigned_split].add(edge_description)

    return assignment

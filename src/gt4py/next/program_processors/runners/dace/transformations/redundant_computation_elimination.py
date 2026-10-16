# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Elimination of redundant computations from GT4Py based SDFGs.

The GT4Py frontend lowers every field expression into its own Map, nested
inside the SDFG of the enclosing lambda. Identical sub expressions, for
example the projection of a vector field onto the edge normal that is needed
by two different consumers, are therefore lowered multiple times, once per
occurrence. After the lambda SDFGs have been inlined, these duplicates show up
as structurally equivalent, sibling Map scopes inside the same state, that
only differ in the names of the transient arrays they define.

The function :func:`gt_eliminate_redundant_computations` detects such
duplicate Map scopes and removes them by redirecting every read of the
duplicate's output array to the canonical scope's output array. Because the
duplicate and the canonical scope are structurally equivalent, this is
semantics preserving. The detection is based on a graph isomorphism test
between the scopes, with the names of the transient arrays masked out, so
only truly equivalent computations are merged. The transformation is safe by
construction: any scope that does not satisfy the strict structural
requirements, such as a single transient output, no accesses outside its
defining state, or no views and nested SDFGs, is left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Iterable, Optional

import dace
from dace import subsets as dace_subsets
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_utils as gtx_dace_mfu,
)


if TYPE_CHECKING:
    import networkx as nx


_INTERNAL_PLACEHOLDER: Final = "__REDUNDANT_INTERNAL__"
_MAX_ITERATIONS: Final = 20


def gt_eliminate_redundant_computations(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    """Removes Map scopes that compute the same thing as another Map scope.

    The function scans every state of `sdfg` for sibling Map scopes (scopes
    that have the same parent scope) that are structurally equivalent and
    write to a single transient array. For every such group of duplicates all
    but the first scope are removed and the reads of their output arrays are
    redirected to the output array of the remaining scope.

    Removing a duplicate can make previously different scopes equal, for
    example when two chains of Maps become identical after the outputs of
    their common producer have been unified. The function therefore iterates
    until no more duplicates are found.

    The transformation should be run after the nested lambda SDFGs have been
    inlined, for example as part of `gt_simplify()`, such that the duplicate
    scopes are siblings inside the same state.

    Args:
        sdfg: The SDFG to process, it is modified in place.
        validate: Perform validation after the transformation has run.
        validate_all: Perform extensive validation.

    Returns:
        The number of Map scopes that were removed.
    """
    num_removed_total = 0
    pending_array_removal: set[str] = set()
    for _ in range(_MAX_ITERATIONS):
        num_removed = 0
        for graph in sdfg.all_sdfgs_recursive():
            for state in graph.states():
                num_removed += _eliminate_in_state(graph, state, pending_array_removal)
        num_removed_total += num_removed
        if num_removed == 0:
            break

    # Remove the arrays of the eliminated scopes. This is deferred until the
    #  fixpoint iteration has converged, because the connector names of the
    #  consumers still refer to the removed arrays and the equivalence test
    #  relies on the names being resolvable.
    for name in pending_array_removal:
        if name not in sdfg.arrays:
            continue
        if not any(
            node.data == name
            for graph in sdfg.all_sdfgs_recursive()
            for state in graph.states()
            for node in state.data_nodes()
        ):
            sdfg.remove_data(name, validate=False)

    if num_removed_total and (validate or validate_all):
        sdfg.validate()

    return num_removed_total


def _eliminate_in_state(
    sdfg: dace.SDFG, state: dace.sdfg.SDFGState, pending_array_removal: set[str]
) -> int:
    """Eliminate duplicate sibling Map scopes in a single state."""
    scope_dict = state.scope_dict()
    # Group the Map entries by their parent scope, only scopes that are
    #  siblings, i.e. have the same parent, can be merged.
    siblings: dict[Optional[dace_nodes.MapEntry], list[dace_nodes.MapEntry]] = {}
    for node in state.nodes():
        if isinstance(node, dace_nodes.MapEntry):
            parent = scope_dict[node]
            assert parent is None or isinstance(parent, dace_nodes.MapEntry)
            siblings.setdefault(parent, []).append(node)

    num_removed = 0
    for map_entries in siblings.values():
        num_removed += _eliminate_in_sibling_group(sdfg, state, map_entries, pending_array_removal)
    return num_removed


def _eliminate_in_sibling_group(
    sdfg: dace.SDFG,
    state: dace.sdfg.SDFGState,
    map_entries: list[dace_nodes.MapEntry],
    pending_array_removal: set[str],
) -> int:
    """Eliminate duplicate Map scopes within a group of sibling Maps."""

    # Cheap pre-bucketing to avoid quadratic isomorphism checks. Entries that
    #  have become stale, because they were nested inside a scope that was
    #  removed by an earlier group, are skipped.
    buckets: dict[tuple, list[dace_nodes.MapEntry]] = {}
    for map_entry in map_entries:
        if not _is_live_scope(state, map_entry):
            continue
        info = _analyze_scope(sdfg, state, map_entry)
        if info is None:
            continue
        buckets.setdefault(info.bucket_key, []).append(map_entry)

    num_removed = 0
    for candidates in buckets.values():
        # The first scope of a bucket is kept as the canonical one, every
        #  other scope that is isomorphic to it is removed. Scopes are
        #  re-analyzed inside the comparison, such that changes from earlier
        #  removals are taken into account. Removing a scope also removes the
        #  Maps nested inside it, thus entries may become stale and have to
        #  be skipped.
        for position, canonical in enumerate(candidates):
            if not _is_live_scope(state, canonical):
                continue
            for candidate in list(candidates[position + 1 :]):
                if not _is_live_scope(state, candidate):
                    continue
                if _scopes_equivalent(sdfg, state, canonical, candidate):
                    if _redirect_and_remove(
                        sdfg, state, candidate, canonical, pending_array_removal
                    ):
                        num_removed += 1
    return num_removed


def _is_live_scope(state: dace.sdfg.SDFGState, map_entry: dace_nodes.MapEntry) -> bool:
    """Check if the Map entry is still part of the state."""
    try:
        return state.exit_node(map_entry) is not None
    except (KeyError, StopIteration):
        return False


class _ScopeInfo:
    """Structural information about a Map scope relevant for deduplication."""

    def __init__(
        self,
        map_entry: dace_nodes.MapEntry,
        internal_data: set[str],
        output_data: str,
        bucket_key: tuple,
    ) -> None:
        self.map_entry = map_entry
        self.internal_data = internal_data
        self.output_data = output_data
        self.bucket_key = bucket_key


def _subset_string(subset: Any) -> str:
    """A canonical string representation of a Memlet subset."""
    if subset is None:
        return "None"
    try:
        return str(dace_subsets.Range(subset))
    except TypeError:
        return str(subset)


def _analyze_scope(
    sdfg: dace.SDFG, state: dace.sdfg.SDFGState, map_entry: dace_nodes.MapEntry
) -> Optional[_ScopeInfo]:
    """Check if the scope is a deduplication candidate and compute its bucket key.

    A scope is a candidate if:
    - it writes to exactly one transient, non-pooled array through its exit,
    - the output array is only accessed inside this state and only written by
      this scope, without any views in between,
    - the scope only contains tasklets, access nodes and nested Maps.
    """
    map_exit = state.exit_node(map_entry)
    if map_exit is None:
        return None

    # The scope must write to exactly one data container.
    out_edges = state.out_edges(map_exit)
    out_data = {e.data.data for e in out_edges}
    if len(out_data) != 1:
        return None
    output_data = next(iter(out_data))
    if output_data not in sdfg.arrays:
        return None
    output_desc = sdfg.arrays[output_data]
    if not output_desc.transient or not isinstance(
        output_desc, (dace.data.Array, dace.data.Scalar)
    ):
        return None
    if getattr(output_desc, "pool", False):
        # Pooled transients share their storage with other arrays, aliasing
        #  them is not safe.
        return None

    # Restrict the scope contents to node types that can be compared safely.
    scope_nodes = list(
        state.scope_subgraph(map_entry, include_entry=False, include_exit=False).nodes()
    )
    for node in scope_nodes:
        if not isinstance(
            node,
            (dace_nodes.Tasklet, dace_nodes.AccessNode, dace_nodes.MapEntry, dace_nodes.MapExit),
        ):
            return None

    # The transient arrays defined and used inside the scope.
    internal_data = {node.data for node in scope_nodes if isinstance(node, dace_nodes.AccessNode)}
    # The output array must not be one of the scope's internal arrays.
    if output_data in internal_data:
        return None

    # Every access to the output array must happen inside this state.
    for other_state in sdfg.states():
        if other_state is not state:
            if any(node.data == output_data for node in other_state.data_nodes()):
                return None

    # Every write to the output array must come from this scope's exit node
    #  and the array must not be accessed through views.
    for node in state.data_nodes():
        if node.data != output_data:
            continue
        for edge in state.in_edges(node):
            if edge.src is not map_exit:
                return None
        for edge in state.out_edges(node):
            if isinstance(edge.dst, dace_nodes.AccessNode):
                return None

    # A cheap bucket key: map ranges, the tasklet code inside the scope and
    #  the external input interface. Scopes in different buckets can not be
    #  equivalent.
    tasklet_codes = sorted(
        node.code.as_string for node in scope_nodes if isinstance(node, dace_nodes.Tasklet)
    )
    ext_inputs = sorted(
        (edge.data.data, _subset_string(edge.data.subset)) for edge in state.in_edges(map_entry)
    )
    map_range = _subset_string(map_entry.map.range)
    bucket_key = (map_range, tuple(tasklet_codes), tuple(ext_inputs))

    return _ScopeInfo(map_entry, internal_data, output_data, bucket_key)


def _mask_string(text: Optional[str], names: Iterable[str]) -> Optional[str]:
    """Replace occurrences of the given array names inside a string.

    Longer names are replaced first to avoid mangling names that are a prefix
    of another name.
    """
    if text is None:
        return None
    masked = text
    for name in sorted(names, key=len, reverse=True):
        if name in masked:
            masked = masked.replace(name, _INTERNAL_PLACEHOLDER)
    return masked


def _node_signature(node: dace_nodes.Node, internal_data: set[str]) -> tuple:
    """A structural signature of a node, with internal array names masked."""
    if isinstance(node, dace_nodes.Tasklet):
        return ("tasklet", node.code.as_string)
    if isinstance(node, dace_nodes.AccessNode):
        return ("access", _INTERNAL_PLACEHOLDER if node.data in internal_data else node.data)
    if isinstance(node, dace_nodes.MapEntry):
        return ("map_entry", tuple(node.map.params), _subset_string(node.map.range))
    if isinstance(node, dace_nodes.MapExit):
        return ("map_exit",)
    return ("node", type(node).__name__)


def _transient_names(sdfg: dace.SDFG) -> list[str]:
    """The names of all transient arrays of the SDFG."""
    return [name for name, desc in sdfg.arrays.items() if desc.transient]


def _scopes_equivalent(
    sdfg: dace.SDFG,
    state: dace.sdfg.SDFGState,
    canonical: dace_nodes.MapEntry,
    candidate: dace_nodes.MapEntry,
) -> bool:
    """Test if two sibling Map scopes are structurally equivalent.

    The comparison is a graph isomorphism test between the two scope
    subgraphs, including their entry and exit nodes, that ignores the names of
    transient arrays, but compares everything else, in particular tasklet
    code, map ranges, Memlet data and subsets, exactly. In addition the
    external interface, i.e. the data read through the entry nodes, and the
    subsets written through the exit nodes must be identical.
    """
    import networkx as nx

    info_canonical = _analyze_scope(sdfg, state, canonical)
    info_candidate = _analyze_scope(sdfg, state, candidate)
    if info_canonical is None or info_candidate is None:
        return False

    # The output arrays must be interchangeable.
    desc_canonical = sdfg.arrays[info_canonical.output_data]
    desc_candidate = sdfg.arrays[info_candidate.output_data]
    if not desc_canonical.is_equivalent(desc_candidate):
        return False
    if tuple(str(s) for s in desc_canonical.strides) != tuple(
        str(s) for s in desc_candidate.strides
    ):
        return False

    # The external interface, i.e. the data read through the entry node, must
    #  be identical.
    if sorted(
        (e.data.data, _subset_string(e.data.subset)) for e in state.in_edges(canonical)
    ) != sorted((e.data.data, _subset_string(e.data.subset)) for e in state.in_edges(candidate)):
        return False

    # The subsets written through the exit must be identical.
    if sorted(
        _subset_string(e.data.subset) for e in state.out_edges(state.exit_node(canonical))
    ) != sorted(_subset_string(e.data.subset) for e in state.out_edges(state.exit_node(candidate))):
        return False

    graph_canonical = _scope_nx_graph(
        state,
        canonical,
        info_canonical.internal_data,
        info_canonical.output_data,
        _transient_names(sdfg),
    )
    graph_candidate = _scope_nx_graph(
        state,
        candidate,
        info_candidate.internal_data,
        info_candidate.output_data,
        _transient_names(sdfg),
    )
    if graph_canonical.number_of_nodes() != graph_candidate.number_of_nodes():
        return False
    if graph_canonical.number_of_edges() != graph_candidate.number_of_edges():
        return False

    def node_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return n1["sig"] == n2["sig"]

    def edge_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        return e1["sigs"] == e2["sigs"]

    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        graph_candidate, graph_canonical, node_match=node_match, edge_match=edge_match
    )
    return matcher.is_isomorphic()


def _scope_nx_graph(
    state: dace.sdfg.SDFGState,
    map_entry: dace_nodes.MapEntry,
    internal_data: set[str],
    output_data: str,
    transient_names: list[str],
) -> nx.DiGraph:
    """Build a `networkx.DiGraph` representation of the scope subgraph.

    Entry and exit nodes are part of the graph to anchor the isomorphism. The
    names of the transient arrays are masked in connector names, since the
    lowering derives the connector names from the data names and duplicates
    use different names for the arrays they define. The Memlet data, in
    contrast, is compared exactly, except for the transients defined inside
    the scope and the scope's output array, which are masked as well. Note
    that a scope can not access its own output internally, this is excluded
    by `_analyze_scope()`, such that masking the output name is safe.

    Parallel edges between the same pair of nodes, which are common for
    Memlets, are aggregated into a sorted multiset of edge signatures, such
    that a simple `DiGraph` isomorphism test with edge matching can be used.
    """
    import networkx as nx

    masked_data = internal_data | {output_data}
    subgraph = state.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    graph = nx.DiGraph()
    for node in subgraph.nodes():
        graph.add_node(node, sig=_node_signature(node, internal_data))
    edge_signatures: dict[tuple[dace_nodes.Node, dace_nodes.Node], list[tuple]] = {}
    for edge in subgraph.edges():
        data = _INTERNAL_PLACEHOLDER if edge.data.data in masked_data else edge.data.data
        signature = (
            _mask_connector(edge.src_conn, masked_data, transient_names),
            _mask_connector(edge.dst_conn, masked_data, transient_names),
            data,
            _subset_string(edge.data.subset),
        )
        edge_signatures.setdefault((edge.src, edge.dst), []).append(signature)
    for (src, dst), signatures in edge_signatures.items():
        graph.add_edge(src, dst, sigs=tuple(sorted(signatures, key=repr)))
    return graph


def _mask_connector(
    connector: Optional[str], internal_data: set[str], transient_names: list[str]
) -> Optional[str]:
    """Mask array names inside a connector name.

    The names of the transients defined inside the scope are always masked.
    For the remaining connector names the names of all transients of the SDFG
    are masked, because the data that flows through the connector has already
    been verified on the Memlets itself and connector names of duplicates may
    refer to stale array names after a redirection.
    """
    if connector is None:
        return None
    masked = _mask_string(connector, internal_data)
    masked = _mask_string(masked, transient_names)
    return masked


def _redirect_and_remove(
    sdfg: dace.SDFG,
    state: dace.sdfg.SDFGState,
    candidate: dace_nodes.MapEntry,
    canonical: dace_nodes.MapEntry,
    pending_array_removal: set[str],
) -> bool:
    """Remove a duplicate scope and redirect its consumers to the canonical output.

    Assumes that the caller has verified through `_scopes_equivalent()` that
    the candidate scope computes the same as the canonical scope. The removal
    of the arrays that become unused is deferred, they are collected in
    `pending_array_removal`, and must be removed by the caller after the
    fixpoint iteration has converged. Returns `True` if the duplicate was
    removed.
    """
    info_candidate = _analyze_scope(sdfg, state, candidate)
    info_canonical = _analyze_scope(sdfg, state, canonical)
    if info_candidate is None or info_canonical is None:
        return False
    output_candidate = info_candidate.output_data
    output_canonical = info_canonical.output_data
    if output_candidate == output_canonical:
        return False

    # Collect all read edges of the duplicate's output array and their memlet
    #  trees. The trees are needed to redirect the data through the consumer
    #  scopes. This is done in a separate step before any modification, such
    #  that the transformation is aborted without side effects if a tree can
    #  not be processed.
    read_edges = []
    for node in state.data_nodes():
        if node.data != output_candidate:
            continue
        for edge in state.out_edges(node):
            read_edges.append((node, edge))
    trees = []
    for node, edge in read_edges:
        try:
            tree_edges = list(state.memlet_tree(edge))
        except Exception:
            return False
        for tree_edge in tree_edges:
            if isinstance(tree_edge.src, dace_nodes.NestedSDFG) or isinstance(
                tree_edge.dst, dace_nodes.NestedSDFG
            ):
                return False
            # The tree must consistently refer to the output array of the
            #  duplicate. Trees that switch the data, for example when
            #  passing through views, can not be redirected safely.
            if tree_edge.data is None or tree_edge.data.data is None:
                continue
            if tree_edge.data.data != output_candidate:
                return False
        trees.append((node, edge, tree_edges))

    # Redirect the data of all memlets that read the duplicate's output array
    #  to the canonical output array, following the memlet trees through the
    #  consumer scopes.
    for _, _, tree_edges in trees:
        for tree_edge in tree_edges:
            if tree_edge.data is not None and tree_edge.data.data == output_candidate:
                tree_edge.data.data = output_canonical

    # Re-source the external read edges to the access node of the canonical
    #  output array that receives the write of the canonical scope. Reusing
    #  the write node, instead of creating a new access node, ensures that
    #  the reads are ordered after the write inside the state.
    canonical_exit = state.exit_node(canonical)
    canonical_write_nodes = [
        edge.dst
        for edge in state.out_edges(canonical_exit)
        if isinstance(edge.dst, dace_nodes.AccessNode) and edge.dst.data == output_canonical
    ]
    if not canonical_write_nodes:
        return False
    canonical_node = canonical_write_nodes[0]
    for _, edge, _ in trees:
        destination, dst_conn = edge.dst, edge.dst_conn
        memlet = edge.data
        state.remove_edge(edge)
        state.add_edge(canonical_node, None, destination, dst_conn, memlet)

    # Remove the duplicate scope itself.
    candidate_exit = state.exit_node(candidate)
    gtx_dace_mfu.delete_map(state, candidate, candidate_exit)

    # Remove the isolated access nodes of the arrays that have become unused
    #  and mark the arrays for removal, which is deferred to the caller.
    for name in {*info_candidate.internal_data, output_candidate}:
        for node in [n for n in state.data_nodes() if n.data == name]:
            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                state.remove_node(node)
        pending_array_removal.add(name)

    return True

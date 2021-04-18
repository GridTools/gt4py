from copy import deepcopy

import dace
import dace.sdfg.graph
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import networkx as nx
import pytest

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.definitions import BuildOptions
from gt4py.frontend.gtscript_frontend import GTScriptFrontend
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
from gtc.dace_to_oir import convert
from gtc.gtir_to_oir import GTIRToOIR, oir_iteration_space_computation
from gtc.oir_to_dace import OirSDFGBuilder
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging

from ...test_integration.stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from ...test_integration.stencil_definitions import REGISTRY as stencil_registry


def stencil_def_to_oir(stencil_def, externals):

    build_options = BuildOptions(
        name=stencil_def.__name__, module=__name__, rebuild=True, backend_opts={}, build_info=None
    )
    definition_ir = GTScriptFrontend.generate(
        stencil_def, externals=externals, options=build_options
    )
    gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
    return GTIRToOIR().visit(gtir)


def edge_match(edge1, edge2):
    edge1 = next(iter(edge1.values()))
    edge2 = next(iter(edge2.values()))
    try:
        if edge1["src_conn"] is not None:
            assert edge2["src_conn"] is not None
            assert edge1["src_conn"] == edge2["src_conn"]
        else:
            assert edge2["src_conn"] is None
        assert edge1["data"] == edge2["data"]
        assert edge1["data"].data == edge2["data"].data
    except AssertionError:
        return False
    return True


def node_match(n1, n2):
    n1 = n1["node"]
    n2 = n2["node"]
    try:
        if not isinstance(
            n1, (dace.nodes.AccessNode, VerticalLoopLibraryNode, HorizontalExecutionLibraryNode)
        ):
            raise TypeError
        if isinstance(n1, dace.nodes.AccessNode):
            assert isinstance(n2, dace.nodes.AccessNode)
            assert n1.access == n2.access
            assert n1.data == n2.data
        elif isinstance(n1, VerticalLoopLibraryNode):
            assert isinstance(n2, VerticalLoopLibraryNode)
            assert len(n1.sections) == len(n2.sections)
            for (interval1, he_sdfg1), (interval2, he_sdfg2) in zip(n1.sections, n2.sections):
                assert interval1.covers(interval2) and interval2.covers(interval1)
                assert_sdfg_equal(he_sdfg1, he_sdfg2)
        else:
            assert isinstance(n2, HorizontalExecutionLibraryNode)
            assert len(n1.oir_node.body) == len(n2.oir_node.body)
            assert list(sorted(n1.in_connectors)) == list(sorted(n2.in_connectors))
            assert list(sorted(n1.out_connectors)) == list(sorted(n2.out_connectors))
            if n1.oir_node.mask is None:
                assert n2.oir_node.mask is None
    except AssertionError:
        return False
    return True


def assert_sdfg_equal(sdfg1: dace.SDFG, sdfg2: dace.SDFG):
    assert len(sdfg1.states()) == 1
    assert len(sdfg2.states()) == 1
    state1 = sdfg1.states()[0]
    state2 = sdfg2.states()[0]

    # SDFGState.nx does not contain any node info in the networkx node attrs (but does for edges),
    # so we add it here manually.
    nx.set_node_attributes(state1.nx, {n: n for n in state1.nx.nodes}, "node")
    nx.set_node_attributes(state2.nx, {n: n for n in state2.nx.nodes}, "node")

    assert nx.is_isomorphic(state1.nx, state2.nx, edge_match=edge_match, node_match=node_match)

    for name in sdfg1.arrays.keys():
        assert isinstance(sdfg1.arrays[name], type(sdfg2.arrays[name]))
        assert isinstance(sdfg2.arrays[name], type(sdfg1.arrays[name]))
        assert sdfg1.arrays[name].dtype == sdfg2.arrays[name].dtype
        assert sdfg1.arrays[name].transient == sdfg2.arrays[name].transient
        assert sdfg1.arrays[name].shape == sdfg2.arrays[name].shape


@pytest.mark.parametrize("stencil_name", stencil_registry.keys())
def test_stencils_roundtrip_raw(stencil_name):

    stencil_def = stencil_registry[stencil_name]
    externals = externals_registry[stencil_name]
    oir = stencil_def_to_oir(stencil_def, externals)
    sdfg = OirSDFGBuilder.build(oir.name, oir)

    sdfg_pre = deepcopy(sdfg)

    oir = convert(sdfg)
    sdfg_post = OirSDFGBuilder.build(oir.name, oir)
    assert_sdfg_equal(sdfg_pre, sdfg_post)


@pytest.mark.parametrize("stencil_name", stencil_registry.keys())
@hyp.given(
    use_greedy_merging=hyp_st.booleans(),
    use_adjacent_loop_merging=hyp_st.booleans(),
    use_local_temporaries_to_scalars=hyp_st.booleans(),
    use_write_before_read_temporaries_to_scalars=hyp_st.booleans(),
    use_on_the_fly_merging=hyp_st.booleans(),
    use_no_field_access_pruning=hyp_st.booleans(),
    use_ij_cache_detection=hyp_st.booleans(),
    use_k_cache_detection=hyp_st.booleans(),
    use_prune_k_cache_fills=hyp_st.booleans(),
    use_prune_k_cache_flushes=hyp_st.booleans(),
)
def test_stencils_roundtrip_optimized(
    stencil_name,
    use_greedy_merging,
    use_adjacent_loop_merging,
    use_local_temporaries_to_scalars,
    use_write_before_read_temporaries_to_scalars,
    use_on_the_fly_merging,
    use_no_field_access_pruning,
    use_ij_cache_detection,
    use_k_cache_detection,
    use_prune_k_cache_fills,
    use_prune_k_cache_flushes,
):

    stencil_def = stencil_registry[stencil_name]
    externals = externals_registry[stencil_name]
    oir = stencil_def_to_oir(stencil_def, externals)
    use_on_the_fly_merging = False
    if use_greedy_merging:
        oir = GreedyMerging().visit(oir)
    if use_adjacent_loop_merging:
        oir = AdjacentLoopMerging().visit(oir)
    if use_local_temporaries_to_scalars:
        oir = LocalTemporariesToScalars().visit(oir)
    if use_write_before_read_temporaries_to_scalars:
        oir = WriteBeforeReadTemporariesToScalars().visit(oir)
    if use_on_the_fly_merging:
        oir = OnTheFlyMerging().visit(oir)
    if use_no_field_access_pruning:
        oir = NoFieldAccessPruning().visit(oir)
    if use_ij_cache_detection:
        oir = IJCacheDetection().visit(oir)
    if use_k_cache_detection:
        oir = KCacheDetection().visit(oir)
    if use_prune_k_cache_fills:
        oir = PruneKCacheFills().visit(oir)
    if use_prune_k_cache_flushes:
        oir = PruneKCacheFlushes().visit(oir)
    oir = oir_iteration_space_computation(oir)
    sdfg = OirSDFGBuilder.build(oir.name, oir)

    sdfg_pre = deepcopy(sdfg)

    oir = convert(sdfg)
    sdfg_post = OirSDFGBuilder.build(oir.name, oir)
    assert_sdfg_equal(sdfg_pre, sdfg_post)

from copy import deepcopy

import dace
import dace.sdfg.graph
import networkx as nx
import pytest

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.definitions import BuildOptions
from gt4py.frontend.gtscript_frontend import GTScriptFrontend
from gtc.dace.dace_to_oir import convert
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.gtir_to_oir import GTIRToOIR
from gtc.passes.gtir_pipeline import GtirPipeline

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
            assert n1.loop_order == n2.loop_order
            assert n1.caches == n2.caches
            assert len(n1.sections) == len(n2.sections)
            for (interval1, he_sdfg1), (interval2, he_sdfg2) in zip(n1.sections, n2.sections):
                assert interval1 == interval2
                assert_sdfg_equal(he_sdfg1, he_sdfg2)
        else:
            assert isinstance(n2, HorizontalExecutionLibraryNode)
            assert n1.as_oir() == n2.as_oir()
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
    sdfg = OirSDFGBuilder().visit(oir)

    sdfg_pre = deepcopy(sdfg)

    oir = convert(sdfg)
    sdfg_post = OirSDFGBuilder().visit(oir)
    assert_sdfg_equal(sdfg_pre, sdfg_post)

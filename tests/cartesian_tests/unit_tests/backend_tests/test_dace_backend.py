# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# Skip this module when we collecting tests and "dace" is not installed as a dependency.
pytest.importorskip("dace")


from dace import nodes
from dace import sdfg as dace_sdfg
from dace.sdfg.state import LoopRegion
import dace.sdfg.analysis.schedule_tree.treenodes as tn

from gt4py.cartesian import backend
from gt4py.cartesian.backend.dace_backend import SDFGManager
from gt4py.cartesian.gtscript import computation, PARALLEL, interval, Field
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.cartesian.gtc.dace.treeir import Axis

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


@pytest.mark.parametrize(
    ["name", "device"],
    [
        ("dace:cpu", "cpu"),
        ("dace:cpu_kfirst", "cpu"),
        ("dace:cpu_KJI", "cpu"),
        pytest.param("dace:gpu", "gpu", marks=[pytest.mark.requires_gpu]),
    ],
)
def test_dace_backend(name: str, device: str):
    dace_backend = backend.from_name(name)

    assert dace_backend.storage_info["device"] == device


def copy_stencil(
    in_field: Field[float],  # type: ignore
    out_field: Field[float],  # type: ignore
):
    with computation(PARALLEL), interval(...):
        out_field = in_field


def copy_forward_stencil(
    in_field: Field[float],  # type: ignore
    out_field: Field[float],  # type: ignore
):
    with computation(FORWARD), interval(...):
        out_field = in_field


def test_default_schedule_is_KJI():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu")
    manager = SDFGManager(builder)

    stree = manager.schedule_tree()

    expected_K_map = stree.get_root().children[0]

    assert (
        isinstance(expected_K_map, tn.MapScope)
        and len(expected_K_map.node.params) == 1
        and expected_K_map.node.params[0].startswith(Axis.K.iteration_symbol())
    )

    expected_JI = expected_K_map.children[0]

    assert (
        isinstance(expected_JI, tn.MapScope)
        and len(expected_JI.node.params) == 2
        and expected_JI.node.params[0].startswith(Axis.J.iteration_symbol())
        and expected_JI.node.params[1].startswith(Axis.I.iteration_symbol())
    )


def test_dace_cpu_loop_structure():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    loop_indices = [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)]
    assert len(loop_indices[0]) == 1 and loop_indices[0][0].startswith("__k_")
    assert loop_indices[1] == ["__i", "__j"]


def test_dace_cpu_kfirst_loop_structure():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu_kfirst")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    loop_indices = [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)]
    assert loop_indices[0] == ["__i", "__j"]
    assert len(loop_indices[1]) == 1 and loop_indices[1][0].startswith("__k_")

    builder = StencilBuilder(copy_forward_stencil, backend="dace:cpu_kfirst")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    assert len(list(sdfg.states())) == 1, "expect one state"
    state = sdfg.states()[0]

    # Expect a Map for IJ outside
    map_entry_nodes = [node for node in state.nodes() if isinstance(node, nodes.MapEntry)]
    assert len(map_entry_nodes) == 1, "expect one MapEntry node"
    assert map_entry_nodes[0].map.params == ["__i", "__j"]

    # Expect LoopRegion for K inside map
    nsdfg_nodes = [node for node in state.nodes() if isinstance(node, nodes.NestedSDFG)]
    assert len(nsdfg_nodes) == 1
    for_nested_nodes = nsdfg_nodes[0].sdfg.nodes()
    assert len(for_nested_nodes) == 1
    loop_region = for_nested_nodes[0]
    assert isinstance(loop_region, LoopRegion)
    assert loop_region.loop_variable.startswith("__k")


def test_dace_cpu_KJI_loop_structure():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu_KJI")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    loop_indices = [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)]
    assert len(loop_indices[0]) == 1 and loop_indices[0][0].startswith("__k_")
    assert loop_indices[1] == ["__j", "__i"]

    builder = StencilBuilder(copy_forward_stencil, backend="dace:cpu_KJI")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()

    # Expect LoopRegion for K outside
    loop_region: LoopRegion = list(sdfg.all_control_flow_blocks())[0]
    assert loop_region.loop_variable.startswith("__k")

    # Expect JI Map and in loop_body state (#2)
    state = loop_region.start_block
    assert [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)] == [
        ["__j", "__i"]
    ]

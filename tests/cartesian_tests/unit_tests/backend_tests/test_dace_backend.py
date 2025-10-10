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

from gt4py.cartesian import backend
from gt4py.cartesian.backend.dace_backend import SDFGManager
from gt4py.cartesian.gtscript import computation, PARALLEL, interval, Field
from gt4py.cartesian.stencil_builder import StencilBuilder

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


@pytest.mark.parametrize(
    ["name", "device"],
    [
        ("dace:cpu", "cpu"),
        ("dace:cpu_kfirst", "cpu"),
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


def test_dace_cpu_loop_structure():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    assert [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)] == [
        ["__k_0"],
        ["__i", "__j"],
    ]


def test_dace_cpu_kfirst_loop_structure():
    builder = StencilBuilder(copy_stencil, backend="dace:cpu_kfirst")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    assert [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)] == [
        ["__i", "__j"],
        ["__k_0"],
    ]

    builder = StencilBuilder(copy_forward_stencil, backend="dace:cpu_kfirst")
    manager = SDFGManager(builder)

    sdfg = manager.sdfg_via_schedule_tree()
    state = sdfg.states()[0]

    # Expect IJ Map and For loop construct (Nested SDFG, four guard states)
    assert [node.map.params for node in state.nodes() if isinstance(node, nodes.MapEntry)] == [
        ["__i", "__j"]
    ]
    for_nested_nodes = [
        node.sdfg.nodes() for node in state.nodes() if isinstance(node, nodes.NestedSDFG)
    ]
    assert [isinstance(node, dace_sdfg.SDFGState) for node in for_nested_nodes[0]] == [
        True,
        True,
        True,
        True,
    ]

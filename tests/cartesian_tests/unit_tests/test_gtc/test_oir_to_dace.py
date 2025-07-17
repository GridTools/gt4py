# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dace
else:
    dace = pytest.importorskip("dace")

from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.common import DataType
from gt4py.cartesian.gtc.dace.nodes import StencilComputation
from gt4py.cartesian.gtc.dace.oir_to_dace import OirSDFGBuilder

from cartesian_tests.unit_tests.test_gtc.oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    ScalarAccessFactory,
    StencilFactory,
)

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_oir_sdfg_builder_copy_stencil() -> None:
    stencil_name = "copy"
    stencil = StencilFactory(
        name=stencil_name,
        params=[
            FieldDeclFactory(name="A", dtype=DataType.FLOAT32),
            FieldDeclFactory(name="B", dtype=DataType.FLOAT32),
        ],
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left=FieldAccessFactory(name="B"), right=FieldAccessFactory(name="A"))
        ],
    )
    sdfg = OirSDFGBuilder().visit(stencil)

    assert isinstance(sdfg, dace.SDFG), "DaCe SDFG expected"
    assert sdfg.name == stencil_name, "Stencil name is preserved"
    assert len(sdfg.arrays) == 2, "two arrays expected (A and B)"

    a_array = sdfg.arrays.get("A")
    assert a_array is not None, "Array A expected to be defined"
    assert a_array.ctype == "float", "A is of type `float`"
    assert a_array.offset == (0, 0, 0), "CartesianOffset.zero() expected"

    b_array = sdfg.arrays.get("B")
    assert b_array is not None, "Array B expected to be defined"
    assert b_array.ctype == "float", "B is of type `float`"
    assert b_array.offset == (0, 0, 0), "CartesianOffset.zero() expected"

    states = sdfg.nodes()
    assert len(states) >= 1, "at least one state expected"

    # expect StencilComputation, AccessNode(A), and AccessNode(B) in the last block
    last_block = states[len(states) - 1]
    nodes = last_block.nodes()
    assert (
        len(list(filter(lambda node: isinstance(node, StencilComputation), nodes))) == 1
    ), "one StencilComputation library node"
    assert (
        len(
            list(
                filter(
                    lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "A", nodes
                )
            )
        )
        == 1
    ), "one AccessNode of A"
    assert (
        len(
            list(
                filter(
                    lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "B", nodes
                )
            )
        )
        == 1
    ), "one AccessNode of B"

    edges = last_block.edges()
    assert len(edges) == 2, "read and write memlet path expected"

    library_node = list(filter(lambda node: isinstance(node, StencilComputation), nodes))[0]
    read_access = list(
        filter(lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "A", nodes)
    )[0]
    write_access = list(
        filter(lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "B", nodes)
    )[0]

    assert edges[0].src == read_access and edges[0].dst == library_node, "read access expected"
    assert edges[1].src == library_node and edges[1].dst == write_access, "write access expected"


def test_oir_sdfg_builder_assign_scalar_param() -> None:
    stencil_name = "scalar_assign"
    stencil = StencilFactory(
        name=stencil_name,
        params=[
            FieldDeclFactory(name="A", dtype=DataType.FLOAT64),
            oir.ScalarDecl(name="b", dtype=DataType.INT32),
        ],
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(
                left=FieldAccessFactory(name="A"), right=ScalarAccessFactory(name="b")
            )
        ],
    )
    sdfg = OirSDFGBuilder().visit(stencil)

    assert isinstance(sdfg, dace.SDFG), "DaCe SDFG expected"
    assert sdfg.name == stencil_name, "Stencil name is preserved"
    assert len(sdfg.arrays) == 1, "one array expected (A)"

    a_array = sdfg.arrays.get("A")
    assert a_array is not None, "Array A expected to be defined"
    assert a_array.ctype == "double", "Array A is of type `double`"
    assert a_array.offset == (0, 0, 0), "CartesianOffset.zeros() expected"
    assert "b" in sdfg.symbols.keys(), "expected `b` as scalar parameter"

    states = sdfg.nodes()
    assert len(states) >= 1, "at least one state expected"

    last_block = states[len(states) - 1]
    nodes = last_block.nodes()
    assert (
        len(list(filter(lambda node: isinstance(node, StencilComputation), nodes))) == 1
    ), "one StencilComputation library node"
    assert (
        len(
            list(
                filter(
                    lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "A", nodes
                )
            )
        )
        == 1
    ), "one AccessNode of A"

    edges = last_block.edges()
    library_node = list(filter(lambda node: isinstance(node, StencilComputation), nodes))[0]
    write_access = list(
        filter(lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "A", nodes)
    )[0]
    assert len(edges) == 1, "write memlet path expected"
    assert edges[0].src == library_node and edges[0].dst == write_access, "write access expected"

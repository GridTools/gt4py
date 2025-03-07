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

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.common import DataType
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gt4py.cartesian.gtc.dace.nodes import StencilComputation
from gt4py.cartesian.gtc.dace.oir_to_dace import OirSDFGBuilder
from gt4py.cartesian.gtc.dace.expansion.expansion import StencilComputationExpansion

from cartesian_tests.unit_tests.test_gtc.oir_utils import (
    AssignStmtFactory,
    BinaryOpFactory,
    FieldAccessFactory,
    FieldDeclFactory,
    HorizontalExecutionFactory,
    LiteralFactory,
    LocalScalarFactory,
    MaskStmtFactory,
    ScalarAccessFactory,
    StencilFactory,
)

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def _dcir_from_stencil(stencil: StencilFactory) -> list[dcir.NestedSDFG]:
    sdfg = OirSDFGBuilder().visit(stencil)
    assert isinstance(sdfg, dace.SDFG)

    expansions = []
    for state in sdfg.nodes():
        for node in state.nodes():
            if not isinstance(node, StencilComputation):
                continue

            arrays = StencilComputationExpansion._get_parent_arrays(node, state, sdfg)
            nested_SDFG = DaCeIRBuilder().visit(
                node.oir_node,
                global_ctx=DaCeIRBuilder.GlobalContext(library_node=node, arrays=arrays),
            )
            expansions.append(nested_SDFG)

    return expansions


def test_dcir_code_structure() -> None:
    stencil = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left=ScalarAccessFactory(name="tmp"),
                        right=BinaryOpFactory(
                            left=LiteralFactory(value="0"), right=LiteralFactory(value="2")
                        ),
                    ),
                    MaskStmtFactory(),
                    AssignStmtFactory(
                        left=ScalarAccessFactory(name="other"),
                        right=ScalarAccessFactory(name="tmp"),
                    ),
                ],
                declarations=[LocalScalarFactory(name="tmp"), LocalScalarFactory(name="other")],
            ),
        ]
    )
    expansions = _dcir_from_stencil(stencil)
    # We can assert the following structure
    # - ComputationState
    # - Condition
    #   - true_state: [ComputationState]
    #   - false_state: []
    # - ComputationState
    # from the above example

    # Let's have a similar one for WhileLoops, e.g.
    # - ComputationState
    # - WhileLoop
    #   - body: [ComputationState]
    # - ComputationState

    # We can't really validate the access nodes.
    # (the one that Florian was asking for)
    # Let's do this in a test that is processing the daceir to an actual sdfg
    assert len(expansions) == 1, "expect one vertical loop to be expanded"

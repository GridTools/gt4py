# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc.dace import daceir as dcir

from cartesian_tests.unit_tests.test_gtc.dace import utils
from cartesian_tests.unit_tests.test_gtc.oir_utils import (
    AssignStmtFactory,
    BinaryOpFactory,
    HorizontalExecutionFactory,
    LiteralFactory,
    LocalScalarFactory,
    MaskStmtFactory,
    ScalarAccessFactory,
    StencilFactory,
    WhileFactory,
)


# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_dcir_code_structure_condition() -> None:
    """Tests the following code structure:

    ComputationState
    Condition
        true_states: [ComputationState]
        false_states: []
    ComputationState
    """
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
    expansions = utils.library_node_expansions(stencil)
    assert len(expansions) == 1, "expect one vertical loop to be expanded"

    nested_SDFG = utils.nested_SDFG_inside_triple_loop(expansions[0])
    assert isinstance(nested_SDFG.states[0], dcir.ComputationState)
    assert isinstance(nested_SDFG.states[1], dcir.Condition)
    assert nested_SDFG.states[1].true_states
    assert isinstance(nested_SDFG.states[1].true_states[0], dcir.ComputationState)
    assert not nested_SDFG.states[1].false_states
    assert isinstance(nested_SDFG.states[2], dcir.ComputationState)


def test_dcir_code_structure_while() -> None:
    """Tests the following code structure

    ComputationState
    WhileLoop
        body: [ComputationState]
    ComputationState
    """
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
                    WhileFactory(),
                    AssignStmtFactory(
                        left=ScalarAccessFactory(name="other"),
                        right=ScalarAccessFactory(name="tmp"),
                    ),
                ],
                declarations=[LocalScalarFactory(name="tmp"), LocalScalarFactory(name="other")],
            ),
        ]
    )
    expansions = utils.library_node_expansions(stencil)
    assert len(expansions) == 1, "expect one vertical loop to be expanded"

    nested_SDFG = utils.nested_SDFG_inside_triple_loop(expansions[0])
    assert isinstance(nested_SDFG.states[0], dcir.ComputationState)
    assert isinstance(nested_SDFG.states[1], dcir.WhileLoop)
    assert nested_SDFG.states[1].body
    assert isinstance(nested_SDFG.states[1].body[0], dcir.ComputationState)
    assert isinstance(nested_SDFG.states[2], dcir.ComputationState)

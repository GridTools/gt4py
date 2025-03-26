# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace

from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gt4py.cartesian.gtc.dace.nodes import StencilComputation
from gt4py.cartesian.gtc.dace.oir_to_dace import OirSDFGBuilder
from gt4py.cartesian.gtc.dace.expansion.expansion import StencilComputationExpansion

from cartesian_tests.unit_tests.test_gtc.oir_utils import StencilFactory


def library_node_expansions(stencil: StencilFactory) -> list[dcir.NestedSDFG]:
    """Return all expanded library nodes in a given stencil."""
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


def nested_SDFG_inside_triple_loop(nSDFG: dcir.NestedSDFG) -> dcir.NestedSDFG:
    """Pick the inner nested SDFG out of the triple loop."""
    assert isinstance(nSDFG, dcir.NestedSDFG)
    assert isinstance(nSDFG.states[0], dcir.ComputationState)
    assert isinstance(nSDFG.states[0].computations[0], dcir.DomainMap)
    assert isinstance(nSDFG.states[0].computations[0].computations[0], dcir.DomainMap)
    assert isinstance(
        nSDFG.states[0].computations[0].computations[0].computations[0], dcir.DomainMap
    )
    assert isinstance(
        nSDFG.states[0].computations[0].computations[0].computations[0].computations[0],
        dcir.NestedSDFG,
    )
    return nSDFG.states[0].computations[0].computations[0].computations[0].computations[0]

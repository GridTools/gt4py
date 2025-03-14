# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest

from gt4py.cartesian.gtc.common import BuiltInLiteral, DataType
from gt4py.cartesian.gtc.dace.expansion.sdfg_builder import StencilComputationSDFGBuilder

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
)


# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_scalar_access_multiple_tasklets() -> None:
    """Test scalar access if an oir.CodeBlock is split over multiple Tasklets.

    We are breaking up vertical loops inside stencils in multiple Tasklets. It might thus happen that
    we write a "local" scalar in one Tasklet and read it in another Tasklet (downstream).
    We thus create output connectors for all writes to scalar variables inside Tasklets. And input
    connectors for all scalar reads unless previously written in the same Tasklet. DaCe's simplify
    pipeline will get rid of any dead dataflow introduced with this general approach.
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
                    MaskStmtFactory(
                        mask=LiteralFactory(value=BuiltInLiteral.TRUE, dtype=DataType.BOOL), body=[]
                    ),
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
    nsdfg = StencilComputationSDFGBuilder().visit(expansions[0])
    assert isinstance(nsdfg.sdfg, dace.SDFG)

    for node in nsdfg.sdfg.nodes()[1].nodes():
        if not isinstance(node, dace.nodes.NestedSDFG):
            continue

        nested = node.sdfg
        for state in nested.states():
            if state.name == "block_0":
                nodes = state.nodes()
                assert (
                    len(list(filter(lambda node: isinstance(node, dace.nodes.Tasklet), nodes))) == 1
                )
                assert (
                    len(
                        list(
                            filter(
                                lambda node: isinstance(node, dace.nodes.AccessNode)
                                and node.data == "tmp",
                                nodes,
                            )
                        )
                    )
                    == 1
                ), "one AccessNode of tmp"

                edges = state.edges()
                tasklet = list(filter(lambda node: isinstance(node, dace.nodes.Tasklet), nodes))[0]
                write_access = list(
                    filter(
                        lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "tmp",
                        nodes,
                    )
                )[0]
                assert len(edges) == 1, "one edge expected"
                assert (
                    edges[0].src == tasklet and edges[0].dst == write_access
                ), "write access of 'tmp'"

            if state.name == "block_1":
                nodes = state.nodes()
                assert (
                    len(list(filter(lambda node: isinstance(node, dace.nodes.Tasklet), nodes))) == 1
                )
                assert (
                    len(
                        list(
                            filter(
                                lambda node: isinstance(node, dace.nodes.AccessNode)
                                and node.data == "tmp",
                                nodes,
                            )
                        )
                    )
                    == 1
                ), "one AccessNode of tmp"

                edges = state.edges()
                tasklet = list(filter(lambda node: isinstance(node, dace.nodes.Tasklet), nodes))[0]
                read_access = list(
                    filter(
                        lambda node: isinstance(node, dace.nodes.AccessNode) and node.data == "tmp",
                        nodes,
                    )
                )[0]
                write_access = list(
                    filter(
                        lambda node: isinstance(node, dace.nodes.AccessNode)
                        and node.data == "other",
                        nodes,
                    )
                )[0]
                assert len(edges) == 2, "two edges expected"
                assert (
                    edges[0].src == tasklet and edges[0].dst == write_access
                ), "write access of 'other'"
                assert (
                    edges[1].src == read_access and edges[1].dst == tasklet
                ), "read access of 'tmp'"

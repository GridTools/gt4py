# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from numpy import int32, int64

from gt4py import next as gtx
from gt4py.eve import SymbolRef
from gt4py.next import NeighborTableOffsetProvider, common
from gt4py.next.program_processors import otf_compile_executor
from gt4py.next.program_processors.runners.gtfn import run_gtfn_with_temporaries
from tests.next_tests.integration_tests.cases import Case
from tests.next_tests.toy_connectivity import Cell, Edge

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import E2V, KDim, Vertex, cartesian_case, unstructured_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    reduction_setup,
)


run_gtfn_with_temporaries_and_sizes = otf_compile_executor.OTFBackend(
    executor=otf_compile_executor.OTFCompileExecutor(
        name="run_gtfn_with_temporaries_and_sizes",
        otf_workflow=run_gtfn_with_temporaries.executor.otf_workflow.replace(
            translation=run_gtfn_with_temporaries.executor.otf_workflow.translation.replace(
                symbolic_domain_sizes={
                    "Cell": "num_cells",
                    "Edge": "num_edges",
                    "Vertex": "num_vertices",
                },
            ),
        ),
    ),
    allocator=run_gtfn_with_temporaries.allocator,
)


@pytest.fixture
def prepare_testee(reduction_setup):
    @gtx.field_operator
    def testee_op(a: cases.VField) -> cases.EField:
        amul = a * 2
        return amul(E2V[0]) + amul(E2V[1])

    @gtx.program
    def prog(
        a: cases.VField,
        out: cases.EField,
        num_vertices: int32,
        num_edges: int64,
        num_cells: int32,
    ):
        testee_op(a, out=out)

    unstructured_case = Case(
        run_gtfn_with_temporaries_and_sizes,
        offset_provider=reduction_setup.offset_provider,
        default_sizes={
            Vertex: reduction_setup.num_vertices,
            Edge: reduction_setup.num_edges,
            Cell: reduction_setup.num_cells,
            KDim: reduction_setup.k_levels,
        },
        grid_type=common.GridType.UNSTRUCTURED,
    )

    a = cases.allocate(unstructured_case, prog, "a")()
    out = cases.allocate(unstructured_case, prog, "out")()

    return unstructured_case, a, out, prog


class TestTemporariesWithSizes:
    def test_verification(self, prepare_testee, reduction_setup):
        unstructured_case, a, out, prog = prepare_testee

        ref = (a.ndarray * 2)[unstructured_case.offset_provider["E2V"].table[:, 0]] + (
            a.ndarray * 2
        )[unstructured_case.offset_provider["E2V"].table[:, 1]]

        cases.verify(
            unstructured_case,
            prog,
            a,
            out,
            reduction_setup.num_vertices,
            reduction_setup.num_edges,
            reduction_setup.num_cells,
            inout=out,
            ref=ref,
        )

    def test_temporary_symbols(self, prepare_testee, reduction_setup):
        unstructured_case, a, out, prog = prepare_testee

        e2v_offset_provider = {
            "E2V": NeighborTableOffsetProvider(
                table=reduction_setup.e2v_table,
                origin_axis=Edge,
                neighbor_axis=Vertex,
                max_neighbors=2,
            )
        }

        ir_with_tmp = run_gtfn_with_temporaries_and_sizes.executor.otf_workflow.translation._preprocess_program(
            prog.itir, e2v_offset_provider
        )

        params = ["num_vertices", "num_edges", "num_cells"]
        for param in params:
            assert any([param == str(p) for p in ir_with_tmp.fencil.params])

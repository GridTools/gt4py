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
from gt4py.next import common
from gt4py.next.iterator.transforms import LiftMode, apply_common_transforms
from gt4py.next.program_processors import otf_compile_executor
from gt4py.next.program_processors.runners.gtfn import run_gtfn_with_temporaries

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    Case,
    KDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    mesh_descriptor,
)
from next_tests.toy_connectivity import Cell, Edge


@pytest.fixture
def run_gtfn_with_temporaries_and_symbolic_sizes():
    return otf_compile_executor.OTFBackend(
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
def testee():
    @gtx.field_operator
    def testee_op(a: cases.VField) -> cases.EField:
        amul = a * 2
        return amul(E2V[0]) + amul(E2V[1])

    @gtx.program
    def prog(
        a: cases.VField,
        out: cases.EField,
        num_vertices: int32,
        num_edges: int32,
        num_cells: int32,
    ):
        testee_op(a, out=out)

    return prog


def test_verification(testee, run_gtfn_with_temporaries_and_symbolic_sizes, mesh_descriptor):
    unstructured_case = Case(
        run_gtfn_with_temporaries_and_symbolic_sizes.executor,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Vertex: mesh_descriptor.num_vertices,
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
            KDim: 10,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=run_gtfn_with_temporaries_and_symbolic_sizes.allocator,
    )

    a = cases.allocate(unstructured_case, testee, "a")()
    out = cases.allocate(unstructured_case, testee, "out")()

    first_nbs, second_nbs = (mesh_descriptor.offset_provider["E2V"].table[:, i] for i in [0, 1])
    ref = (a.ndarray * 2)[first_nbs] + (a.ndarray * 2)[second_nbs]

    cases.verify(
        unstructured_case,
        testee,
        a,
        out,
        mesh_descriptor.num_vertices,
        mesh_descriptor.num_edges,
        mesh_descriptor.num_cells,
        inout=out,
        ref=ref,
    )


def test_temporary_symbols(testee, mesh_descriptor):
    itir_with_tmp = apply_common_transforms(
        testee.itir,
        lift_mode=LiftMode.USE_TEMPORARIES,
        offset_provider=mesh_descriptor.offset_provider,
    )

    params = ["num_vertices", "num_edges", "num_cells"]
    for param in params:
        assert any([param == str(p) for p in itir_with_tmp.fencil.params])

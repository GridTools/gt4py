# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from numpy import int32

from gt4py import next as gtx
from gt4py.next import backend, common
from gt4py.next.iterator.transforms import apply_common_transforms
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    Case,
    KDim,
    Vertex,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    mesh_descriptor,
)
from next_tests.toy_connectivity import Cell, Edge


# Override the exec_alloc_descriptor with a custom Backend,
# see https://docs.pytest.org/en/latest/how-to/fixtures.html#override-a-fixture-on-a-test-module-level
@pytest.fixture
def exec_alloc_descriptor():
    return backend.Backend(
        name="run_gtfn_with_temporaries_and_sizes",
        transforms=backend.DEFAULT_TRANSFORMS,
        executor=run_gtfn.executor.replace(
            translation=run_gtfn.executor.translation.replace(
                symbolic_domain_sizes={
                    "Cell": "num_cells",
                    "Edge": "num_edges",
                    "Vertex": "num_vertices",
                }
            )
        ),
        allocator=run_gtfn.allocator,
    )


@pytest.fixture
def testee():
    @gtx.field_operator
    def testee_op(a: cases.VField) -> cases.EField:
        amul = a * 2
        return amul(E2V[0]) + amul(E2V[1])

    @gtx.program
    def prog(
        a: cases.VField, out: cases.EField, num_vertices: int32, num_edges: int32, num_cells: int32
    ):
        testee_op(a, out=out)

    return prog


# @pytest.mark.parametrize("exec_alloc_descriptor", [run_gtfn_with_temporaries_and_symbolic_sizes])
def test_verification(testee, exec_alloc_descriptor, mesh_descriptor):
    unstructured_case = Case(
        exec_alloc_descriptor,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Vertex: mesh_descriptor.num_vertices,
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
            KDim: 10,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )

    a = cases.allocate(unstructured_case, testee, "a")()
    out = cases.allocate(unstructured_case, testee, "out")()

    first_nbs, second_nbs = (mesh_descriptor.offset_provider["E2V"].asnumpy()[:, i] for i in [0, 1])
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
    gtir_with_tmp = apply_common_transforms(
        testee.gtir,
        extract_temporaries=True,
        offset_provider=mesh_descriptor.offset_provider,
    )

    params = ["num_vertices", "num_edges", "num_cells"]
    for param in params:
        assert any([param == str(p) for p in gtir_with_tmp.params])

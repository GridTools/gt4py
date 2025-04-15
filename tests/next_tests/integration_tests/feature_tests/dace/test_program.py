# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next import common

from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    Edge,
    IDim,
    JDim,
    KDim,
    Vertex,
    mesh_descriptor,  # noqa: F401
)


dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners import dace as dace_backends


# Override the exec_alloc_descriptor with a custom Backend,
# see https://docs.pytest.org/en/latest/how-to/fixtures.html#override-a-fixture-on-a-test-module-level
@pytest.fixture(
    params=[
        pytest.param(dace_backends.run_dace_cpu, marks=pytest.mark.requires_dace),
        pytest.param(
            dace_backends.run_dace_gpu, marks=(pytest.mark.requires_gpu, pytest.mark.requires_dace)
        ),
    ]
)
def exec_alloc_descriptor(request):
    yield request.param


@pytest.fixture
def cartesian(request, gtir_dace_backend):
    yield cases.Case(
        backend=gtir_dace_backend,
        offset_provider={
            "Ioff": IDim,
            "Joff": JDim,
            "Koff": KDim,
        },
        default_sizes={IDim: 10, JDim: 10, KDim: 10},
        grid_type=common.GridType.CARTESIAN,
        allocator=gtir_dace_backend.allocator,
    )


@pytest.fixture
def unstructured(request, exec_alloc_descriptor, mesh_descriptor):  # noqa: F811
    yield cases.Case(
        backend=exec_alloc_descriptor,
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


def test_halo_exchange_helper_attrs(unstructured):
    local_int = gtx.int

    @gtx.field_operator(backend=unstructured.backend)
    def testee_op(
        a: gtx.Field[[Vertex, KDim], gtx.int],
    ) -> gtx.Field[[Vertex, KDim], gtx.int]:
        return a + local_int(10)

    @gtx.program(backend=unstructured.backend)
    def testee_prog(
        a: gtx.Field[[Vertex, KDim], gtx.int],
        b: gtx.Field[[Vertex, KDim], gtx.int],
        c: gtx.Field[[Vertex, KDim], gtx.int],
    ):
        testee_op(b, out=c)
        testee_op(a, out=b)

    dace_storage_type = (
        dace.StorageType.GPU_Global
        if unstructured.backend == dace_backends.run_dace_gpu
        else dace.StorageType.Default
    )

    rows = dace.symbol("rows")
    cols = dace.symbol("cols")

    @dace.program
    def testee_dace(
        a: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        b: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        c: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
    ):
        testee_prog(a, b, c)

    # if simplify=True, DaCe might inline the nested SDFG coming from Program.__sdfg__,
    # effectively erasing the attributes we want to test for here
    sdfg = testee_dace.to_sdfg(simplify=False)

    testee = next(
        subgraph for subgraph in sdfg.all_sdfgs_recursive() if subgraph.name == "testee_prog"
    )

    assert testee.gt4py_program_input_fields == {"a": Vertex, "b": Vertex}
    assert testee.gt4py_program_output_fields == {"b": Vertex, "c": Vertex}

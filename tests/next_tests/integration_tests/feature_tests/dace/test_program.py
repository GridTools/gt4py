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
from gt4py.next.program_processors.runners.dace_fieldview import program as dace_prg

from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    Edge,
    IDim,
    JDim,
    KDim,
    Vertex,
    mesh_descriptor,
)


try:
    import dace
    from gt4py.next.program_processors.runners.dace import gtir_cpu, gtir_gpu
except ImportError:
    from typing import Optional
    from types import ModuleType
    from gt4py.next import backend as next_backend

    dace: Optional[ModuleType] = None
    gtir_cpu: Optional[next_backend.Backend] = None
    gtir_gpu: Optional[next_backend.Backend] = None


@pytest.fixture(
    params=[
        pytest.param(gtir_cpu, marks=pytest.mark.requires_dace),
        pytest.param(gtir_gpu, marks=(pytest.mark.requires_gpu, pytest.mark.requires_dace)),
    ]
)
def gtir_dace_backend(request):
    yield request.param


@pytest.fixture
def cartesian(request, gtir_dace_backend):
    if gtir_dace_backend is None:
        yield None

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
def unstructured(request, gtir_dace_backend, mesh_descriptor):
    if gtir_dace_backend is None:
        yield None

    yield cases.Case(
        backend=gtir_dace_backend,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Vertex: mesh_descriptor.num_vertices,
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
            KDim: 10,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=gtir_dace_backend.allocator,
    )


@pytest.mark.skipif(dace is None, reason="DaCe not found")
def test_input_names_extractor_cartesian(cartesian):
    @gtx.field_operator(backend=cartesian.backend)
    def testee_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program(backend=cartesian.backend)
    def testee(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        testee_op(b, out=c)
        testee_op(a, out=b)

    input_field_names = dace_prg.InputNamesExtractor.only_fields(testee.itir)
    assert input_field_names == {"a", "b"}


@pytest.mark.skipif(dace is None, reason="DaCe not found")
def test_output_names_extractor(cartesian):
    @gtx.field_operator(backend=cartesian.backend)
    def testee_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program(backend=cartesian.backend)
    def testee(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        testee_op(a, out=b)
        testee_op(a, out=c)

    output_field_names = dace_prg.OutputNamesExtractor.only_fields(testee.itir)
    assert output_field_names == {"b", "c"}


@pytest.mark.skipif(dace is None, reason="DaCe not found")
def test_halo_exchange_helper_attrs(unstructured):
    @gtx.field_operator(backend=unstructured.backend)
    def testee_op(
        a: gtx.Field[[Vertex, KDim], gtx.int],
    ) -> gtx.Field[[Vertex, KDim], gtx.int]:
        return a

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
        if unstructured.backend == gtir_gpu
        else dace.StorageType.Default
    )

    rows = dace.symbol("rows")
    cols = dace.symbol("cols")
    OffsetProvider_t = dace.data.Structure(
        {
            key: dace.data.Array(dtype=dace.int64, shape=[rows, cols], storage=dace_storage_type)
            for key in unstructured.offset_provider
        },
        name="OffsetProvider",
    )

    @dace.program
    def testee_dace(
        a: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        b: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        c: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        offset_provider: OffsetProvider_t,
        connectivities: dace.compiletime,
    ):
        testee_prog.with_grid_type(unstructured.grid_type).with_connectivities(connectivities)(
            a, b, c, offset_provider=offset_provider
        )

    sdfg = testee_dace.to_sdfg(connectivities=unstructured.offset_provider)

    testee = next(
        subgraph for subgraph in sdfg.all_sdfgs_recursive() if subgraph.name == "testee_prog"
    )

    assert testee.gt4py_program_input_fields == {"a": Vertex, "b": Vertex}
    assert testee.gt4py_program_output_fields == {"b": Vertex, "c": Vertex}
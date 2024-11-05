# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Optional
from types import ModuleType
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend
from gt4py.next.otf import arguments

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, unstructured_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
    Vertex,
    Edge,
    E2V,
)
from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_laplacian import (
    lap_program,
    laplap_program,
    lap_ref,
)

try:
    import dace
    from gt4py.next.program_processors.runners.dace import (
        itir_cpu as run_dace_cpu,
        itir_gpu as run_dace_gpu,
    )
except ImportError:
    dace: Optional[ModuleType] = None  # type:ignore[no-redef]
    run_dace_cpu: Optional[next_backend.Backend] = None
    run_dace_gpu: Optional[next_backend.Backend] = None

pytestmark = pytest.mark.requires_dace


def test_sdfgConvertible_laplap(cartesian_case):
    # TODO(kotsaloscv): Temporary solution until the `requires_dace` marker is fully functional
    if cartesian_case.backend not in [run_dace_cpu, run_dace_gpu]:
        pytest.skip("DaCe-related test: Test SDFGConvertible interface for GT4Py programs")

    if cartesian_case.backend == run_dace_gpu:
        import cupy as xp
    else:
        import numpy as xp

    in_field = cases.allocate(cartesian_case, laplap_program, "in_field")()
    out_field = cases.allocate(cartesian_case, laplap_program, "out_field")()

    connectivities = {}  # Dict of NeighborOffsetProviders, where self.table = None
    for k, v in cartesian_case.offset_provider.items():
        if hasattr(v, "table"):
            connectivities[k] = arguments.CompileTimeConnectivity(
                v.max_neighbors, v.has_skip_values, v.origin_axis, v.neighbor_axis, v.table.dtype
            )
        else:
            connectivities[k] = v

    # Test DaCe closure support
    @dace.program
    def sdfg():
        tmp_field = xp.empty_like(out_field)
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            cartesian_case.backend
        ).with_connectivities(connectivities)(in_field, tmp_field)
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            cartesian_case.backend
        ).with_connectivities(connectivities)(tmp_field, out_field)

    sdfg()

    assert np.allclose(
        gtx.field_utils.asnumpy(out_field)[2:-2, 2:-2],
        lap_ref(lap_ref(in_field.array_ns.asarray(in_field.ndarray))),
    )


@gtx.field_operator
def _testee(a: gtx.Field[gtx.Dims[Vertex], gtx.float64]):
    return a(E2V[0])


@gtx.program
def testee(a: gtx.Field[gtx.Dims[Vertex], gtx.float64], b: gtx.Field[gtx.Dims[Edge], gtx.float64]):
    _testee(a, out=b)


@pytest.mark.uses_unstructured_shift
def test_sdfgConvertible_connectivities(unstructured_case):
    # TODO(kotsaloscv): Temporary solution until the `requires_dace` marker is fully functional
    if unstructured_case.backend not in [run_dace_cpu, run_dace_gpu]:
        pytest.skip("DaCe-related test: Test SDFGConvertible interface for GT4Py programs")

    allocator, backend = unstructured_case.allocator, unstructured_case.backend

    if backend == run_dace_gpu:
        import cupy as xp

        dace_storage_type = dace.StorageType.GPU_Global
    else:
        import numpy as xp

        dace_storage_type = dace.StorageType.Default

    rows = dace.symbol("rows")
    cols = dace.symbol("cols")
    OffsetProvider_t = dace.data.Structure(
        dict(E2V=dace.data.Array(dtype=dace.int64, shape=[rows, cols], storage=dace_storage_type)),
        name="OffsetProvider",
    )

    @dace.program
    def sdfg(
        a: dace.data.Array(dtype=dace.float64, shape=(rows,), storage=dace_storage_type),
        out: dace.data.Array(dtype=dace.float64, shape=(rows,), storage=dace_storage_type),
        offset_provider: OffsetProvider_t,
        connectivities: dace.compiletime,
    ):
        testee.with_backend(backend).with_connectivities(connectivities)(
            a, out, offset_provider=offset_provider
        )

    e2v = gtx.NeighborTableOffsetProvider(
        xp.asarray([[0, 1], [1, 2], [2, 0]]), Edge, Vertex, 2, False
    )
    connectivities = {}
    connectivities["E2V"] = arguments.CompileTimeConnectivity(
        e2v.max_neighbors, e2v.has_skip_values, e2v.origin_axis, e2v.neighbor_axis, e2v.table.dtype
    )
    offset_provider = OffsetProvider_t.dtype._typeclass.as_ctypes()(E2V=e2v.data_ptr())

    SDFG = sdfg.to_sdfg(connectivities=connectivities)
    cSDFG = SDFG.compile()

    a = gtx.as_field([Vertex], xp.asarray([0.0, 1.0, 2.0]), allocator=allocator)
    out = gtx.zeros({Edge: 3}, allocator=allocator)
    # This is a low level interface to call the compiled SDFG.
    # It is not supposed to be used in user code.
    # The high level interface should be provided by a DaCe Orchestrator,
    # i.e. decorator that hides the low level operations.
    # This test checks only that the SDFGConvertible interface works correctly.
    cSDFG(
        a,
        out,
        offset_provider,
        rows=3,
        cols=2,
        connectivity_E2V=e2v.table,
        __connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(
            xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table, 0
        ),
        __connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(
            xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table, 1
        ),
    )

    e2v_xp = xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table
    assert np.allclose(gtx.field_utils.asnumpy(out), gtx.field_utils.asnumpy(a)[e2v_xp[:, 0]])

    e2v = gtx.NeighborTableOffsetProvider(
        xp.asarray([[1, 0], [2, 1], [0, 2]]), Edge, Vertex, 2, False
    )
    offset_provider = OffsetProvider_t.dtype._typeclass.as_ctypes()(E2V=e2v.data_ptr())
    cSDFG(
        a,
        out,
        offset_provider,
        rows=3,
        cols=2,
        connectivity_E2V=e2v.table,
        __connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(
            xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table, 0
        ),
        __connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(
            xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table, 1
        ),
    )

    e2v_xp = xp.asnumpy(e2v.table) if backend == run_dace_gpu else e2v.table
    assert np.allclose(gtx.field_utils.asnumpy(out), gtx.field_utils.asnumpy(a)[e2v_xp[:, 0]])


def get_stride_from_numpy_to_dace(numpy_array: np.ndarray, axis: int) -> int:
    """
    NumPy strides: number of bytes to jump
    DaCe strides: number of elements to jump
    """
    return numpy_array.strides[axis] // numpy_array.itemsize

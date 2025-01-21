# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import allocators as gtx_allocators, common as gtx_common

from gt4py._core import definitions as core_defs
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, unstructured_case  # noqa: F401
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    E2V,
    E2VDim,
    Edge,
    Vertex,
    exec_alloc_descriptor,  # noqa: F401
    mesh_descriptor,  # noqa: F401
)
from next_tests.integration_tests.multi_feature_tests.ffront_tests.test_laplacian import (
    lap_program,
    lap_ref,
    laplap_program,
)


try:
    import dace
except ImportError:
    dace: Optional[ModuleType] = None  # type:ignore[no-redef]

pytestmark = pytest.mark.requires_dace


def test_sdfgConvertible_laplap(cartesian_case):  # noqa: F811
    if not cartesian_case.backend or "dace" not in cartesian_case.backend.name:
        pytest.skip("DaCe-related test: Test SDFGConvertible interface for GT4Py programs")

    # TODO(edopao): add support for range symbols in field domain and re-enable this test
    pytest.skip("Requires support for field domain range.")

    backend = cartesian_case.backend

    in_field = cases.allocate(cartesian_case, laplap_program, "in_field")()
    out_field = cases.allocate(cartesian_case, laplap_program, "out_field")()

    xp = in_field.array_ns

    # Test DaCe closure support
    @dace.program
    def sdfg():
        tmp_field = xp.empty_like(out_field)
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            backend
        ).with_connectivities(gtx_common.offset_provider_to_type(cartesian_case.offset_provider))(
            in_field, tmp_field
        )
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            backend
        ).with_connectivities(gtx_common.offset_provider_to_type(cartesian_case.offset_provider))(
            tmp_field, out_field
        )

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
def test_sdfgConvertible_connectivities(unstructured_case):  # noqa: F811
    if not unstructured_case.backend or "dace" not in unstructured_case.backend.name:
        pytest.skip("DaCe-related test: Test SDFGConvertible interface for GT4Py programs")

    # TODO(edopao): add support for range symbols in field domain and re-enable this test
    pytest.skip("Requires support for field domain range.")

    allocator, backend = unstructured_case.allocator, unstructured_case.backend

    if gtx_allocators.is_field_allocator_for(allocator, gtx_allocators.CUPY_DEVICE):
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

    e2v = gtx.as_connectivity(
        [Edge, E2VDim],
        codomain=Vertex,
        data=xp.asarray([[0, 1], [1, 2], [2, 0]]),
        allocator=allocator,
    )

    testee2 = testee.with_backend(backend).with_connectivities({"E2V": e2v})

    @dace.program
    def sdfg(
        a: dace.data.Array(dtype=dace.float64, shape=(rows,), storage=dace_storage_type),
        out: dace.data.Array(dtype=dace.float64, shape=(rows,), storage=dace_storage_type),
        offset_provider: OffsetProvider_t,
        connectivities: dace.compiletime,
    ):
        testee2.with_connectivities(connectivities)(a, out, offset_provider=offset_provider)
        return out

    connectivities = {"E2V": e2v}  # replace 'e2v' with 'e2v.__gt_type__()' when GTIR is AOT
    offset_provider = OffsetProvider_t.dtype._typeclass.as_ctypes()(E2V=e2v.data_ptr())

    SDFG = sdfg.to_sdfg(connectivities=connectivities)
    cSDFG = SDFG.compile()

    a = gtx.as_field([Vertex], xp.asarray([0.0, 1.0, 2.0]), allocator=allocator)
    out = gtx.zeros({Edge: 3}, allocator=allocator)

    def get_stride_from_numpy_to_dace(arg: core_defs.NDArrayObject, axis: int) -> int:
        # NumPy strides: number of bytes to jump
        # DaCe strides: number of elements to jump
        return arg.strides[axis] // arg.itemsize

    cSDFG(
        a,
        out,
        offset_provider,
        rows=3,
        cols=2,
        connectivity_E2V=e2v,
        __connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(e2v.ndarray, 0),
        __connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(e2v.ndarray, 1),
    )

    e2v_np = e2v.asnumpy()
    assert np.allclose(out.asnumpy(), a.asnumpy()[e2v_np[:, 0]])

    e2v = gtx.as_connectivity(
        [Edge, E2VDim],
        codomain=Vertex,
        data=xp.asarray([[1, 0], [2, 1], [0, 2]]),
        allocator=allocator,
    )
    offset_provider = OffsetProvider_t.dtype._typeclass.as_ctypes()(E2V=e2v.data_ptr())
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        cSDFG(
            a,
            out,
            offset_provider,
            rows=3,
            cols=2,
            connectivity_E2V=e2v,
            __connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(e2v.ndarray, 0),
            __connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(e2v.ndarray, 1),
        )

    e2v_np = e2v.asnumpy()
    assert np.allclose(out.asnumpy(), a.asnumpy()[e2v_np[:, 0]])


def get_stride_from_numpy_to_dace(numpy_array: np.ndarray, axis: int) -> int:
    """
    NumPy strides: number of bytes to jump
    DaCe strides: number of elements to jump
    """
    return numpy_array.strides[axis] // numpy_array.itemsize

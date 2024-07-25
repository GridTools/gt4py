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

import numpy as np
from typing import Optional
from types import ModuleType
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend

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
    from gt4py.next.program_processors.runners.dace import run_dace_cpu, run_dace_gpu
except ImportError:
    dace: Optional[ModuleType] = None  # type:ignore[no-redef]
    run_dace_cpu: Optional[next_backend.Backend] = None
    run_dace_gpu: Optional[next_backend.Backend] = None

pytestmark = pytest.mark.requires_dace


def test_sdfgConvertible_laplap(cartesian_case):
    if cartesian_case.executor == run_dace_gpu:
        import cupy as xp
    else:
        import numpy as xp

    in_field = cases.allocate(cartesian_case, laplap_program, "in_field")()
    out_field = cases.allocate(cartesian_case, laplap_program, "out_field")()

    connectivities = {}  # Dict of NeighborOffsetProviders, where self.table = None
    for k, v in cartesian_case.offset_provider.items():
        if hasattr(v, "table"):
            connectivities[k] = gtx.CompileTimeConnectivity(
                v.max_neighbors, v.has_skip_values, v.origin_axis, v.neighbor_axis, v.table.dtype
            )
        else:
            connectivities[k] = v

    # Test DaCe closure support
    @dace.program
    def sdfg():
        tmp_field = xp.empty_like(out_field)
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            cartesian_case.executor
        ).with_connectivities(connectivities)(in_field, tmp_field)
        lap_program.with_grid_type(cartesian_case.grid_type).with_backend(
            cartesian_case.executor
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
    allocator, backend = unstructured_case.allocator, unstructured_case.executor

    rows = dace.symbol("rows")
    cols = dace.symbol("cols")
    OffsetProvider_t = dace.data.Structure(
        dict(E2V=dace.data.Array(dtype=dace.int64, shape=[rows, cols])), name="OffsetProvider"
    )

    @dace.program
    def sdfg(
        a: dace.data.Array(dtype=dace.float64, shape=(rows,)),
        out: dace.data.Array(dtype=dace.float64, shape=(rows,)),
        offset_provider: OffsetProvider_t,
        connectivities: dace.compiletime,
    ):
        testee.with_backend(backend).with_connectivities(connectivities)(
            a, out, offset_provider=offset_provider
        )

    e2v_array = np.asarray([[0, 1], [1, 2], [2, 0]])
    e2v = gtx.NeighborTableOffsetProvider(e2v_array, Edge, Vertex, 2, False)
    connectivities = {}
    connectivities["E2V"] = gtx.CompileTimeConnectivity(
        e2v.max_neighbors, e2v.has_skip_values, e2v.origin_axis, e2v.neighbor_axis, e2v.table.dtype
    )
    offset_provider = OffsetProvider_t.dtype._typeclass.as_ctypes()(E2V=e2v.data_ptr())

    SDFG = sdfg.to_sdfg(
        connectivities=connectivities
    )
    cSDFG = SDFG.compile()

    a_array = np.asarray([0.0, 1.0, 2.0])
    a = gtx.as_field([Vertex], a_array, allocator=allocator)
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
        __connectivity_E2V=e2v.table,
        ____connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(e2v.table, 0),
        ____connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(e2v.table, 1),
    )

    assert np.allclose(out.ndarray, a_array[e2v_array[:, 0]])

    e2v_array = np.asarray([[1, 0], [2, 1], [0, 2]])
    e2v = gtx.NeighborTableOffsetProvider(e2v_array, Edge, Vertex, 2, False)
    cSDFG(
        a,
        out,
        offset_provider,
        rows=3,
        cols=2,
        __connectivity_E2V=e2v.table,
        ____connectivity_E2V_stride_0=get_stride_from_numpy_to_dace(e2v.table, 0),
        ____connectivity_E2V_stride_1=get_stride_from_numpy_to_dace(e2v.table, 1),
    )

    assert np.allclose(out.ndarray, a_array[e2v_array[:, 0]])


def get_stride_from_numpy_to_dace(numpy_array: np.ndarray, axis: int) -> int:
    """
    NumPy strides: number of bytes to jump
    DaCe strides: number of elements to jump
    """
    return numpy_array.strides[axis] // numpy_array.itemsize

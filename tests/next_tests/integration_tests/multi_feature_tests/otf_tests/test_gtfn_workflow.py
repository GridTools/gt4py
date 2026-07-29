# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests.cases import IDim, JDim


def test_different_buffer_sizes():
    """
    Ensure correct bindings are generated for buffers of different sizes (strides).

    Note:
      gridtools::sid::composite share the same runtime strides if they are of the same kind at compile-time.
      This test would fail if the same strides kind is used for both storages.
    """
    nx = 6
    ny = 7
    out_nx = 5
    out_ny = 5

    inp = gtx.as_field([IDim, JDim], np.reshape(np.arange(nx * ny, dtype=np.int32), (nx, ny)))
    out = gtx.as_field([IDim, JDim], np.zeros((out_nx, out_ny), dtype=np.int32))

    @gtx.field_operator(backend=gtfn.run_gtfn)
    def copy(inp: gtx.Field[[IDim, JDim], gtx.int32]) -> gtx.Field[[IDim, JDim], gtx.int32]:
        return inp

    copy(inp, out=out, offset_provider={})

    assert np.allclose(inp[:out_nx, :out_ny].asnumpy(), out.asnumpy())

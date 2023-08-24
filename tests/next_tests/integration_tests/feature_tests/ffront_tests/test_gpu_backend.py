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

import gt4py.next as gtx
from gt4py.next.iterator import embedded
from gt4py.next.program_processors.runners import gtfn_cpu

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case  # noqa: F401
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401
    fieldview_backend,
)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("fieldview_backend", [gtfn_cpu.run_gtfn_gpu])
def test_copy(cartesian_case, fieldview_backend):  # noqa: F811 # fixtures
    try:
        import cupy as cp
    except:
        pytest.skip("cupy is not available on the system, cannot run GPU tests")

    @gtx.field_operator(backend=fieldview_backend)
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a

    inp_arr = cp.full(shape=(3, 4, 5), fill_value=3, dtype=cp.int32)
    outp_arr = cp.zeros_like(inp_arr)
    inp = embedded.np_as_located_field(cases.IDim, cases.JDim, cases.KDim)(inp_arr)
    outp = embedded.np_as_located_field(cases.IDim, cases.JDim, cases.KDim)(outp_arr)

    testee(inp, out=outp, offset_provider={})
    assert cp.allclose(inp_arr, outp_arr)

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

import warnings

import pytest

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case  # noqa: F401
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401
    fieldview_backend,
)


_GPU_BACKENDS = [gtfn.run_gtfn_gpu]

try:
    from gt4py.next.program_processors.runners import dace_iterator

    _GPU_BACKENDS.append(dace_iterator.run_dace_gpu)
except:
    warnings.warn("Skipping dace backend because dace module is not installed.")


@pytest.mark.requires_gpu
@pytest.mark.parametrize("fieldview_backend", _GPU_BACKENDS)
def test_copy(fieldview_backend):  # noqa: F811 # fixtures
    import cupy as cp

    @gtx.field_operator(backend=fieldview_backend)
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a

    domain = {
        cases.IDim: common.unit_range(3),
        cases.JDim: common.unit_range(4),
        cases.KDim: common.unit_range(5),
    }
    inp_field = gtx.full(domain, fill_value=3, allocator=fieldview_backend, dtype=cp.int32)
    out_field = gtx.zeros(domain, allocator=fieldview_backend, dtype=cp.int32)
    testee(inp_field, out=out_field, offset_provider={})
    assert cp.allclose(inp_field.ndarray, out_field.ndarray)

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
import pytest
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from gt4py import next as gtx
from gt4py.next.ffront.experimental import concat_where
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_boundary_same_size_fields(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(k == 0, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.zeros(interior.shape)
    ref[:, :, 0] = boundary.asnumpy()[:, :, 0]
    ref[:, :, 1:] = interior.asnumpy()[:, :, 1:]

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)


# TODO:
# - IJField as boundary
# - IJKField with 1 level as boundary
# - mask that contains multiple regions of true/false

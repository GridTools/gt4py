# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
from functools import reduce
from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    astype,
    broadcast,
    common,
    domain,
    errors,
    float32,
    float64,
    int32,
    int64,
    minimum,
    neighbor_sum,
    utils as gt_utils,

)
from gt4py.next.ffront.experimental import as_offset

from gt4py.next.ffront.experimental import concat_where


from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    Edge,
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)

pytestmark = pytest.mark.uses_concat_where



def test_concat_where(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(domain({KDim: (0, 1)}), ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground")()
    air = cases.allocate(cartesian_case, testee, "air")()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(k[np.newaxis, np.newaxis, :] == 0, ground.asnumpy(), air.asnumpy())
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


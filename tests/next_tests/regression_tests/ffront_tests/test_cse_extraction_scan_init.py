# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import gt4py.next as gtx
import numpy as np

from next_tests.integration_tests.cases import KDim, cartesian_case
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_scan
def test_scan_init_duplicated(cartesian_case):
    @gtx.scan_operator(axis=KDim, forward=True, init=((1.0,), (1.0,)))
    def testee_scan(
        state: tuple[tuple[float], tuple[float]], inp: float
    ) -> tuple[tuple[float], tuple[float]]:
        return (state[0][0] + inp,), (state[1][0] + inp,)

    @gtx.field_operator
    def testee(
        inp: gtx.Field[[KDim], float],
    ) -> tuple[tuple[gtx.Field[[KDim], float]], tuple[gtx.Field[[KDim], float]]]:
        return testee_scan(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        out=out,
        ref=(
            (np.cumsum(inp.asnumpy(), axis=0) + 1.0,),
            (np.cumsum(inp.asnumpy(), axis=0) + 1.0,),
        ),
    )

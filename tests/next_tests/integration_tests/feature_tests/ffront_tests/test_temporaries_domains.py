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
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.program_processors.runners import roundtrip

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import KDim, cartesian_case


# A temporary read or written outside its domain is often off by a single element, which gtfn
# does not detect; this backend extracts the same temporaries and checks every access.
@pytest.fixture
def exec_alloc_descriptor():
    return roundtrip.with_temporaries


@pytest.mark.uses_concat_where
@pytest.mark.uses_cartesian_shift
@pytest.mark.parametrize("flag", [True, False])
def test_if_selecting_field_in_shifted_concat_where(cartesian_case, flag):
    @gtx.field_operator
    def testee(a: cases.IJKFloatField, z: cases.IJKFloatField, flag: bool) -> cases.IJKFloatField:
        t = a if flag else a + 1.0
        return concat_where(KDim >= 0, t, z)(KDim - 1) + t

    a = cases.allocate(cartesian_case, testee, "a")()
    z = cases.allocate(cartesian_case, testee, "z").extend({KDim: (1, 0)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    t = a.asnumpy() if flag else a.asnumpy() + 1.0
    ref = np.concatenate((z.asnumpy()[:, :, :1], t[:, :, :-1]), axis=2) + t
    cases.verify(cartesian_case, testee, a, z, flag, out=out, ref=ref)


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_tuple_returns
@pytest.mark.parametrize("flag", [True, False])
def test_if_selecting_temporary_on_larger_domain(cartesian_case, flag):
    @gtx.field_operator
    def testee(
        a: cases.IJKFloatField, b: cases.IJKFloatField, flag: bool
    ) -> tuple[cases.IJKFloatField, cases.IJKFloatField]:
        u = a + 1.0
        return u(KDim + 1), u if flag else b

    a = cases.allocate(cartesian_case, testee, "a").extend({KDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    u = a.asnumpy() + 1.0
    ref = (u[:, :, 1:], u[:, :, :-1] if flag else b.asnumpy())
    cases.verify(cartesian_case, testee, a, b, flag, out=out, ref=ref)

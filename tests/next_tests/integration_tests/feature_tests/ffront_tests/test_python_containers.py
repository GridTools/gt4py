# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx
import dataclasses

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

# TODO: as long as direct fieldoperator calls and program calls follow a different code path we should probably repeat all tests for both


@dataclasses.dataclass
class Velocity:
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


def test_named_tuple_like_constructed_outside(cartesian_case):
    @gtx.field_operator
    def fop(
        vel: Velocity,
    ) -> gtx.Field[[IDim, JDim], gtx.float32]:
        return vel.u + vel.v

    @gtx.program
    def testee(
        vel: Velocity,
        out: gtx.Field[[IDim, JDim], gtx.float32],
    ) -> gtx.Field[[IDim, JDim], gtx.float32]:
        fop(vel, out=out)

    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(cartesian_case, testee, "out")()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out,
        inout=out,
        ref=(vel.u + vel.v),
    )


def test_named_tuple_like_constructed_inside(cartesian_case):
    @gtx.field_operator
    def testee(
        vel: Velocity,
    ) -> Velocity:
        return Velocity(v=vel.u - vel.v, u=vel.u + vel.v)  # order swapped to show kwargs work

    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out=out,
        ref=Velocity(u=vel.u + vel.v, v=vel.u - vel.v),
    )

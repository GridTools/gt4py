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
import dataclasses

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from gt4py.next.type_system import type_specifications as ts

pytestmark = pytest.mark.uses_cartesian_shift

# TODO move to a proper location

# IDim = gtx.Dimension("i")
# JDim = gtx.Dimension("j")


@dataclasses.dataclass
class Velocity:
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


# class Velocity:
#     def __init__(self, u, v):
#         self.u = u
#         self.v = v

#     def __gt_type__(self):
#         return ts.NamedTupleType(
#             types=[
#                 ts.FieldType(
#                     dims=[IDim, JDim],
#                     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                 ),
#                 ts.FieldType(
#                     dims=[IDim, JDim],
#                     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                 ),
#             ],
#             keys=["u", "v"],
#         )


@gtx.field_operator
def foo(
    vel: Velocity,
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def foo_program(
    vel: Velocity,
    out: gtx.Field[[IDim, JDim], gtx.float32],
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    foo(vel, out=out)


def test_foo(cartesian_case):
    vel = cases.allocate(cartesian_case, foo_program, "vel")()
    out = cases.allocate(cartesian_case, foo_program, "out")()

    cases.verify(
        cartesian_case,
        foo_program,
        vel,
        out,
        inout=out,
        ref=(vel.u + vel.v),
    )


@gtx.field_operator
def bar(
    vel: Velocity,
) -> tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]]:
    # ) -> Velocity:
    tmp = Velocity(vel.u + vel.v, vel.u - vel.v)
    return tmp.u, tmp.v  # later return Velocity directly


def test_bar(cartesian_case):
    vel = cases.allocate(cartesian_case, bar, "vel")()
    out = cases.allocate(cartesian_case, bar, cases.RETURN)()

    cases.verify(
        cartesian_case,
        bar,
        vel,
        out=out,
        ref=(vel.u + vel.v, vel.u - vel.v),
    )

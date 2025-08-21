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


def test_named_tuple_like_constructed_outside(cartesian_case):
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
) -> Velocity:
    return Velocity(v=vel.u - vel.v, u=vel.u + vel.v)  # order swapped to show kwargs work


def test_named_tuple_like_constructed_inside(cartesian_case):
    vel = cases.allocate(cartesian_case, bar, "vel")()
    out = cases.allocate(cartesian_case, bar, cases.RETURN)()

    cases.verify(
        cartesian_case,
        bar,
        vel,
        out=out,
        ref=Velocity(u=vel.u + vel.v, v=vel.u - vel.v),
    )


# - container with only index access and uniform type
# - container with only index access and runtime known size
# - implicit unroll of tuple_type with non-tuple type (could be implemented on standard tuples independently?)


# - advanced tracers example
# @dataclasses.dataclass
# class Tracer:
#     tracer: gtx.Field[[IDim, JDim], gtx.float32]
#     kind: int


# class TracerList:
#     def __init__(self, tracers: list[Tracer]):
#         self.tracers = tracers

#     def __gt_type__(self):
#         return ts.ListType(  # if reusing this makes sense
#             element_type=ts.NamedTupleType(
#                 types=[
#                     ts.FieldType(
#                         dims=[IDim, JDim],
#                         dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                     ),
#                     ts.ScalarType(kind=ts.ScalarKind.INT32),
#                 ],
#                 keys=["tracer", "kind"],
#             )
#         )


# @gtx.field_operator
# def select_tracer(
#     tracers: TracerList,
#     kind: int,
# ) -> gtx.Field[[IDim, JDim], gtx.float32]:
#     return TracerList(tracer for tracer in tracers if tracer.kind == kind)


# or can we avoid the loop-like construct (it's not a bad loop here, but adds a lot of syntax to the language)

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next.ffront import decorator
import dataclasses

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

# TODO(havogt): Since currently direct field_operator calls and program calls take different code paths the tests are duplicated.
# Remove the program calls, once field_operator calls go via `compiled_program` path.


@dataclasses.dataclass
class DataclassContainer:
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


class NamedTupleContainer(NamedTuple):
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


@gtx.field_operator
def constructed_outside_named_tuple(
    vel: NamedTupleContainer,
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def constructed_outside_named_tuple_program(
    vel: NamedTupleContainer,
    out: gtx.Field[[IDim, JDim], gtx.float32],
) -> None:
    constructed_outside_named_tuple(vel, out=out)


@gtx.field_operator
def constructed_outside_dataclass(
    vel: DataclassContainer,
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def constructed_outside_dataclass_program(
    vel: DataclassContainer,
    out: gtx.Field[[IDim, JDim], gtx.float32],
) -> None:
    constructed_outside_dataclass(vel, out=out)


@pytest.mark.parametrize(
    "testee",
    [
        constructed_outside_named_tuple,
        constructed_outside_named_tuple_program,
        constructed_outside_dataclass,
        constructed_outside_dataclass_program,
    ],
)
def test_named_tuple_like_constructed_outside(cartesian_case, testee):
    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(
        cartesian_case, testee, "out" if isinstance(testee, decorator.Program) else cases.RETURN
    )()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out=out,
        ref=(vel.u + vel.v),
    )


@gtx.field_operator
def constructed_inside_named_tuple(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
) -> NamedTupleContainer:
    return NamedTupleContainer(
        v=vel[0] - vel[1], u=vel[0] + vel[1]
    )  # order swapped to show kwargs work


@gtx.program
def constructed_inside_named_tuple_program(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    out: NamedTupleContainer,
):
    constructed_inside_named_tuple(vel, out=out)


@gtx.field_operator
def constructed_inside_dataclass(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
) -> DataclassContainer:
    return DataclassContainer(
        v=vel[0] - vel[1], u=vel[0] + vel[1]
    )  # order swapped to show kwargs work


@gtx.program
def constructed_inside_dataclass_program(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    out: DataclassContainer,
):
    constructed_inside_dataclass(vel, out=out)


@pytest.mark.parametrize(
    "testee",
    [
        constructed_inside_named_tuple,
        constructed_inside_named_tuple_program,
        constructed_inside_dataclass,
        constructed_inside_dataclass_program,
    ],
)
def test_named_tuple_like_constructed_inside(cartesian_case, testee):
    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(
        cartesian_case, testee, "out" if isinstance(testee, decorator.Program) else cases.RETURN
    )()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out=out,
        ref=out.__class__(u=vel[0] + vel[1], v=vel[0] - vel[1]),
    )


@dataclasses.dataclass
class NestedContainer:
    a: tuple[gtx.Field[[IDim, JDim], gtx.float32], NamedTupleContainer]
    b: tuple[DataclassContainer, gtx.Field[[IDim, JDim], gtx.float32]]


@gtx.field_operator
def nested_mixed_containers(
    inp: tuple[NestedContainer, gtx.Field[[IDim, JDim], gtx.float32]],
) -> tuple[gtx.Field[[IDim, JDim], gtx.float32], NestedContainer]:
    # swap the single fields and the elements of the containers between a and b
    old_dataclass = inp[0].b[0]
    old_named_tuple = inp[0].a[1]
    new_named_tuple = NamedTupleContainer(u=old_dataclass.u, v=old_dataclass.v)
    new_dataclass = DataclassContainer(u=old_named_tuple.u, v=old_named_tuple.v)
    new_nested_container = NestedContainer(
        a=(inp[0].b[1], new_named_tuple), b=(new_dataclass, inp[0].a[0])
    )
    return (inp[1], new_nested_container)


@gtx.program
def nested_mixed_containers_program(
    inp: tuple[NestedContainer, gtx.Field[[IDim, JDim], gtx.float32]],
    out: tuple[gtx.Field[[IDim, JDim], gtx.float32], NestedContainer],
) -> None:
    nested_mixed_containers(inp, out=out)


def test_nested_mixed_containers(cartesian_case):
    inp = cases.allocate(cartesian_case, nested_mixed_containers, "inp")()
    out = cases.allocate(
        cartesian_case,
        nested_mixed_containers,
        "out",
    )()

    cases.verify(
        cartesian_case,
        nested_mixed_containers,
        inp,
        out=out,
        ref=(
            inp[1],
            NestedContainer(
                a=(inp[0].b[1], NamedTupleContainer(u=inp[0].b[0].u, v=inp[0].b[0].v)),
                b=(DataclassContainer(u=inp[0].a[1].u, v=inp[0].a[1].v), inp[0].a[0]),
            ),
        ),
    )


@pytest.mark.xfail(
    reason="We store the qualified name to the actual Python type in the `NamedTupleType`."
)
def test_locally_defined_container(cartesian_case):
    # We could fix this pattern by storing the actual type (instead of a the qualified name).
    @dataclasses.dataclass
    class LocalContainer:  # not at global scope!
        u: gtx.Field[[IDim, JDim], gtx.float32]
        v: gtx.Field[[IDim, JDim], gtx.float32]

    @gtx.field_operator
    def testee(
        vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    ) -> DataclassContainer:
        return LocalContainer(v=vel[0] - vel[1], u=vel[0] + vel[1])

    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(
        cartesian_case, testee, "out" if isinstance(testee, decorator.Program) else cases.RETURN
    )()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out=out,
        ref=DataclassContainer(u=vel[0] + vel[1], v=vel[0] - vel[1]),
    )

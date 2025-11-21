# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import numpy as np
from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next.ffront import decorator
import dataclasses

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, KDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

# TODO(havogt): Since currently direct field_operator calls and program calls take different code paths the tests are duplicated.
# Remove the program calls, once field_operator calls go via `compiled_program` path.


@dataclasses.dataclass
class DataclassNamedCollection:
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


class NamedTupleNamedCollection(NamedTuple):
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]


@gtx.field_operator
def constructed_outside_named_tuple(
    vel: NamedTupleNamedCollection,
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def constructed_outside_named_tuple_program(
    vel: NamedTupleNamedCollection,
    out: gtx.Field[[IDim, JDim], gtx.float32],
):
    constructed_outside_named_tuple(vel, out=out)


@gtx.field_operator
def constructed_outside_dataclass(
    vel: DataclassNamedCollection,
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def constructed_outside_dataclass_program(
    vel: DataclassNamedCollection,
    out: gtx.Field[[IDim, JDim], gtx.float32],
):
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
) -> NamedTupleNamedCollection:
    return NamedTupleNamedCollection(
        v=vel[0] - vel[1], u=vel[0] + vel[1]
    )  # order swapped to show kwargs work


@gtx.program
def constructed_inside_named_tuple_program(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    out: NamedTupleNamedCollection,
):
    constructed_inside_named_tuple(vel, out=out)


@gtx.field_operator
def constructed_inside_dataclass(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
) -> DataclassNamedCollection:
    return DataclassNamedCollection(
        v=vel[0] - vel[1], u=vel[0] + vel[1]
    )  # order swapped to show kwargs work


@gtx.program
def constructed_inside_dataclass_program(
    vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    out: DataclassNamedCollection,
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
def test_named_collection_constructed_inside(cartesian_case, testee):
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
class NestedNamedCollection:
    a: tuple[gtx.Field[[IDim, JDim], gtx.float32], NamedTupleNamedCollection]
    b: tuple[DataclassNamedCollection, gtx.Field[[IDim, JDim], gtx.float32]]


@gtx.field_operator
def nested_mixed_named_collections(
    inp: tuple[NestedNamedCollection, gtx.Field[[IDim, JDim], gtx.float32]],
) -> tuple[gtx.Field[[IDim, JDim], gtx.float32], NestedNamedCollection]:
    # swap the single fields and the elements of the named collections between a and b
    old_dataclass = inp[0].b[0]
    old_named_tuple = inp[0].a[1]
    new_named_tuple = NamedTupleNamedCollection(u=old_dataclass.u, v=old_dataclass.v)
    new_dataclass = DataclassNamedCollection(u=old_named_tuple.u, v=old_named_tuple.v)
    new_nested_named_collection = NestedNamedCollection(
        a=(inp[0].b[1], new_named_tuple), b=(new_dataclass, inp[0].a[0])
    )
    return (inp[1], new_nested_named_collection)


@gtx.program
def nested_mixed_named_collections_program(
    inp: tuple[NestedNamedCollection, gtx.Field[[IDim, JDim], gtx.float32]],
    out: tuple[gtx.Field[[IDim, JDim], gtx.float32], NestedNamedCollection],
):
    nested_mixed_named_collections(inp, out=out)


def test_nested_mixed_named_collections(cartesian_case):
    inp = cases.allocate(cartesian_case, nested_mixed_named_collections_program, "inp")()
    out = cases.allocate(cartesian_case, nested_mixed_named_collections_program, "out")()

    cases.verify(
        cartesian_case,
        nested_mixed_named_collections,
        inp,
        out=out,
        ref=(
            inp[1],
            NestedNamedCollection(
                a=(inp[0].b[1], NamedTupleNamedCollection(u=inp[0].b[0].u, v=inp[0].b[0].v)),
                b=(DataclassNamedCollection(u=inp[0].a[1].u, v=inp[0].a[1].v), inp[0].a[0]),
            ),
        ),
    )


@dataclasses.dataclass
class StateDataclass:
    value: gtx.float32


class StateNamedTuple(NamedTuple):
    value: gtx.float32


@gtx.scan_operator(axis=cases.KDim, forward=True, init=StateDataclass(value=0.0))
def scan_dataclass(
    state: StateDataclass,
    inp: gtx.float32,
) -> StateDataclass:
    return StateDataclass(value=inp + state.value)


@gtx.field_operator
def scan_dataclass_wrapper(inp: gtx.Field[[KDim], gtx.float32]) -> gtx.Field[[KDim], gtx.float32]:
    # Note: `scan_dataclass(inp)` is of a (implicit) type `StateDataclass` with `gtx.float32` replaced by `gtx.Field[[...], gtx.float32]`.
    # Consequently, we need to extract the `value` field as we cannot properly annotate the return type.
    return scan_dataclass(inp).value


@gtx.scan_operator(axis=cases.KDim, forward=True, init=StateNamedTuple(value=0.0))
def scan_named_tuple(
    state: StateNamedTuple,
    inp: gtx.float32,
) -> StateNamedTuple:
    return StateNamedTuple(value=inp + state.value)


@gtx.field_operator
def scan_named_tuple_wrapper(inp: gtx.Field[[KDim], gtx.float32]) -> gtx.Field[[KDim], gtx.float32]:
    # Note: `scan_named_tuple(inp)` is of a (implicit) type `StateNamedTuple` with `gtx.float32` replaced by `gtx.Field[[...], gtx.float32]`.
    # Consequently, we need to extract the `value` field as we cannot properly annotate the return type.
    return scan_named_tuple(inp).value


@pytest.mark.parametrize(
    "testee",
    [scan_named_tuple_wrapper, scan_dataclass_wrapper],
)
def test_scan(cartesian_case, testee):
    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        out=out,
        ref=np.cumsum(inp.asnumpy(), axis=0),
    )


@pytest.mark.xfail(
    reason="We store the qualified name to the actual Python type in the `NamedTupleType`."
)
def test_locally_defined_named_collection(cartesian_case):
    # We could fix this pattern by storing the actual type (instead of a the qualified name).
    @dataclasses.dataclass
    class LocalNamedCollection:  # not at global scope!
        u: gtx.Field[[IDim, JDim], gtx.float32]
        v: gtx.Field[[IDim, JDim], gtx.float32]

    @gtx.field_operator
    def testee(
        vel: tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]],
    ) -> DataclassNamedCollection:
        return LocalNamedCollection(v=vel[0] - vel[1], u=vel[0] + vel[1])

    vel = cases.allocate(cartesian_case, testee, "vel")()
    out = cases.allocate(
        cartesian_case, testee, "out" if isinstance(testee, decorator.Program) else cases.RETURN
    )()

    cases.verify(
        cartesian_case,
        testee,
        vel,
        out=out,
        ref=DataclassNamedCollection(u=vel[0] + vel[1], v=vel[0] - vel[1]),
    )

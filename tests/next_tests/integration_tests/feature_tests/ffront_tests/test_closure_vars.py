# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)


def test_constant_closure_vars_with_frozen_namespace(cartesian_case):
    from gt4py.eve.utils import FrozenNamespace

    constants = FrozenNamespace(PI=np.float64(3.142), E=np.float64(2.718))

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return constants.PI * constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: constants.PI * constants.E * input
    )


def test_constant_closure_vars_with_enums(cartesian_case):
    import enum

    class Constants(np.float64, enum.Enum):
        PI = 3.142
        E = 2.718

    @gtx.field_operator
    def consume_constants(input: cases.IFloatField) -> cases.IFloatField:
        return Constants.PI * Constants.E * input

    cases.verify_with_default_data(
        cartesian_case, consume_constants, ref=lambda input: Constants.PI * Constants.E * input
    )


def test_constant_closure_vars_in_program(cartesian_case):
    import enum

    from gt4py.eve.utils import FrozenNamespace

    class Bounds(enum.IntEnum):
        LOWER = 1
        UPPER = 3

    constants = FrozenNamespace(factor=np.int32(2), inner=FrozenNamespace(offset=np.int32(1)))

    @gtx.field_operator
    def scale(a: cases.IField, factor: gtx.int32) -> cases.IField:
        return factor * a

    @gtx.program
    def testee(a: cases.IField, out: cases.IField):
        scale(
            a,
            constants.factor,
            out=out,
            domain={IDim: (Bounds.LOWER, Bounds.UPPER + constants.inner.offset)},
        )

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, "out").zeros()()
    ref = np.zeros_like(a.asnumpy())
    ref[1:4] = 2 * a.asnumpy()[1:4]

    cases.verify(cartesian_case, testee, a, out=out, ref=ref)
